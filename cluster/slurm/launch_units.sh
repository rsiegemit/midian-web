#!/bin/bash
# THE launcher for llm-backend grids: one unit per job, sized from measurement, resumable. Use this, never a copy.
#   [RTE_FW_PARALLEL=8 MEM=40G] [ONLY_MISSING=1] cluster/slurm/launch_units.sh <grid> [--nice N]   (parallel framework requests need the 40 G)
#   ONLY_MISSING=1: plan only the units with a row still missing (the runner's own rid check), not every unit of the grid
#   TIME=D-HH:MM:SS: one walltime for every unit, for grids job_time.py has no measurement for (1.5 x the source grid's p95)
#   UNITS=<ERE>: submit only the planned units whose "method|only|seed" key matches (staging, e.g. a probe warm-up first);
#     a later run without UNITS submits the rest (the log is the state)
#   LOCAL=P: no SLURM -- run the units on this machine, at most P at once (memory, not CPUs, sets P: MEM each), each
#     under flock (never two copies of one unit), a MemoryMax=MEM systemd scope and `timeout` of the planned walltime;
#     the log gets "local-rc=<exit> <key>" per finished unit and a re-run retries every key without local-rc=0
#   Needs RTE_DATA, RTE_ACCOUNT, RTE_CPU_PARTITIONS (cluster/cluster.env.example). scripts/launch_units.sh is a symlink here.
# Enforces OPS_RULES.md:
#   R1  units come from the GRID LOADER (midian.run), never a hand-written filter list (a hand list dropped a regime once)
#   R2  1 CPU: a unit only waits on HTTP to the fleet (2 CPUs ran at ~5% efficiency)
#   R3  walltime = cluster/slurm/job_time.py (1.5 x measured p95 per size x framework), never a flat multi-day limit
#   R4  resumable: the log IS the state; re-run to continue, never start over with a fresh log
#   R5  on QOSMaxSubmitJobPerUserLimit wait and retry the SAME unit, never drop it
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/../env.sh"
need RTE_DATA; [ -n "${LOCAL:-}" ] || need RTE_ACCOUNT RTE_CPU_PARTITIONS
[[ -z "${LOCAL:-}" || $LOCAL =~ ^[1-9][0-9]*$ ]] || { echo "LOCAL must be a positive integer (units at once)" >&2; exit 2; }
if [ -n "${TODO:-}" ]; then echo "TODO=1 was renamed ONLY_MISSING=1 (refusing to plan every unit)" >&2; exit 2; fi
G=${1:?usage: launch_units.sh <grid> [--nice N]}; NICE=0; [ "${2:-}" = "--nice" ] && NICE=${3:?}
PY="$RTE_DATA/env/rte/bin/python"; LOG="$RTE_DATA/logs/launch_${G}.txt"; mkdir -p "$RTE_DATA/logs/units"; touch "$LOG"
cd "$(dirname "$(readlink -f "$0")")/../.."; export PYTHONPATH="$PWD"
"$PY" - "$G" <<'PY' > "$LOG.plan"
import os, sys
from midian.run import RTE_DATA, blocks, cells, method_specs, row_id, seeds
from cluster.slurm.job_time import minutes, slurm
from scripts.figures.lib.grids import config
g = sys.argv[1]; cfg = config(); out = f"{RTE_DATA}/results/{g}"
have = None
if os.environ.get("ONLY_MISSING"):                                   # the runner's done-set: rows.d filenames + rows.csv rids
    have = {f[:-5] for f in os.listdir(f"{out}/rows.d")} if os.path.isdir(f"{out}/rows.d") else set()
    if os.path.exists(f"{out}/rows.csv"):
        import pandas as pd; have |= set(pd.read_csv(f"{out}/rows.csv", usecols=["rid"]).rid.dropna())
plan = {}
# a grid that varies lie_mode (configs/grids/review.yaml) gets one unit per lie mode; other grids keep their old keys
axes = ("dist", "beta", "liar_select") + (("lie_mode",) if len({c["lie_mode"] for b in blocks(cfg, g) for c in cells(b)}) > 1 else ())
for blk in blocks(cfg, g):
    specs = method_specs(blk)
    for cell in cells(blk):
        only = ",".join(f"{a}={cell[a]}" for a in axes)
        for seed in seeds(blk["seeds"]):
            for s in specs:
                if have is None or row_id(cell, s["name"], s["params"], seed) not in have: plan[(s["name"], only, seed)] = 1
for m, only, seed in sorted(plan, key=lambda k: (k[1], k[2], k[0])): print(f"{m}\t{only}\t{seed}\t{slurm(minutes(g, m))}")
PY
echo "$G: $(wc -l < "$LOG.plan") units, $(wc -l < "$LOG") already submitted"
run_local() {   # <key> <method> <only> <seed> <D-HH:MM:SS>
  local d=0 t=$5 h mm s; [[ $t == *-* ]] && { d=${t%%-*}; t=${t#*-}; }; IFS=: read -r h mm s <<< "$t"
  flock -n "$RTE_DATA/logs/units/${G}__$(md5sum <<< "$1" | cut -c1-16).lock" \
    systemd-run --user --scope -q -p MemoryMax="${MEM:-32G}" -p MemorySwapMax=0 \
    env RTE_PYTHON="$PY" RTE_WORKERS=1 RTE_CONSOLIDATE=0 RTE_FW_PARALLEL="${RTE_FW_PARALLEL:-1}" \
    timeout -k 30 $((d*86400 + 10#$h*3600 + 10#$mm*60 + 10#$s)) bash cluster/slurm/run_grid.sbatch "$G" --methods "$2" --only "$3" --seeds "$4" \
    < /dev/null > "$RTE_DATA/logs/units/rte_${G}__$2-local$BASHPID.out" 2>&1
  echo "local-rc=$? $1" >> "$LOG"
}
sub=0
while IFS=$'\t' read -r m only seed t; do
  key="$m|$only|$seed"
  if [ -n "${LOCAL:-}" ]; then grep -qxF "local-rc=0 $key" "$LOG" && continue; else grep -qF "$key" "$LOG" && continue; fi
  [ -n "${UNITS:-}" ] && ! grep -qE "$UNITS" <<< "$key" && continue
  if [ -n "${LOCAL:-}" ]; then
    while [ "$(jobs -rp | wc -l)" -ge "$LOCAL" ]; do wait -n; done
    run_local "$key" "$m" "$only" "$seed" "${TIME:-$t}" & sub=$((sub+1)); continue
  fi
  while :; do
    jid=$(sbatch --parsable -p "$RTE_CPU_PARTITIONS" -A "$RTE_ACCOUNT" --nice="$NICE" -c 1 --mem="${MEM:-32G}" --time="${TIME:-$t}" \
      --job-name="rte_${G}__${m}" -o "$RTE_DATA/logs/units/%x-%j.out" -e "$RTE_DATA/logs/units/%x-%j.err" \
      --export=ALL,RTE_PYTHON="$PY",RTE_WORKERS=1,RTE_CONSOLIDATE=0,RTE_FW_PARALLEL="${RTE_FW_PARALLEL:-1}" \
      cluster/slurm/run_grid.sbatch "$G" --methods "$m" --only "$only" --seeds "$seed" 2>&1)
    case "$jid" in
      *QOSMax*) sleep 300;;
      ''|*[!0-9]*) echo "FAIL $key :: $jid" >&2; break;;
      *) echo "$jid $key" >> "$LOG"; sub=$((sub+1)); break;;
    esac
  done
done < "$LOG.plan"
wait
echo "$G: SUBMITTED $sub"
