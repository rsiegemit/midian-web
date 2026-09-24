#!/bin/bash
# THE launcher for llm-backend grids: one unit per job, sized from measurement, resumable. Use this, never a copy.
#   [RTE_FW_PARALLEL=8 MEM=40G] [TODO=1] scripts/launch_units.sh <grid> [--nice N]     (parallel framework requests need the 40 G)
#   TODO=1: plan only the units with a row still missing (the runner's own rid check), not every unit of the grid
# Enforces OPS_RULES.md:
#   R1  units come from the GRID LOADER (rte.run), never a hand-written filter list (a hand list dropped a regime once)
#   R2  1 CPU: a unit only waits on HTTP to the fleet (2 CPUs ran at ~5% efficiency)
#   R3  walltime = scripts/job_time.py (1.5 x measured p95 per size x framework), never a flat multi-day limit
#   R4  resumable: the log IS the state; re-run to continue, never start over with a fresh log
#   R5  on QOSMaxSubmitJobPerUserLimit wait and retry the SAME unit, never drop it
set -uo pipefail
G=${1:?usage: launch_units.sh <grid> [--nice N]}; NICE=0; [ "${2:-}" = "--nice" ] && NICE=${3:?}
export RTE_DATA="${RTE_DATA:-/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte}"
PY="$RTE_DATA/env/rte/bin/python"; LOG="$RTE_DATA/logs/launch_${G}.txt"; mkdir -p "$RTE_DATA/logs/units"; touch "$LOG"
cd "$(dirname "$0")/.."; export PYTHONPATH="$PWD"
"$PY" - "$G" <<'PY' > "$LOG.plan"
import os, sys, yaml
sys.path.insert(0, "scripts")
from rte.run import RTE_DATA, blocks, cells, method_specs, row_id, seeds
from job_time import minutes, slurm
g = sys.argv[1]; cfg = yaml.safe_load(open("configs/grid.yaml")); out = f"{RTE_DATA}/results/{g}"
have = None
if os.environ.get("TODO"):                                   # the runner's done-set: rows.d filenames + rows.csv rids
    have = {f[:-5] for f in os.listdir(f"{out}/rows.d")} if os.path.isdir(f"{out}/rows.d") else set()
    if os.path.exists(f"{out}/rows.csv"):
        import pandas as pd; have |= set(pd.read_csv(f"{out}/rows.csv", usecols=["rid"]).rid.dropna())
plan = {}
for blk in blocks(cfg, g):
    specs = method_specs(blk)
    for cell in cells(blk):
        only = ",".join(f"{a}={cell[a]}" for a in ("dist", "beta", "liar_select"))
        for seed in seeds(blk["seeds"]):
            for s in specs:
                if have is None or row_id(cell, s["name"], s["params"], seed) not in have: plan[(s["name"], only, seed)] = 1
for m, only, seed in sorted(plan, key=lambda k: (k[1], k[2], k[0])): print(f"{m}\t{only}\t{seed}\t{slurm(minutes(g, m))}")
PY
echo "$G: $(wc -l < "$LOG.plan") units, $(wc -l < "$LOG") already submitted"
sub=0
while IFS=$'\t' read -r m only seed t; do
  key="$m|$only|$seed"; grep -qF "$key" "$LOG" && continue
  while :; do
    jid=$(sbatch --parsable -p sapphire,serial_requeue -A sompolinsky_lab --nice="$NICE" -c 1 --mem="${MEM:-32G}" --time="$t" \
      --job-name="rte_${G}__${m}" -o "$RTE_DATA/logs/units/%x-%j.out" -e "$RTE_DATA/logs/units/%x-%j.err" \
      --export=ALL,RTE_PYTHON="$PY",RTE_WORKERS=1,RTE_CONSOLIDATE=0,RTE_FW_PARALLEL="${RTE_FW_PARALLEL:-1}" \
      scripts/run_grid.sbatch "$G" --methods "$m" --only "$only" --seeds "$seed" 2>&1)
    case "$jid" in
      *QOSMax*) sleep 300;;
      ''|*[!0-9]*) echo "FAIL $key :: $jid" >&2; break;;
      *) echo "$jid $key" >> "$LOG"; sub=$((sub+1)); break;;
    esac
  done
done < "$LOG.plan"
echo "$G: SUBMITTED $sub"
