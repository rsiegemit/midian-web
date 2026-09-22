#!/bin/bash
# Rerun exactly the units listed in results/quarantine_units.tsv (from quarantine_fallback_rows.py --apply).
# Same sizing as scripts/launch_units.sh (OPS_RULES R1-R2, H3-H4): 1 CPU, measured walltime, resumable, QOS-retry.
# rte.run recomputes only the rids that are absent, so a unit whose other arms survived redoes just the quarantined one.
export RTE_DATA="${RTE_DATA:-/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte}"
PY="$RTE_DATA/env/rte/bin/python"; TSV="$RTE_DATA/results/quarantine_units.tsv"; LOG="$RTE_DATA/logs/rerun_quarantine.txt"
cd "$(dirname "$0")/../.."; export PYTHONPATH="$PWD"; mkdir -p "$RTE_DATA/logs/units"; touch "$LOG"
"$PY" - "$TSV" <<'PY' > "$LOG.plan"
import sys; sys.path.insert(0, "scripts")
import pandas as pd
from job_time import minutes, slurm
for r in pd.read_csv(sys.argv[1], sep="\t").itertuples():
    print(f"{r.grid}\t{r.method}\tdist={r.dist},beta={float(r.beta)},liar_select={r.liar_select}\t{int(r.seed)}\t{slurm(minutes(r.grid, r.method))}")
PY
sub=0
HB="$RTE_DATA/logs/rerun_submitter.heartbeat"          # campaign_tick starts a submitter only if this goes stale
while IFS=$'\t' read -r g m only seed t; do
  touch "$HB"; key="$g|$m|$only|$seed"; grep -qF "$key" "$LOG" && continue
  while :; do
    jid=$(sbatch --parsable -p sapphire,serial_requeue -A sompolinsky_lab -c 1 --mem=32G --time="$t" \
      --job-name="rte_${g}__${m}" -o "$RTE_DATA/logs/units/%x-%j.out" -e "$RTE_DATA/logs/units/%x-%j.err" \
      --export=ALL,RTE_PYTHON="$PY",RTE_WORKERS=1,RTE_CONSOLIDATE=0 \
      scripts/run_grid.sbatch "$g" --methods "$m" --only "$only" --seeds "$seed" 2>&1)
    case "$jid" in *QOSMax*) touch "$HB"; sleep 300;; ''|*[!0-9]*) echo "FAIL $key :: $jid" >&2; break;; *) echo "$jid $key" >> "$LOG"; sub=$((sub+1)); break;; esac
  done
done < "$LOG.plan"
echo "SUBMITTED $sub of $(wc -l < "$LOG.plan")"
