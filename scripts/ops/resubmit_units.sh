#!/bin/bash
# Resubmit an explicit unit list with the parallel framework path (RTE_FW_PARALLEL, FrameworkMethod.prefetch).
#   scripts/ops/resubmit_units.sh <plan.tsv> <log>      plan lines: grid <TAB> method <TAB> only <TAB> seed <TAB> nice
# Same sizing rules as launch_units.sh (OPS_RULES R1-R2, H3-H4): 1 CPU, job_time walltime, resumable log, QOS retry.
# 40 G, not 32 G: a unit's main process sits at ~29.8 G and the 8 framework workers add their own interpreters.
export RTE_DATA="${RTE_DATA:-/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte}"
PLAN=${1:?plan}; LOG=${2:?log}; PAR=${RTE_FW_PARALLEL:-8}
PY="$RTE_DATA/env/rte/bin/python"; cd "$(dirname "$0")/../.."; export PYTHONPATH="$PWD"; touch "$LOG"
HB="$LOG.heartbeat"; sub=0
while IFS=$'\t' read -r g m only seed nice; do
  touch "$HB"; key="$g|$m|$only|$seed"; grep -qF "$key" "$LOG" && continue
  t=$("$PY" -c "import sys; sys.path.insert(0,'scripts'); from job_time import minutes, slurm; print(slurm(minutes('$g','$m')))")
  while :; do
    jid=$(sbatch --parsable -p sapphire,serial_requeue -A sompolinsky_lab --nice="$nice" -c 1 --mem=40G --time="$t" \
      --job-name="rte_${g}__${m}" -o "$RTE_DATA/logs/units/%x-%j.out" -e "$RTE_DATA/logs/units/%x-%j.err" \
      --export=ALL,RTE_PYTHON="$PY",RTE_WORKERS=1,RTE_CONSOLIDATE=0,RTE_FW_PARALLEL="$PAR" \
      scripts/run_grid.sbatch "$g" --methods "$m" --only "$only" --seeds "$seed" 2>&1)
    case "$jid" in *QOSMax*) touch "$HB"; sleep 300;; ''|*[!0-9]*) echo "FAIL $key :: $jid" >&2; break;; *) echo "$jid $key" >> "$LOG"; sub=$((sub+1)); break;; esac
  done
done < "$PLAN"
echo "SUBMITTED $sub of $(wc -l < "$PLAN")"
