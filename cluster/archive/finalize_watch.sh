#!/bin/bash
# Two-stage finalize (OPS_RULES D4). Stage 1 regenerates as soon as the HEADLINE + control reruns and lietext_th drain,
# so writing is not gated on the low-priority SOTA 4-arm ablation; stage 2 regenerates again once that drains too.
# Waits on explicit job IDs from the launcher logs, never name patterns.
. "$(dirname "$(readlink -f "$0")")/../env.sh"
need RTE_DATA RTE_ACCOUNT
cd "$RTE_REPO"
ABL='fw_live_n(1000|100)(_lowskill)?_sota\|'
while ps -p 1410672 >/dev/null 2>&1; do sleep 300; done                    # second rerun pass still submitting
ids() { cat $RTE_DATA/logs/rerun_quarantine.txt $RTE_DATA/logs/launch_lietext_th.txt 2>/dev/null | grep -Ev "$1" | awk '{print $1}' | sort -u; }
wait_for() {  # $1 = regex of grids to EXCLUDE ('^$' excludes nothing)
  while :; do
    live=$(comm -12 <(ids "$1") <(squeue -u "$USER" -h -o "%i" | sort -u) | wc -l)
    echo "$(date -Is) [$2] $live jobs still queued or running"; [ "$live" -eq 0 ] && return; sleep 900
  done
}
refresh() {
  $RTE_DATA/env/rte/bin/python cluster/ops/build_job_sizing.py
  sbatch --parsable -p "$RTE_TEST_PARTITION" -A "$RTE_ACCOUNT" -c 8 --mem=96G -t 4:00:00 -J "finalize_$1" \
    -o $RTE_DATA/logs/finalize_$1.log -e $RTE_DATA/logs/finalize_$1.err $RTE_DATA/jobs/finalize_refresh.sh
  echo "$(date -Is) finalize_$1 submitted"
}
wait_for "$ABL" stage1-headline; refresh stage1
wait_for '^$'   stage2-all;      refresh stage2
