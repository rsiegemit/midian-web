#!/bin/bash
# Login-node half: wait until every rerun + lietext_th job has drained (job IDs from the launcher logs -- explicit IDs,
# never name patterns), rebuild the sizing table (R3), then submit the compute-node refresh.
export RTE_DATA=/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte; cd /n/home02/rsiegelmann/rte
while ps -p 1410672 >/dev/null 2>&1; do sleep 300; done                     # second rerun pass still submitting
ids() { cat $RTE_DATA/logs/rerun_quarantine.txt $RTE_DATA/logs/launch_lietext_th.txt 2>/dev/null | awk '{print $1}' | sort -u; }
while :; do
  live=$(comm -12 <(ids) <(squeue -u "$USER" -h -o "%i" | sort -u) | wc -l)
  echo "$(date -Is) $live rerun/lietext jobs still queued or running"; [ "$live" -eq 0 ] && break; sleep 900
done
$RTE_DATA/env/rte/bin/python scripts/ops/build_job_sizing.py
sbatch --parsable -p test -A sompolinsky_lab -c 8 --mem=96G -t 4:00:00 -J finalize_refresh \
  -o $RTE_DATA/logs/finalize_refresh.log -e $RTE_DATA/logs/finalize_refresh.err $RTE_DATA/jobs/finalize_refresh.sh
echo "finalize_refresh submitted"
