#!/bin/bash
# Cancel every running/pending live-grid job and resubmit all seeds (rows already written are kept: the runner resumes).
# Slow by design: SLURM throttles scancel/sbatch RPCs (~1 s each) -> run detached: nohup scripts/restart_wave.sh &
set -uo pipefail
cd "${RTE_REPO:-$HOME/rte}"; L=/scratch/rte/logs; T=$(date +%H%M)
ids=$(squeue -u $USER -h -o "%i %j" | grep "rte_.*__" | awk '{print $1}')
echo "[$(date +%T)] cancelling $(echo $ids | wc -w) live jobs"; scancel $ids 2>/dev/null; sleep 10
for s in 1 2 3 4 5; do RTE_RUN_ARGS="--seeds $s" scripts/launch_live.sh > "$L/launch_live_wave_${T}_seed${s}.txt" 2>&1; echo "[$(date +%T)] seed $s: $(grep -c . "$L/launch_live_wave_${T}_seed${s}.txt") submitted"; done
echo "[$(date +%T)] DONE: $(squeue -u $USER -h -o '%j' | grep -c 'rte_.*__') live jobs queued"
