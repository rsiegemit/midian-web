#!/bin/bash
# Run a command once every SLURM job whose name starts with PREFIX has left the queue.
#
#   scripts/run_after.sh <job-name-prefix> <command...>
#   scripts/run_after.sh n100k2_ sbatch scripts/run_grid.sbatch cohort_rte
#
# Generalises the several one-off "wait for stage N, then launch stage N+1" wrappers. Polls every 10 min, which is
# cheap against squeue and fine for multi-hour stages. Requires the prefix to be non-empty so a typo cannot match all
# jobs and fire immediately.
set -uo pipefail
PREFIX="${1:?usage: run_after.sh <job-name-prefix> <command...>}"; shift
[ $# -gt 0 ] || { echo "no command given" >&2; exit 64; }
while true; do
  N=$(squeue -u "$USER" -h -o "%j" | grep -c "^$PREFIX") || N=0
  [ "$N" -eq 0 ] && break
  echo "[run_after] $(date +%H:%M) waiting on $N job(s) matching '$PREFIX*'"
  sleep 600
done
echo "[run_after] $(date +%H:%M) '$PREFIX*' drained; running: $*"
exec "$@"
