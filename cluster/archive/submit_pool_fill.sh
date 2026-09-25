#!/bin/bash
# Pool-fill arms (configs/grid.yaml pool_fill_*) on the non-live backends. Resumable via logs/launch_pool_fill.txt.
# Live pool_fill_n* units go through the pack driver plan instead (one rte.run per unit, LLM memo per process).
D=/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte; LOG=$D/logs/launch_pool_fill.txt; touch $LOG; cd /n/home02/rsiegelmann/rte
sub() {  # key partition cpus mem time workers grid args...
  local key=$1 p=$2 c=$3 m=$4 t=$5 w=$6 g=$7; shift 7
  grep -qF " $key" $LOG && return
  while :; do
    j=$(sbatch --parsable -p $p -A sompolinsky_lab -c $c --mem=$m -t $t -J rte_$g -o $D/logs/units/%x-%j.out -e $D/logs/units/%x-%j.err \
        --export=ALL,RTE_DATA=$D,RTE_PYTHON=$D/env/rte/bin/python,RTE_WORKERS=$w,RTE_CONSOLIDATE=0 scripts/run_grid.sbatch $g "$@" 2>&1)
    case "$j" in *QOSMax*) sleep 120;; ''|*[!0-9]*) echo "FAIL $key $j" >&2; return;; *) echo "$j $key" >> $LOG; return;; esac
  done
}
sub re5k sapphire,serial_requeue 4 64G 4:00:00 4 pool_fill_routereval5k
sub lrb sapphire,serial_requeue 4 32G 2:00:00 4 pool_fill_llmrouterbench
for beta in 0.0 0.5; do sub bern_$beta sapphire 4 256G 12:00:00 4 pool_fill_bernoulli_1e7 --only dist=specialist,beta=$beta,liar_select=low_skill_first; done
for d in specialist heavy_tail bimodal; do for beta in 0.0 0.5; do sub rep_${d}_$beta sapphire 4 128G 8:00:00 4 pool_fill_replay_1e6 --only dist=$d,beta=$beta,liar_select=low_skill_first; done; done
echo "POOL_FILL_SUBMITTED $(wc -l < $LOG)"
