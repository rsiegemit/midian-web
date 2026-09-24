#!/bin/bash
# Erratum-30 reruns (configs/grid.yaml *_cal / *_split_cal / *_norep_cal, non-framework grids), --nice 100000: behind the focus
# work, ahead of linucb_fix_* (1e6). Resumable via logs/launch_erratum30.txt. Framework grids: scripts/launch_units.sh.
D=/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte; LOG=$D/logs/launch_erratum30.txt; touch $LOG; cd /n/home02/rsiegelmann/rte
sub() {  # key partition cpus mem time workers grid args...
  local key=$1 p=$2 c=$3 m=$4 t=$5 w=$6 g=$7; shift 7
  grep -qF " $key" $LOG && return
  while :; do
    j=$(sbatch --parsable --nice=100000 -p $p -A sompolinsky_lab -c $c --mem=$m -t $t -J rte_$g -o $D/logs/units/%x-%j.out -e $D/logs/units/%x-%j.err \
        --export=ALL,RTE_DATA=$D,RTE_PYTHON=$D/env/rte/bin/python,RTE_WORKERS=$w,RTE_CONSOLIDATE=0 scripts/run_grid.sbatch $g "$@" 2>&1)
    case "$j" in *QOSMax*) sleep 120;; ''|*[!0-9]*) echo "FAIL $key $j" >&2; return;; *) echo "$j $key" >> $LOG; return;; esac
  done
}
sub re5k_cal sapphire,serial_requeue 4 64G 2:00:00 4 routereval5k_cal
sub lrb_cal sapphire,serial_requeue 4 32G 1:00:00 4 llmrouterbench_cal
sub mmlu_norep sapphire,serial_requeue 4 32G 2:00:00 4 routereval_mmlu_norep_cal
sub lrb_norep sapphire,serial_requeue 4 64G 6:00:00 4 llmrouterbench_norep_cal
sub re5k_norep_fast sapphire,serial_requeue 4 64G 6:00:00 4 routereval5k_norep_cal \
    --methods midian_va,midian,flat_probe_argmax,flat_nsw_router,cluster_head_router,disrouter_cascade,ucb_per_family,thompson_per_family,warm_start_bandit,linucb_honest,trueskill_per_family,declared_argmax,random
for beta in 0.0 0.5; do for s in 1 2 3; do for b in 1 3 5; do   # kNN: ~4.1 h per row single-threaded; one job per (regime, seed, b), 8 threads
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 sub re5k_norep_knn_${beta}_${s}_b$b sapphire,shared 8 48G 12:00:00 1 routereval5k_norep_cal --methods knn_router --only beta=$beta,b=$b --seeds $s; done; done; done
for d in specialist heavy_tail bimodal; do for beta in 0.0 0.5; do for s in 1-15 16-30; do
  sub rep_${d}_${beta}_$s sapphire 4 128G 12:00:00 4 replay_1e6_split_cal --only dist=$d,beta=$beta,liar_select=low_skill_first --seeds $s; done; done; done
for beta in 0.0 0.5; do for lo in $(seq 1 10 91); do
  sub bern_${beta}_s$lo sapphire 4 256G 12:00:00 4 bernoulli_1e7_cal --only dist=specialist,beta=$beta,liar_select=low_skill_first --seeds $lo-$((lo+9)); done; done
echo "ERRATUM30_SUBMITTED $(wc -l < $LOG)"
