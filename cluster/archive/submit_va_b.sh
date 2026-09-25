#!/bin/bash
# MIDIAN at b in {1,5} for condensed A / B. Resumable via logs/launch_va_b.txt.
. "$(dirname "$(readlink -f "$0")")/../env.sh"
need RTE_DATA RTE_ACCOUNT
D=$RTE_DATA; LOG=$D/logs/launch_va_b.txt; touch $LOG; cd "$RTE_REPO"
sub() {  # key partition cpus mem time workers grid args...
  local key=$1 p=$2 c=$3 m=$4 t=$5 w=$6 g=$7; shift 7
  grep -qF " $key" $LOG && return
  while :; do
    j=$(sbatch --parsable -p $p -A "$RTE_ACCOUNT" -c $c --mem=$m -t $t -J rte_${g}__midian -o $D/logs/units/%x-%j.out -e $D/logs/units/%x-%j.err \
        --export=ALL,RTE_DATA=$D,RTE_PYTHON=$D/env/rte/bin/python,RTE_WORKERS=$w,RTE_CONSOLIDATE=0 cluster/slurm/run_grid.sbatch $g --methods midian "$@" 2>&1)
    case "$j" in *QOSMax*) sleep 120;; ''|*[!0-9]*) echo "FAIL $key $j" >&2; return;; *) echo "$j $key" >> $LOG; return;; esac
  done
}
for s in 1 2 3; do sub n100k_$s "$RTE_CPU_PARTITIONS" 1 128G 1-12:00:00 1 va_b_n100k --seeds $s; done
for s in 1 2 3; do sub n10k_$s "$RTE_CPU_PARTITIONS" 1 64G 10:00:00 1 va_b_n10k --seeds $s; done
for s in $(seq 1 10); do sub n1000_$s "$RTE_CPU_PARTITIONS" 1 40G 4:00:00 1 va_b_n1000 --seeds $s; done
for s in $(seq 1 10); do sub n100_$s "$RTE_CPU_PARTITIONS" 1 40G 3:00:00 1 va_b_n100 --seeds $s; done
for s in 1 2 3; do sub re5k_$s "$RTE_CPU_PARTITIONS" 1 32G 2:00:00 1 va_b_routereval5k --seeds $s; done
for s in 1 2 3 4 5; do sub lrb_$s "$RTE_CPU_PARTITIONS" 1 16G 1:00:00 1 va_b_llmrouterbench --seeds $s; done
for beta in 0.0 0.5; do sub bern_$beta "$RTE_CPU_PARTITIONS" 4 256G 12:00:00 4 va_b_bernoulli_1e7 --only dist=specialist,beta=$beta,liar_select=low_skill_first --seeds 1-100; done
for d in specialist heavy_tail bimodal; do for beta in 0.0 0.5; do sub rep_${d}_$beta "$RTE_CPU_PARTITIONS" 4 128G 8:00:00 4 va_b_replay_1e6 --only dist=$d,beta=$beta,liar_select=low_skill_first --seeds 1-100; done; done
echo "VA_B_SUBMITTED $(wc -l < $LOG)"
