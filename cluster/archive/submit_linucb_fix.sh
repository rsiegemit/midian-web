#!/bin/bash
# Fixed-bonus LinUCB (grids linucb_fix_*), queued BEHIND everything else (--nice 1000000). Resumable via logs/launch_linucb_fix.txt.
. "$(dirname "$(readlink -f "$0")")/../env.sh"
need RTE_DATA RTE_ACCOUNT
D=$RTE_DATA; LOG=$D/logs/launch_linucb_fix.txt; touch $LOG; cd "$RTE_REPO"
sub() {  # key partition cpus mem time workers grid args...
  local key=$1 p=$2 c=$3 m=$4 t=$5 w=$6 g=$7; shift 7
  grep -qF " $key" $LOG && return
  j=$(sbatch --parsable --nice=10000 -p $p -A "$RTE_ACCOUNT" -c $c --mem=$m -t $t -J rte_${g}__linucb_honest -o $D/logs/units/%x-%j.out -e $D/logs/units/%x-%j.err \
      --export=ALL,RTE_DATA=$D,RTE_PYTHON=$D/env/rte/bin/python,RTE_WORKERS=$w,RTE_CONSOLIDATE=0 scripts/run_grid.sbatch $g --methods linucb_honest "$@" 2>&1)
  case "$j" in ''|*[!0-9]*) echo "FAIL $key $j" >&2;; *) echo "$j $key" >> $LOG;; esac
}
for s in $(seq 1 10); do sub n100_$s "$RTE_CPU_PARTITIONS" 1 40G 6:00:00 1 linucb_fix_n100 --seeds $s; sub n1000_$s "$RTE_CPU_PARTITIONS" 1 40G 8:00:00 1 linucb_fix_n1000 --seeds $s; done
for s in 1 2 3; do sub n10k_$s "$RTE_CPU_PARTITIONS" 1 64G 12:00:00 1 linucb_fix_n10k --seeds $s; done
for s in 1 2 3; do for b in 1 3 5; do sub n100k_${s}_b$b "$RTE_CPU_PARTITIONS" 1 128G 1-00:00:00 1 linucb_fix_n100k --seeds $s --only b=$b; done; done
sub re5k "$RTE_CPU_PARTITIONS" 4 64G 8:00:00 4 linucb_fix_routereval5k
sub lrb "$RTE_CPU_PARTITIONS" 4 32G 4:00:00 4 linucb_fix_llmrouterbench
for beta in 0.0 0.5; do for lo in $(seq 1 5 96); do sub bern_${beta}_s${lo}_5 "$RTE_CPU_PARTITIONS" 4 256G 8:00:00 4 linucb_fix_bernoulli_1e7 --only dist=specialist,beta=$beta,liar_select=low_skill_first --seeds $lo-$((lo+4)); done; done
echo "LINUCB_FIX_SUBMITTED $(wc -l < $LOG)"
