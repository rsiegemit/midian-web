#!/bin/bash
# TrueSkill after the probe-duplicate fix (grids trueskill_fix_*), nice 10000. Resumable via logs/launch_trueskill_fix.txt.
D=/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte; LOG=$D/logs/launch_trueskill_fix.txt; touch $LOG; cd /n/home02/rsiegelmann/rte
sub() {  # key partition cpus mem time workers grid args...
  local key=$1 p=$2 c=$3 m=$4 t=$5 w=$6 g=$7; shift 7
  grep -qF " $key" $LOG && return
  j=$(sbatch --parsable --nice=10000 -p $p -A sompolinsky_lab -c $c --mem=$m -t $t -J rte_${g}__trueskill_per_family -o $D/logs/units/%x-%j.out -e $D/logs/units/%x-%j.err \
      --export=ALL,RTE_DATA=$D,RTE_PYTHON=$D/env/rte/bin/python,RTE_WORKERS=$w,RTE_CONSOLIDATE=0 scripts/run_grid.sbatch $g --methods trueskill_per_family "$@" 2>&1)
  case "$j" in ''|*[!0-9]*) echo "FAIL $key $j" >&2;; *) echo "$j $key" >> $LOG;; esac
}
for s in $(seq 1 10); do sub n100_$s sapphire,serial_requeue 1 40G 6:00:00 1 trueskill_fix_n100 --seeds $s; sub n1000_$s sapphire,serial_requeue 1 40G 8:00:00 1 trueskill_fix_n1000 --seeds $s; done
for s in 1 2 3; do sub n10k_$s sapphire,shared 1 64G 12:00:00 1 trueskill_fix_n10k --seeds $s; done
echo "TRUESKILL_FIX_SUBMITTED $(wc -l < $LOG)"
