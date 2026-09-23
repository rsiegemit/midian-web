#!/bin/bash
# Feed an explicit unit list through the `test` partition in PACKED jobs (OPS_RULES T1-T3): each job runs K units side by
# side, one rte.run process per unit, framework units with RTE_FW_PARALLEL=8 and 40 G each. `test` starts jobs at once
# but caps us at 5 jobs / 1,000 G, so the driver keeps at most MAXJOBS of its own packs alive and leaves one slot for
# campaign_tick. Resumable: logs/<name>_packed.txt lists every unit already handed to a pack.
#   setsid nohup scripts/ops/pack_driver.sh <plan.tsv> <name> [K] [MEM_PER_UNIT_G] [MAXJOBS] &
#   plan lines: grid <TAB> method <TAB> only <TAB> seed
export RTE_DATA="${RTE_DATA:-/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte}"
PLAN=${1:?plan}; NAME=${2:?name}; K=${3:-8}; MEMU=${4:-40}; MAXJOBS=${5:-3}
L=$RTE_DATA/logs; DONE=$L/${NAME}_packed.txt; HB=$L/${NAME}.heartbeat; REPO=/n/home02/rsiegelmann/rte; touch $DONE
mine() { squeue -u "$USER" -h -p test -o "%j" | grep -c "^rte_pack_${NAME}$"; }
while :; do
  touch $HB
  todo=$(awk -F'\t' '{print $1"|"$2"|"$3"|"$4}' $PLAN | grep -vxFf <(awk '{print $2}' $DONE) | head -$K)
  [ -z "$todo" ] && { echo "$(date -Is) all units handed to packs"; break; }
  if [ "$(mine)" -lt "$MAXJOBS" ]; then
    n=$(echo "$todo" | wc -l); js=$RTE_DATA/jobs/pack_${NAME}_$(date +%s).sh
    { echo '#!/bin/bash'; echo "cd $REPO"
      echo "$todo" | while IFS='|' read -r g m o s; do
        echo "RTE_WORKERS=1 bash scripts/run_grid.sbatch '$g' --methods '$m' --only '$o' --seeds '$s' > \"$L/units/pack_${NAME}_\${SLURM_JOB_ID}_${g}_${m}_s${s}.log\" 2>&1 &"
      done; echo 'wait; echo PACK_DONE'; } > $js; chmod +x $js
    j=$(sbatch --parsable -p test -A sompolinsky_lab -c $n --mem=$((n * MEMU))G -t 8:00:00 -J rte_pack_${NAME} -o $L/pack_${NAME}_%j.out \
        --export=ALL,RTE_DATA=$RTE_DATA,RTE_PYTHON=$RTE_DATA/env/rte/bin/python,RTE_CONSOLIDATE=0,RTE_FW_PARALLEL=8 $js 2>&1)
    case "$j" in ''|*[!0-9]*) echo "$(date -Is) submit refused: $j";; *) echo "$todo" | sed "s/^/$j /" >> $DONE; echo "$(date -Is) pack $j: $n units";; esac
  fi
  sleep 60
done
