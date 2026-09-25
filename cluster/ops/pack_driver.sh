#!/bin/bash
# Feed an explicit unit list through PACKED jobs (OPS_RULES T1-T3, T6): each job runs K units side by side, one rte.run
# process per unit, framework units with RTE_FW_PARALLEL=8 and 40 G each. Two kinds of pack, fed by ONE driver so no
# unit is ever handed out twice:
#   test    -- starts at once, but test caps us at 5 jobs / 1,000 G (leave room for other test work: TEST_MAX)
#   kpack   -- a GPU node: the job serves ONE supervisor replica on its GPU (the GPU works for the whole fleet,
#              OPS_RULES F2) and runs the units on the node's spare CPUs; the replica is stopped (and deregisters) when
#              the units finish. Separate account and fairshare from the CPU partitions (K_MAX jobs).
# Resumable: logs/<name>_packed.txt lists every unit already handed to a pack.
#   TEST_MAX=2 K_MAX=3 setsid nohup cluster/ops/pack_driver.sh <plan.tsv> <name> [K] [MEM_PER_UNIT_G] &
#   Needs RTE_DATA, RTE_REPO, RTE_ACCOUNT, RTE_TEST_PARTITION (+ RTE_GPU_ACCOUNT, RTE_GPU_PARTITION when K_MAX > 0).
#   plan lines: grid <TAB> method <TAB> only <TAB> seed
. "$(dirname "$(readlink -f "$0")")/../env.sh"
need RTE_DATA RTE_ACCOUNT RTE_TEST_PARTITION
PLAN=${1:?plan}; NAME=${2:?name}; K=${3:-8}; MEMU=${4:-40}; TEST_MAX=${TEST_MAX:-2}; K_MAX=${K_MAX:-0}
L=$RTE_DATA/logs; DONE=$L/${NAME}_packed.txt; HB=$L/${NAME}.heartbeat; REPO=$RTE_REPO; touch $DONE
[ "$K_MAX" -gt 0 ] && need RTE_GPU_ACCOUNT RTE_GPU_PARTITION
mine() { squeue -u "$USER" -h -o "%j" | grep -c "^rte_${1}_${NAME}$"; }
while :; do
  touch $HB
  todo=$(awk -F'\t' '{print $1"|"$2"|"$3"|"$4}' $PLAN | grep -vxFf <(awk '{print $2}' $DONE) | head -$K)
  [ -z "$todo" ] && { echo "$(date -Is) all units handed to packs"; break; }
  kind=""; [ "$(mine pack)" -lt "$TEST_MAX" ] && kind=pack; [ -z "$kind" ] && [ "$(mine kpack)" -lt "$K_MAX" ] && kind=kpack
  if [ -n "$kind" ]; then
    n=$(echo "$todo" | wc -l); js=$RTE_DATA/jobs/${kind}_${NAME}_$(date +%s).sh
    { echo '#!/bin/bash'; echo "cd $REPO"
      [ $kind = kpack ] && { echo "RTE_AS_REPLICA=1 RTE_FLEET_PLACEMENT=$REPO/configs/fleet_supervisor.yaml bash cluster/slurm/serve_fleet.sbatch > \"$L/kpack_\${SLURM_JOB_ID}_fleet.log\" 2>&1 &"
                             echo 'FLEET=$!'; }
      echo 'P=()'
      echo "$todo" | while IFS='|' read -r g m o s; do
        echo "RTE_WORKERS=1 bash cluster/slurm/run_grid.sbatch '$g' --methods '$m' --only '$o' --seeds '$s' > \"$L/units/${kind}_${NAME}_\${SLURM_JOB_ID}_${g}_${m}_s${s}.log\" 2>&1 & P+=(\$!)"
      done
      echo 'wait "${P[@]}"'
      [ $kind = kpack ] && echo 'kill -TERM $FLEET; wait $FLEET'      # the replica deregisters in its own trap
      echo 'echo PACK_DONE'; } > $js; chmod +x $js
    if [ $kind = pack ]; then args=(-p "$RTE_TEST_PARTITION" -A "$RTE_ACCOUNT" -c $n --mem=$((n * MEMU))G -t 8:00:00)
    else args=(-p "$RTE_GPU_PARTITION" -A "$RTE_GPU_ACCOUNT" --gres=gpu:1 -c $((n + 8)) --mem=$((n * MEMU + 64))G -t 12:00:00); fi
    j=$(sbatch --parsable "${args[@]}" -J rte_${kind}_${NAME} -o $L/${kind}_${NAME}_%j.out \
        --export=ALL,RTE_DATA=$RTE_DATA,RTE_PYTHON=$RTE_DATA/env/rte/bin/python,RTE_CONSOLIDATE=0,RTE_FW_PARALLEL=8 $js 2>&1)
    case "$j" in ''|*[!0-9]*) echo "$(date -Is) $kind submit refused: $j";; *) echo "$todo" | sed "s/^/$j /" >> $DONE; echo "$(date -Is) $kind $j: $n units";; esac
  fi
  sleep 60
done
