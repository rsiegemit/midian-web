#!/bin/bash
# Launch live (llm-backend) grids as one SLURM job PER METHOD (the llm runner is single-process; jobs share the fleet and the
# memo shards). Usage: scripts/launch_live.sh [dependency-jobid]
#   RTE_GRIDS="live_f1_n1000 ..."  grids (default: all live grids)      RTE_ONLY="midian sequential_halving"  methods
#   RTE_SHARD="dist,beta"  one job per value-combination of these cell axes (--only)   RTE_SEED_SHARD=1  ... and per seed
set -uo pipefail
cd "${RTE_REPO:-$HOME/rte}"; export RTE_DATA="${RTE_DATA:-/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte}"
DEP=${1:+--dependency=afterany:$1}
PY="$RTE_DATA/env/rte/bin/python"
for grid in ${RTE_GRIDS:-live_core_n100 fw_live_n100 live_f1_n1000 fw_live_n1000 live_extra_n1000 budget_sweep midian_internals fw_k_sensitivity fw_appendix live_n10k}; do
  for m in $("$PY" -c "
import sys; sys.path.insert(0,'.'); import yaml
from rte.run import blocks, method_specs
cfg = yaml.safe_load(open('configs/grid.yaml')); print(' '.join(sorted({s['name'] for b in blocks(cfg, '$grid') for s in method_specs(b)})))"); do
    [ -n "${RTE_ONLY:-}" ] && [[ " $RTE_ONLY " != *" $m "* ]] && continue
    while read -r only seed; do
      [ "$only" = - ] && only=; [ "$seed" = - ] && seed=
      sbatch --parsable $DEP ${RTE_PARTITION:+-p $RTE_PARTITION} --job-name="rte_${grid}__${m}" -c ${RTE_CPUS:-2} --mem=${RTE_MEM:-24G} --time=2-00:00:00 --export=ALL,RTE_PYTHON="$PY",RTE_WORKERS=1 \
        scripts/run_grid.sbatch "$grid" --methods "$m" ${only:+--only $only} ${seed:+--seeds $seed} ${RTE_RUN_ARGS:-} | sed "s/$/  $grid $m $only $seed/"
    done < <("$PY" -c "
import sys; sys.path.insert(0,'.'); import yaml
from itertools import product
from rte.run import blocks, seeds
cfg = yaml.safe_load(open('configs/grid.yaml')); axes = [a for a in '${RTE_SHARD:-}'.split(',') if a]
for b in blocks(cfg, '$grid'):
    for combo in product(*[b[a] for a in axes]):
        for s in (seeds(b['seeds']) if '${RTE_SEED_SHARD:-}' else ['-']):
            print(','.join(f'{a}={v}' for a, v in zip(axes, combo)) or '-', s)")
  done
done
