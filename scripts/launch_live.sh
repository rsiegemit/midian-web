#!/bin/bash
# Launch every live (llm-backend) grid as one SLURM job PER METHOD (the llm runner is single-process; jobs share the fleet
# and the memo shards). Usage: scripts/launch_live.sh [dependency-jobid]
set -uo pipefail
cd "${RTE_REPO:-$HOME/rte}"; export RTE_DATA="${RTE_DATA:-/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte}"
DEP=${1:+--dependency=afterany:$1}
PY="$RTE_DATA/env/rte/bin/python"
for grid in live_core_n100 fw_live_n100 live_f1_n1000 fw_live_n1000 live_extra_n1000 budget_sweep midian_internals fw_k_sensitivity fw_appendix live_n10k; do
  for m in $("$PY" -c "
import sys; sys.path.insert(0,'.'); import yaml
from rte.run import blocks, method_specs
cfg = yaml.safe_load(open('configs/grid.yaml')); print(' '.join(sorted({s['name'] for b in blocks(cfg, '$grid') for s in method_specs(b)})))"); do
    sbatch --parsable $DEP --job-name="rte_${grid}__${m}" -c 2 --mem=24G --time=2-00:00:00 --export=ALL,RTE_PYTHON="$PY",RTE_WORKERS=1 \
      scripts/run_grid.sbatch "$grid" --methods "$m" | sed "s/$/  $grid $m/"
  done
done
