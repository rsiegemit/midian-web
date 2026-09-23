#!/bin/bash
# One run process PER SEED inside one job (rte.run runs llm-backend units one at a time, so workers>1 does nothing there).
#   seed_pack.sh <grid> <method> <seed> [<seed> ...]
cd /n/home02/rsiegelmann/rte; G=$1; M=$2; shift 2
for s in "$@"; do RTE_WORKERS=1 bash scripts/run_grid.sbatch "$G" --methods "$M" --seeds "$s" > "$RTE_DATA/logs/units/pack_${G}_s$s.log" 2>&1 & done
wait; echo "PACK_DONE $G"
