#!/bin/bash
# One run process per --only cell inside one job:  only_pack.sh <grid> <methods> <seeds> <workers> <only> [<only> ...]
cd "$RTE_REPO"; G=$1; M=$2; S=$3; W=$4; shift 4
for o in "$@"; do RTE_WORKERS=$W bash scripts/run_grid.sbatch "$G" --methods "$M" --only "$o" --seeds "$S" > "$RTE_DATA/logs/units/pack_${G}_${o//[=,]/_}.log" 2>&1 & done
wait; echo "PACK_DONE $G"
