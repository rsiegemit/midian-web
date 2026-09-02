#!/bin/bash
#SBATCH --job-name=rte_grid
#SBATCH --partition=sapphire
#SBATCH --account=sompolinsky_lab
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G --time=12:00:00
#SBATCH --output=/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte/logs/%x_%j.out
#SBATCH --error=/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte/logs/%x_%j.err
# CPU grid runner (bernoulli / replay backends; the llm backend runs from the fleet job instead).
#   sbatch scripts/run_grid.sbatch <grid> [rte.run args]
#   sbatch -p bigmem -c 48 --mem 700G --export=ALL,RTE_WORKERS=4 scripts/run_grid.sbatch bernoulli_scale   # 1e6-1e7: memory-bound
#   RTE_PYTHON=$RTE_DATA/env/rte/bin/python sbatch scripts/run_grid.sbatch live_core_n100 --methods midian   # llm grids: project env, one method per job
set -euo pipefail
export RTE_DATA="${RTE_DATA:-/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte}" HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1        # parallelism is over cells, not inside numpy
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE   # llm backend talks to the fleet
cd "${SLURM_SUBMIT_DIR:-$HOME/rte}"; export PYTHONPATH="$PWD"
PY="${RTE_PYTHON:-$HOME/miniconda3/bin/python}"
echo "[$(date -Is)] $(hostname) grid=$1 workers=${RTE_WORKERS:-$SLURM_CPUS_PER_TASK}"
"$PY" -m rte.run --grid "$@" --workers "${RTE_WORKERS:-$SLURM_CPUS_PER_TASK}"
"$PY" -m rte.analyze --grid "$1"
