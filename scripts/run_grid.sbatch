#!/bin/bash
#SBATCH --job-name=rte_grid
#SBATCH --partition=sapphire
#SBATCH --account=sompolinsky_lab
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# CPU grid runner (bernoulli / replay backends; the llm backend runs from the fleet job instead).
#   sbatch scripts/run_grid.sbatch <grid> [rte.run args]
#   sbatch -p bigmem -c 48 --mem 700G --export=ALL,RTE_WORKERS=4 scripts/run_grid.sbatch bernoulli_scale   # 1e6-1e7: memory-bound
#   sbatch scripts/run_grid.sbatch live_core_n100 --methods midian --only beta=0,liar_select=random --seeds 1   # llm grids: one UNIT per job
set -euo pipefail
export RTE_DATA="${RTE_DATA:-/scratch/rte}" HF_HUB_OFFLINE=1
# Parallelism is over cells, not inside numpy -- EXCEPT when a job holds one cell (--only ... --seeds N), where there
# is no cross-cell parallelism to protect and the learned routers embed n*K*b prompts through MiniLM on one thread.
# Caller-provided values win, so `OMP_NUM_THREADS=8 sbatch -c 8 ...` is honoured; the default is unchanged.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE   # llm backend talks to the fleet
cd "${SLURM_SUBMIT_DIR:-$HOME/rte}"; export PYTHONPATH="$PWD"
# Default to the PROJECT env: only it has torch/transformers/sentence_transformers, which the learned routers
# (knn_router, mlp_router) import via rte/methods/_learned.py. The miniconda interpreter runs every other
# method fine, which is why a missing RTE_PYTHON stayed invisible until knn_router at n = 10^5 (2026-09-08).
PY="${RTE_PYTHON:-$RTE_DATA/env/rte/bin/python}"
[ -x "$PY" ] || PY="$HOME/miniconda3/bin/python"
echo "[$(date -Is)] $(hostname) grid=$1 workers=${RTE_WORKERS:-$SLURM_CPUS_PER_TASK}"
"$PY" -m rte.run --grid "$@" --workers "${RTE_WORKERS:-$SLURM_CPUS_PER_TASK}"
"$PY" -m rte.analyze --grid "$1"
