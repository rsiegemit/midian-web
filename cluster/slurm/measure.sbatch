#!/bin/bash
#SBATCH --job-name=rte_measure
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-measure-%j.out
#SBATCH --error=slurm-measure-%j.err
#
# CPU-only: it calls the vLLM fleet over HTTP and runs the verifiers and python sandboxes locally.
#   sbatch -A "$RTE_ACCOUNT" -p <cpu partition> cluster/slurm/measure.sbatch <dist> <n> [seed]      # e.g. specialist 100 1
set -euo pipefail
: "${RTE_DATA:?RTE_DATA is not set (cluster/cluster.env.example)}"
export RTE_DATA
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NODE_USE_ENV_PROXY
unset SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE NODE_EXTRA_CA_CERTS
cd "${RTE_REPO:-$HOME/rte}"
"$RTE_DATA/env/rte/bin/python" -m midian.measure --dist "$1" --n "$2" --K 16 --seed "${3:-1}"
