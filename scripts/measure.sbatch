#!/bin/bash
#SBATCH --job-name=rte_measure
#SBATCH --account=sompolinsky_lab
#SBATCH --partition=sapphire
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte/logs/measure_%j.out
#SBATCH --error=/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte/logs/measure_%j.err
#
# CPU-only: it calls the vLLM fleet over HTTP and runs the verifiers and python sandboxes locally.
#   sbatch scripts/measure.sbatch <dist> <n> [seed]      # e.g. specialist 100 1
set -euo pipefail
export RTE_DATA="${RTE_DATA:-/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NODE_USE_ENV_PROXY
unset SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE NODE_EXTRA_CA_CERTS
cd "${RTE_REPO:-$HOME/rte}"
"$RTE_DATA/env/rte/bin/python" -m rte.measure --dist "$1" --n "$2" --K 16 --seed "${3:-1}"
