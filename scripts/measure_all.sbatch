#!/bin/bash
#SBATCH --job-name=rte_measure_all --account=sompolinsky_lab --partition=sapphire
#SBATCH --cpus-per-task=16 --mem=64G --time=12:00:00
#SBATCH --output=/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte/logs/%x_%j.out --error=/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte/logs/%x_%j.err
# Every population the live grids need (SPEC §3), sequentially in ONE process: S is per signature and memoised, so after the
# first measurement each population costs only its verifier/description time. Failing the >=1.5 gate is recorded, not fatal.
set -uo pipefail
export RTE_DATA="${RTE_DATA:-/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
cd "${RTE_REPO:-$HOME/rte}"; PY="$RTE_DATA/env/rte/bin/python"
for seed in 1 2 3 4 5; do
  for dist in specialist heavy_tail bimodal; do for n in 100 1000; do "$PY" -m rte.measure --dist $dist --n $n --K 16 --seed $seed || echo "GATE/ERROR $dist $n $seed"; done; done
  for dist in correlated iid_uniform; do "$PY" -m rte.measure --dist $dist --n 1000 --K 16 --seed $seed || echo "GATE/ERROR $dist 1000 $seed"; done
  "$PY" -m rte.measure --dist specialist --n 10000 --K 16 --seed $seed || echo "GATE/ERROR specialist 10000 $seed"
done
