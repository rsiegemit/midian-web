#!/bin/bash
#SBATCH --job-name=rte_serve_fleet
#SBATCH --account=kempner_sompolinsky_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte/logs/serve_fleet_%j.out
#SBATCH --error=/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte/logs/serve_fleet_%j.err
#
# The RTE model fleet: one vLLM server per model in configs/models.yaml, packed onto 4 GPUs.
# Each registers itself in $RTE_DATA/endpoints.d/ once /health answers, and deregisters on exit.
# WHICH models, WHICH gpu, WHICH port and HOW MUCH HBM are all in configs/models.yaml -- no model
# id appears in this file.
#
#   sbatch scripts/serve_fleet.sbatch                              # kempner_h100, 80 GB/GPU
#   sbatch --partition=kempner_h200 scripts/serve_fleet.sbatch     # 141 GB/GPU, more headroom
set -uo pipefail

export RTE_DATA="${RTE_DATA:-/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte}"
REPO="${RTE_REPO:-$HOME/rte}"
ENV_PREFIX="$RTE_DATA/env/rte"
source "$REPO/scripts/_serve_env.sh"

HOSTNAME_FQ="$(hostname -s)"
JOBLOG="$RTE_DATA/logs/serve_fleet_${SLURM_JOB_ID:-manual}"
mkdir -p "$JOBLOG"
echo "[fleet] host=$HOSTNAME_FQ job=${SLURM_JOB_ID:-manual} logs=$JOBLOG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PY" -c 'import vllm, sys; print("vllm", vllm.__version__, "on python", sys.version.split()[0])'

"$PY" "$REPO/scripts/_register_endpoint.py" clear
PIDS=()
cleanup() {
  echo "[fleet] cleanup: deregistering endpoints and killing servers"
  for m in "${MODELS[@]:-}"; do "$PY" "$REPO/scripts/_register_endpoint.py" remove "$m${RTE_AS_REPLICA:+#${SLURM_JOB_ID:-manual}}" >/dev/null 2>&1 || true; done   # only OUR keys, never a global clear
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
  sleep 5
  for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null; done
}
trap cleanup EXIT TERM INT

# configs/models.yaml -> "id gpu port gpu_share", largest first.
while read -r MODEL GPU PORT UTIL; do
  [ -z "$MODEL" ] && continue
  if ! snapshot_ok "$MODEL"; then
    echo "[fleet] SKIP $MODEL (no complete snapshot in \$HF_HOME; run scripts/01_download_weights.py)"
    continue
  fi
  echo "[fleet] launching $MODEL on GPU $GPU port $PORT util $UTIL"
  serve_model "$MODEL" "$PORT" "$UTIL" "$GPU" "$JOBLOG/${MODEL//\//__}.log"
  PIDS+=($!)
  # Stagger: vLLM sizes its KV cache as total_HBM * util minus what it measures as already in use,
  # so two servers profiling the same GPU at once can both size against a GPU they think is empty.
  sleep "${RTE_LAUNCH_STAGGER:-90}"
  watch_health "$MODEL" "$PORT" "$HOSTNAME_FQ" 240 &
done < <("$PY" -c "
import yaml
# configs/models.yaml is ordered ASCENDING by params_b (a profile picks its model by index there).
# Launch largest first so each later server on a shared GPU sees its neighbour's allocation.
for m in sorted(yaml.safe_load(open('$REPO/configs/models.yaml'))['models'],
                key=lambda m: -m['params_b']):
    print(m['id'], m['gpu'], m['port'], m['gpu_share'])
")

echo "[fleet] ${#PIDS[@]} servers launched; endpoints: $RTE_DATA/endpoints.json"
wait
