#!/bin/bash
#SBATCH --job-name=rte_serve_fleet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-serve_fleet-%j.out
#SBATCH --error=slurm-serve_fleet-%j.err
#
# The RTE model fleet: one vLLM server per model in configs/models.yaml, packed onto 4 GPUs.
# Each registers itself in $RTE_DATA/endpoints.d/ once /health answers, and deregisters on exit.
# WHICH models, WHICH gpu, WHICH port and HOW MUCH HBM are all in configs/models.yaml -- no model
# id appears in this file.
#
# Account / partition from the caller (cluster/cluster.env.example); scripts/serve_fleet.sbatch is a symlink here.
#   sbatch -A "$RTE_GPU_ACCOUNT" -p "$RTE_GPU_PARTITION" cluster/slurm/serve_fleet.sbatch   # 80 GB/GPU H100s: the placement in models.yaml
set -uo pipefail

: "${RTE_DATA:?RTE_DATA is not set (cluster/cluster.env.example)}"
export RTE_DATA
REPO="${RTE_REPO:-$HOME/rte}"
ENV_PREFIX="$RTE_DATA/env/rte"
source "$REPO/cluster/slurm/_serve_env.sh"

HOSTNAME_FQ="$(hostname -s)"
JOBLOG="$RTE_DATA/logs/serve_fleet_${SLURM_JOB_ID:-manual}"
mkdir -p "$JOBLOG"
echo "[fleet] host=$HOSTNAME_FQ job=${SLURM_JOB_ID:-manual} logs=$JOBLOG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PY" -c 'import vllm, sys; print("vllm", vllm.__version__, "on python", sys.version.split()[0])'

[ -z "${RTE_AS_REPLICA:-}" ] && [ -n "${RTE_CLEAR_REGISTRY:-}" ] && "$PY" "$REPO/cluster/slurm/_register_endpoint.py" clear   # never wipe other servers' entries by default
PIDS=()
cleanup() {
  echo "[fleet] cleanup: deregistering endpoints and killing servers"
  for k in "${KEYS[@]:-}"; do "$PY" "$REPO/cluster/slurm/_register_endpoint.py" remove "$k" >/dev/null 2>&1 || true; done   # only OUR keys, never a global clear
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
  sleep 5
  for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null; done
}
trap cleanup EXIT TERM INT

# configs/models.yaml -> "id gpu port gpu_share", largest first.
KEYS=()
while read -r MODEL GPU PORT UTIL; do
  [ -z "$MODEL" ] && continue
  KEYS+=("$MODEL${RTE_AS_REPLICA:+#${SLURM_JOB_ID:-manual}}")
  if ! snapshot_ok "$MODEL"; then
    echo "[fleet] SKIP $MODEL (no complete snapshot in \$HF_HOME; run scripts/setup/01_download_weights.py)"
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
import os, yaml
# configs/models.yaml is ordered ASCENDING by params_b (a profile picks its model by index there), so it is never
# edited for placement. RTE_FLEET_PLACEMENT names a file of {id: {gpu, gpu_share}} that SELECTS and PLACES models:
# only listed ids are served. Rule: request exactly the GPUs the placement uses -- a fleet that loads models nobody
# calls leaves GPUs at 0% (2026-09-22: the 7B supervisor took 487,222 requests in 16 h, the other six models 8).
ms = yaml.safe_load(open('$REPO/configs/models.yaml'))['models']
pl = os.environ.get('RTE_FLEET_PLACEMENT')
if pl:
    over = yaml.safe_load(open(pl))['place']
    ms = [dict(m, **over[m['id']]) for m in ms if m['id'] in over]
# Launch largest first so each later server on a shared GPU sees its neighbour's allocation.
for m in sorted(ms, key=lambda m: -m['params_b']):
    print(m['id'], m['gpu'], m['port'], m['gpu_share'])
")

echo "[fleet] ${#PIDS[@]} servers launched; endpoints: $RTE_DATA/endpoints.json"
wait
