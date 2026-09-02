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
# The RTE model fleet: 7 vLLM servers packed onto 4 GPUs, one OpenAI-compatible
# endpoint each. Each writes itself into $RTE_DATA/endpoints.json once /health
# answers 200, and removes itself on exit.
#
#   sbatch --partition=kempner_h100 scripts/serve_fleet.sbatch          # 80 GB/GPU
#   sbatch --partition=kempner_h200 scripts/serve_fleet.sbatch          # 141 GB/GPU, more headroom
#
# Packing (fractions of one GPU's HBM, weights in bf16):
#   GPU0  Qwen2.5-14B (28 GB)                      0.88
#   GPU1  Qwen2.5-7B (15 GB) + gemma-2-2b (5 GB)   0.55 + 0.35
#   GPU2  Qwen2.5-0.5B + 1.5B + 3B (10 GB total)   0.15 + 0.25 + 0.35
#   GPU3  gemma-2-9b (18.5 GB)                     0.88
set -uo pipefail

export RTE_DATA="${RTE_DATA:-/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte}"
REPO="${RTE_REPO:-$HOME/rte}"
ENV_PREFIX="$RTE_DATA/env/rte"

# --- container-env decontamination -----------------------------------------
# Our shell container exports SSL/proxy vars for its own use that do not resolve
# on a compute node; they crashed vLLM boot inside huggingface_hub's httpx client
# (qwen_determinism job 41409968). Same block as
# qwen_freeagents_campaign/scripts/sbatch_serve_campaign.slurm:25-32.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NODE_USE_ENV_PROXY
unset SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE NODE_EXTRA_CA_CERTS DENO_CERT GIT_HTTP_PROXY_AUTHMETHOD

export HF_HOME="$RTE_DATA/hf_cache"
export HF_HUB_OFFLINE=1                      # compute nodes have no internet
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=WARNING
export OUTLINES_CACHE_DIR="$RTE_DATA/cache/outlines"
export XDG_CACHE_HOME="$RTE_DATA/cache/xdg"

# FlashInfer JIT-compiles its top-k/top-p sampler with ninja on first use, inside
# determine_available_memory()'s dummy run. On these nodes that build FAILS
# (CalledProcessError from flashinfer/jit/cpp_ext.py, smoke job 43858361) and takes the whole
# engine down after the weights have already loaded. We decode greedily at temperature 0, so the
# FlashInfer sampler buys us nothing -- turn it off and use vLLM's torch path.
# `--generation-config vllm` ignores each repo's generation_config.json. Qwen2.5 ships
# repetition_penalty=1.1 and its own temperature there; letting per-model defaults through
# would mean the ladder's models are sampled under different rules, a confound in a
# benchmark whose whole point is comparing them. The client sets temperature 0 / seed 0.
export VLLM_USE_FLASHINFER_SAMPLER=0
# ... and if any other FlashInfer module JITs, keep its build tree off the near-full home quota.
export FLASHINFER_CACHE_DIR="$RTE_DATA/cache/flashinfer"
# A writable HOME on netscratch, the same shim the validated qwen_determinism serve jobs use:
# several libraries write to ~/.cache unconditionally.
export HOME="$RTE_DATA/home_shim"
mkdir -p "$HOME" "$FLASHINFER_CACHE_DIR"
mkdir -p "$OUTLINES_CACHE_DIR" "$XDG_CACHE_HOME" "$RTE_DATA/logs"

# $ENV_PREFIX is a venv (scripts/00_build_env.sh), so activate it by PATH, not `conda activate`.
export PATH="$ENV_PREFIX/bin:$PATH"
PY="$ENV_PREFIX/bin/python"
"$PY" -c 'import vllm, sys; print("vllm", vllm.__version__, "on python", sys.version.split()[0])'

HOSTNAME_FQ="$(hostname -s)"
JOBLOG="$RTE_DATA/logs/serve_fleet_${SLURM_JOB_ID:-manual}"
mkdir -p "$JOBLOG"
echo "[fleet] host=$HOSTNAME_FQ job=${SLURM_JOB_ID:-manual} logs=$JOBLOG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# model | gpu | port | gpu_memory_utilization | max_num_seqs
FLEET=(
  "Qwen/Qwen2.5-14B-Instruct|0|8001|0.88|64"
  "Qwen/Qwen2.5-7B-Instruct|1|8002|0.55|64"
  "google/gemma-2-2b-it|1|8003|0.35|64"
  "Qwen/Qwen2.5-0.5B-Instruct|2|8004|0.15|64"
  "Qwen/Qwen2.5-1.5B-Instruct|2|8005|0.25|64"
  "Qwen/Qwen2.5-3B-Instruct|2|8006|0.35|64"
  "google/gemma-2-9b-it|3|8007|0.88|64"
)
# Models the download step skipped are dropped automatically.
MAX_MODEL_LEN="${RTE_MAX_MODEL_LEN:-4096}"

"$PY" "$REPO/scripts/_register_endpoint.py" clear

PIDS=()
cleanup() {
  echo "[fleet] cleanup: deregistering endpoints and killing servers"
  for spec in "${FLEET[@]}"; do
    IFS='|' read -r MODEL _ _ _ _ <<< "$spec"
    "$PY" "$REPO/scripts/_register_endpoint.py" remove "$MODEL" >/dev/null 2>&1 || true
  done
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
  sleep 5
  for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null || true; done
}
trap cleanup EXIT TERM INT

for spec in "${FLEET[@]}"; do
  IFS='|' read -r MODEL GPU PORT UTIL NSEQ <<< "$spec"
  SLUG="${MODEL//\//__}"

  # Skip models with no COMPLETE local snapshot. This is the exact call vLLM makes at load time,
  # so a partial download fails here in one second instead of after the GPU is allocated
  # (smoke job 43851943: an allow_patterns download left out README.md and vLLM refused to boot).
  if ! "$PY" - "$MODEL" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
try:
    snapshot_download(sys.argv[1], local_files_only=True)
except Exception as e:
    print(f"  {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
    sys.exit(1)
PYEOF
  then
    echo "[fleet] SKIP $MODEL (no complete snapshot in \$HF_HOME; run scripts/01_download_weights.py)"
    continue
  fi

  echo "[fleet] launching $MODEL on GPU $GPU port $PORT util $UTIL"
  CUDA_VISIBLE_DEVICES="$GPU" nohup "$ENV_PREFIX/bin/vllm" serve "$MODEL" \
      --served-model-name "$MODEL" \
      --host 0.0.0.0 --port "$PORT" \
      --dtype bfloat16 \
    --generation-config vllm \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$UTIL" \
      --max-num-seqs "$NSEQ" \
      --enable-prefix-caching \
      --enable-auto-tool-choice --tool-call-parser hermes ${RTE_VLLM_EXTRA_ARGS:-} \
      > "$JOBLOG/$SLUG.log" 2>&1 &
  PIDS+=($!)

  # Stagger launches. vLLM sizes its KV cache from `total_gpu_memory * gpu_memory_utilization`
  # minus what it measures as already in use, so two servers profiling the same GPU at the same
  # moment can both size against a GPU they each think is empty. Serving them in order, largest
  # first, with a gap, makes each later server see its neighbour's allocation.
  sleep "${RTE_LAUNCH_STAGGER:-90}"

  # health watcher: register the endpoint the moment /health answers 200
  (
    for _ in $(seq 1 240); do          # up to 40 min for the 14B to load
      if curl -sf --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        "$PY" "$REPO/scripts/_register_endpoint.py" add "$MODEL" "http://$HOSTNAME_FQ:$PORT/v1"
        echo "[fleet] HEALTHY $MODEL -> http://$HOSTNAME_FQ:$PORT/v1"
        exit 0
      fi
      sleep 10
    done
    echo "[fleet] TIMEOUT waiting for $MODEL on port $PORT (see $JOBLOG/$SLUG.log)"
  ) &
done

echo "[fleet] ${#PIDS[@]} servers launched; endpoints file: $RTE_DATA/endpoints.json"
wait
