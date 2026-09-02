#!/bin/bash
#SBATCH --job-name=rte_serve_smoke
#SBATCH --account=kempner_sompolinsky_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte/logs/serve_smoke_%j.out
#SBATCH --error=/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte/logs/serve_smoke_%j.err
#
# One GPU, one model (Qwen2.5-0.5B-Instruct), one hour: the end-to-end check that
# env + weights + vLLM + rte.llm_client + LLMBackend all line up before the real fleet.
#
# NOT gpu_test: that partition hands out A100 MIG slices, and SLURM then sets
# CUDA_VISIBLE_DEVICES to a MIG UUID. vLLM 0.22.1 does int() on that string and dies with
#   ValueError: invalid literal for int() with base 10: 'MIG-adfbd773-...'
# (vllm/platforms/cuda.py, via registry.py:951; smoke job 43857516). Any whole-GPU partition
# works; kempner_h100 is what the real fleet uses anyway.
#
#   sbatch scripts/serve_smoke.sbatch
#   # wait for $RTE_DATA/endpoints.json, then:
#   $RTE_DATA/env/rte/bin/python scripts/smoke_backend.py
set -uo pipefail

export RTE_DATA="${RTE_DATA:-/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte}"
REPO="${RTE_REPO:-$HOME/rte}"
ENV_PREFIX="$RTE_DATA/env/rte"
MODEL="${RTE_SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${RTE_SMOKE_PORT:-8011}"

# Container-env decontamination (see serve_fleet.sbatch).
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NODE_USE_ENV_PROXY
unset SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE NODE_EXTRA_CA_CERTS DENO_CERT GIT_HTTP_PROXY_AUTHMETHOD

export HF_HOME="$RTE_DATA/hf_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=WARNING
export OUTLINES_CACHE_DIR="$RTE_DATA/cache/outlines"
export XDG_CACHE_HOME="$RTE_DATA/cache/xdg"

# FlashInfer JIT-compiles its top-k/top-p sampler with ninja on first use, inside
# determine_available_memory()'s dummy run. On these nodes that build FAILS
# (CalledProcessError from flashinfer/jit/cpp_ext.py, smoke job 43858361) and takes the whole
# engine down after the weights have already loaded. We decode greedily at temperature 0, so the
# FlashInfer sampler buys us nothing -- turn it off and use vLLM's torch path.
# `--generation-config vllm --enable-auto-tool-choice --tool-call-parser hermes` ignores each repo's generation_config.json. Qwen2.5 ships
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

# $ENV_PREFIX is a venv, activated by PATH (see scripts/00_build_env.sh).
export PATH="$ENV_PREFIX/bin:$PATH"
PY="$ENV_PREFIX/bin/python"

HOSTNAME_FQ="$(hostname -s)"
LOG="$RTE_DATA/logs/serve_smoke_${SLURM_JOB_ID:-manual}.vllm.log"
echo "[smoke] host=$HOSTNAME_FQ job=${SLURM_JOB_ID:-manual} model=$MODEL port=$PORT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

cleanup() {
  "$PY" "$REPO/scripts/_register_endpoint.py" remove "$MODEL" >/dev/null 2>&1 || true
  [ -n "${VPID:-}" ] && kill "$VPID" 2>/dev/null
  sleep 3
  [ -n "${VPID:-}" ] && kill -9 "$VPID" 2>/dev/null
  true
}
trap cleanup EXIT TERM INT

"$ENV_PREFIX/bin/vllm" serve "$MODEL" \
    --served-model-name "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --dtype bfloat16 \
    --generation-config vllm --enable-auto-tool-choice --tool-call-parser hermes \
    --max-model-len "${RTE_MAX_MODEL_LEN:-4096}" \
    --gpu-memory-utilization 0.60 \
    --max-num-seqs 64 \
    --enable-prefix-caching \
    ${RTE_VLLM_EXTRA_ARGS:-} \
    > "$LOG" 2>&1 &
VPID=$!

for _ in $(seq 1 90); do
  if curl -sf --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    "$PY" "$REPO/scripts/_register_endpoint.py" add "$MODEL" "http://$HOSTNAME_FQ:$PORT/v1"
    echo "[smoke] HEALTHY -> http://$HOSTNAME_FQ:$PORT/v1"
    break
  fi
  if ! kill -0 "$VPID" 2>/dev/null; then
    echo "[smoke] vLLM died during startup; tail of $LOG:"; tail -60 "$LOG"; exit 1
  fi
  sleep 10
done

if ! curl -sf --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "[smoke] TIMEOUT waiting for /health; tail of $LOG:"; tail -60 "$LOG"; exit 1
fi

echo "[smoke] serving until the job ends or is scancel'd"
wait "$VPID"
