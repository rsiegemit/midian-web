#!/bin/bash
# Shared setup for every vLLM serve job (sourced by serve_fleet.sbatch and serve_smoke.sbatch).
# Sets $PY and defines snapshot_ok / serve_model / watch_health. Zero duplication between the two.
# Expects RTE_DATA, REPO, ENV_PREFIX to be set.

# Container-env decontamination: our shell exports SSL/proxy vars for its own use that do not
# resolve on a compute node; they crashed vLLM boot inside huggingface_hub's httpx client
# (qwen_determinism job 41409968).
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NODE_USE_ENV_PROXY
unset SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE NODE_EXTRA_CA_CERTS DENO_CERT
unset GIT_HTTP_PROXY_AUTHMETHOD

export HF_HOME="$RTE_DATA/hf_cache"
export HF_HUB_OFFLINE=1                      # compute nodes have no internet
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=WARNING
export OUTLINES_CACHE_DIR="$RTE_DATA/cache/outlines"
export XDG_CACHE_HOME="$RTE_DATA/cache/xdg"

# FlashInfer JIT-compiles its top-k/top-p sampler with ninja on first use, inside the
# memory-profiling dummy run; that build FAILS on these nodes and takes the engine down after the
# weights have loaded (smoke job 43858361). We decode greedily at temperature 0, so it buys us
# nothing. FLASHINFER_CACHE_DIR and the HOME shim keep any other JIT off the near-full home quota.
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_CACHE_DIR="$RTE_DATA/cache/flashinfer"
export HOME="$RTE_DATA/home_shim"
mkdir -p "$HOME" "$FLASHINFER_CACHE_DIR" "$OUTLINES_CACHE_DIR" "$XDG_CACHE_HOME" "$RTE_DATA/logs"

# $ENV_PREFIX is a venv (scripts/00_build_env.sh), so activate it by PATH, not `conda activate`.
export PATH="$ENV_PREFIX/bin:$PATH"
PY="$ENV_PREFIX/bin/python"

# The exact call vLLM makes at load time: a partial snapshot fails here in a second instead of
# after a GPU is allocated (smoke job 43851943 died on a missing README.md).
snapshot_ok() {
  "$PY" - "$1" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
try:
    snapshot_download(sys.argv[1], local_files_only=True)
except Exception as e:
    print(f"  {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

# serve_model <model> <port> <gpu_share> <gpu|""> <logfile>   -> backgrounded; $! is its pid.
# `--generation-config vllm` ignores each repo's generation_config.json: Qwen2.5 ships
# repetition_penalty=1.1 and its own temperature there, so without it the models on the ladder
# would be sampled under different rules -- a confound in a benchmark built to compare them.
# --enable-auto-tool-choice/--tool-call-parser hermes: without them a request carrying `tools`
# with tool_choice="auto" returns 400 (fleet job 43864511); all 10 framework rivals need
# structured tool_calls from the Qwen2.5-7B supervisor, and Qwen2.5 uses the hermes format.
serve_model() {
  local model="$1" port="$2" util="$3" gpu="$4" log="$5"
  CUDA_VISIBLE_DEVICES="${gpu:-$CUDA_VISIBLE_DEVICES}" nohup "$ENV_PREFIX/bin/vllm" serve "$model" \
      --served-model-name "$model" \
      --host 0.0.0.0 --port "$port" \
      --dtype bfloat16 \
      --generation-config vllm \
      --max-model-len "${RTE_MAX_MODEL_LEN:-8192}" \
      --gpu-memory-utilization "$util" \
      --max-num-seqs "${RTE_MAX_NUM_SEQS:-64}" \
      --enable-prefix-caching \
      --enable-auto-tool-choice --tool-call-parser hermes \
      ${RTE_VLLM_EXTRA_ARGS:-} \
      > "$log" 2>&1 &
}

# watch_health <model> <port> <host> <tries>  -> registers the endpoint once /health answers 200.
watch_health() {
  local model="$1" port="$2" host="$3" tries="${4:-90}"
  for _ in $(seq 1 "$tries"); do
    if curl -sf --max-time 5 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      "$PY" "$REPO/scripts/_register_endpoint.py" add "$model" "http://$host:$port/v1"
      echo "[serve] HEALTHY $model -> http://$host:$port/v1"
      return 0
    fi
    sleep 10
  done
  echo "[serve] TIMEOUT waiting for $model on port $port"
  return 1
}
