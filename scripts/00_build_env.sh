#!/bin/bash
# Build the RTE LLM env at $RTE_DATA/env/rte (python 3.12, vLLM + reasoning-gym + CPU deps).
# Run on the LOGIN node (needs internet). Idempotent-ish: re-running re-pips into the same prefix.
#
# Pins come from the validated cu129 vLLM env at
#   /n/netscratch/sompolinsky_lab/Lab/rsiegelmann/qwen_determinism/env/qwen38-vllm-cu129
# (read-only reference; we do NOT reuse it -- it carries a patched vLLM scheduler overlay).
# If the pinned wheel URL ever 404s, set RTE_VLLM_FALLBACK=1 to `pip install vllm` from PyPI
# and record the resulting version in DEVIATIONS.md.
set -euo pipefail

export RTE_DATA="${RTE_DATA:-/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte}"
ENV_PREFIX="$RTE_DATA/env/rte"

mkdir -p "$RTE_DATA"/{env,hf_cache,logs,cache,populations,results,conda_pkgs,pip_cache}

# Home quota is nearly full: keep every cache off $HOME.
export CONDA_PKGS_DIRS="$RTE_DATA/conda_pkgs"
export PIP_CACHE_DIR="$RTE_DATA/pip_cache"
export HF_HOME="$RTE_DATA/hf_cache"

if [ -f /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh ]; then
    source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh
else
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    conda create -y -p "$ENV_PREFIX" python=3.12
fi
conda activate "$ENV_PREFIX"

python -m pip install --upgrade pip setuptools wheel

VLLM_WHL="https://github.com/vllm-project/vllm/releases/download/v0.22.1/vllm-0.22.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"

if [ "${RTE_VLLM_FALLBACK:-0}" = "1" ]; then
    pip install vllm
else
    # torch first, from the cu129 index, so the vLLM wheel finds its exact dep already satisfied.
    pip install --index-url https://download.pytorch.org/whl/cu129 \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
    pip install "vllm @ ${VLLM_WHL}"
fi

# Benchmark deps. numpy pinned <2 to match the validated vLLM env.
pip install "numpy==1.26.4" reasoning-gym openai pytest pandas scipy pyyaml hnswlib trueskill scikit-learn

python - <<'PY'
import importlib, sys
ok = True
for m in ["numpy", "openai", "pytest", "pandas", "scipy", "yaml", "hnswlib", "trueskill", "sklearn",
          "reasoning_gym", "torch", "vllm"]:
    try:
        mod = importlib.import_module(m)
        print(f"  OK  {m:16s} {getattr(mod, '__version__', '?')}")
    except Exception as e:                                     # noqa: BLE001
        ok = False
        print(f"  FAIL {m:16s} {type(e).__name__}: {e}")
print("ENV_OK" if ok else "ENV_INCOMPLETE")
sys.exit(0 if ok else 1)
PY

echo "env at: $ENV_PREFIX"
