#!/bin/bash
# Build the RTE LLM env at $RTE_DATA/env/rte (python 3.12, vLLM + reasoning-gym + CPU deps).
# Run on the LOGIN node (needs internet). Re-running re-pips into the same prefix.
#
# WHY venv AND NOT `conda create`: the Miniforge base interpreter at
# /n/sw/Miniforge3-25.3.1-0 is already CPython 3.12.11, and `conda create` on this
# cluster serialises on an fcntl lock over the shared repodata cache -- with several
# env builds running concurrently it died with
# `BlockingIOError: [Errno 11] Resource temporarily unavailable` (build_env.log,
# 2026-09-02). A venv off that same 3.12 interpreter needs no solver, no lock, and
# gives an identical `$RTE_DATA/env/rte/bin/python`. Set RTE_USE_CONDA=1 to force
# the conda path instead.
#
# Pins come from the validated cu129 vLLM env at
#   /n/netscratch/sompolinsky_lab/Lab/rsiegelmann/qwen_determinism/env/qwen38-vllm-cu129
# (read-only reference; we do NOT reuse it -- it carries a patched vLLM scheduler overlay).
# Set RTE_VLLM_FALLBACK=1 to `pip install vllm` from PyPI instead of the pinned wheel,
# and record the resulting version in DEVIATIONS.md.
set -euo pipefail

export RTE_DATA="${RTE_DATA:-/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte}"
ENV_PREFIX="$RTE_DATA/env/rte"
CONDA_SH="${RTE_CONDA_SH:-/n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh}"
[ -f "$CONDA_SH" ] || CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
BASE_PY="$(dirname "$(dirname "$CONDA_SH")")/../bin/python"     # <prefix>/etc/profile.d/.. -> <prefix>

mkdir -p "$RTE_DATA"/{env,hf_cache,logs,cache,populations,results,conda_pkgs,pip_cache}

# Home quota is nearly full: keep every cache off $HOME.
# vLLM depends on `llguidance`, which has no wheel for every cp312 build and then compiles from
# Rust via maturin -- which BOOTSTRAPS A RUST TOOLCHAIN, ~1.5 GB, into $HOME/.cache/puccinialin
# unless these are redirected. That very nearly filled the home quota on the first build.
export CONDA_PKGS_DIRS="$RTE_DATA/conda_pkgs"
export PIP_CACHE_DIR="$RTE_DATA/pip_cache"
export HF_HOME="$RTE_DATA/hf_cache"
export TMPDIR="$RTE_DATA/tmp"
export CARGO_HOME="$RTE_DATA/cargo"
export RUSTUP_HOME="$RTE_DATA/rustup"
export XDG_CACHE_HOME="$RTE_DATA/cache/xdg"
mkdir -p "$TMPDIR" "$CARGO_HOME" "$RUSTUP_HOME" "$XDG_CACHE_HOME"

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    if [ "${RTE_USE_CONDA:-0}" = "1" ]; then
        source "$CONDA_SH"
        conda create -y -p "$ENV_PREFIX" python=3.12
    else
        "$BASE_PY" -m venv "$ENV_PREFIX"
    fi
fi
PY="$ENV_PREFIX/bin/python"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version; print("python", sys.version)'

"$PY" -m pip install --upgrade pip setuptools wheel

VLLM_WHL="https://github.com/vllm-project/vllm/releases/download/v0.22.1/vllm-0.22.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"

if [ "${RTE_VLLM_FALLBACK:-0}" = "1" ]; then
    "$PY" -m pip install vllm
else
    # torch first, from the cu129 index, so the vLLM wheel finds its exact dep satisfied.
    "$PY" -m pip install --index-url https://download.pytorch.org/whl/cu129 \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
    # llguidance: vLLM 0.22.1 needs >=1.7.0,<1.8.0, and llguidance's published Linux wheels are
    # manylinux_2_31 while these nodes run glibc 2.28 (Rocky 8.10). pip therefore falls back to the
    # sdist, which bootstraps a Rust toolchain and compiles for ~15+ min. The validated reference
    # env already contains a 1.7.6 abi3 build made on this same cluster, so we reuse it; the
    # `pip install` line below is the from-source path if that env ever disappears.
    LLG_SRC="${RTE_LLGUIDANCE_SRC:-/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/qwen_determinism/env/qwen38-vllm-cu129/lib/python3.12/site-packages}"
    SITE="$ENV_PREFIX/lib/python3.12/site-packages"
    if [ -d "$LLG_SRC/llguidance" ] && [ ! -d "$SITE/llguidance" ]; then
        cp -r "$LLG_SRC/llguidance" "$LLG_SRC/llguidance-1.7.6.dist-info" "$SITE/"
        echo "llguidance 1.7.6 vendored from $LLG_SRC (abi3, built on this cluster)"
    else
        "$PY" -m pip install "llguidance>=1.7.0,<1.8.0"
    fi
    "$PY" -m pip install "vllm @ ${VLLM_WHL}"
fi

# Benchmark deps. numpy is left at whatever vLLM resolved -- pinning it back to the reference
# env's 1.26.4 would fight vLLM's own numba/opencv pins, and the rte package is numpy-2 clean.
"$PY" -m pip install reasoning-gym openai pytest pandas scipy pyyaml \
    hnswlib trueskill scikit-learn

"$PY" - <<'PY'
import importlib, sys
ok = True
for m in ["numpy", "openai", "pytest", "pandas", "scipy", "yaml", "hnswlib", "trueskill", "sklearn",
          "reasoning_gym", "torch", "vllm"]:
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m:16s} {getattr(mod, '__version__', '?')}")
    except Exception as e:                                     # noqa: BLE001
        ok = False
        print(f"  FAIL {m:16s} {type(e).__name__}: {e}")
print("ENV_OK" if ok else "ENV_INCOMPLETE")
sys.exit(0 if ok else 1)
PY

echo "env at: $ENV_PREFIX"
