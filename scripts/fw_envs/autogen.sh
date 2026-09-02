#!/usr/bin/env bash
# Reproducible build of $RTE_DATA/env/fw_autogen (SPEC §6A). Login node only (needs internet).
# conda's repodata cache takes an exclusive lock, so parallel builds are serialized via flock + retry.
set -euo pipefail
RTE_DATA="${RTE_DATA:-/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CONDA_PKGS_DIRS="$RTE_DATA/conda_pkgs" PIP_CACHE_DIR="$RTE_DATA/pip_cache" TMPDIR="$RTE_DATA/tmp"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"
CONDA_SH=/n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh
[ -f "$CONDA_SH" ] || CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
source "$CONDA_SH"
PREFIX="$RTE_DATA/env/fw_autogen"
for try in 1 2 3 4 5; do
  [ -x "$PREFIX/bin/python" ] && break
  flock "$CONDA_PKGS_DIRS/.rte.lock" conda create -y -p "$PREFIX" python=3.12 && break
  sleep $((try * 20))
done
# conda envs put ~/.local/lib/python3.12/site-packages on sys.path, which would let pip skip deps that
# only exist in the user site and leave the env broken. Cut it out before installing anything.
cat > "$PREFIX/lib/python3.12/site-packages/zzz_no_user_site.pth" <<'PTH'
import sys, os; sys.path[:] = [p for p in sys.path if not p.startswith(os.path.expanduser('~/.local/'))]
PTH
"$PREFIX/bin/pip" install --no-input -q -r "$REPO/requirements-frameworks/autogen.txt"
"$PREFIX/bin/python" -c "import requests; print('fw_autogen OK')"
