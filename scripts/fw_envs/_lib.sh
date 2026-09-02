# Shared helper for framework venv builds (SPEC §6A). Sourced by <name>.sh.
# Creates $RTE_DATA/env/fw_<name> with python 3.12 and pip-installs requirements-frameworks/<name>.txt.
set -euo pipefail
RTE_DATA="${RTE_DATA:-/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$RTE_DATA/conda_pkgs}"   # honour a pre-set private cache
export PIP_CACHE_DIR="$RTE_DATA/pip_cache"
export TMPDIR="$RTE_DATA/tmp"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"

fw_build () {                                   # fw_build <name> [python-version]
  local name="$1"; local py="${2:-3.12}"; local prefix="$RTE_DATA/env/fw_$name"
  if [ -f /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh ]; then
    source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh
  else
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  fi
  # conda extracts into the shared package cache; concurrent creates corrupt it -> serialize with a lock
  [ -x "$prefix/bin/python" ] || flock "/tmp/rte_conda_create_$USER.lock" conda create -y -p "$prefix" "python=$py"
  "$prefix/bin/pip" install --no-input -q -r "$REPO/requirements-frameworks/$name.txt"
  "$prefix/bin/python" -c "import requests; print('$name env OK:', '$prefix')"
}
