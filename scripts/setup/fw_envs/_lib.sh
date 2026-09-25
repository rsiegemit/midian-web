# Shared helper for framework venv builds (SPEC §6A). Sourced by <name>.sh.
# Creates $RTE_DATA/env/fw_<name> and pip-installs requirements-frameworks/<name>.txt.
# conda comes from $RTE_CONDA_SH when set (cluster/cluster.env.example), else ~/miniconda3.
set -euo pipefail
: "${RTE_DATA:?RTE_DATA is not set}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$RTE_DATA/conda_pkgs}"   # honour a pre-set private cache
export PIP_CACHE_DIR="$RTE_DATA/pip_cache"
export TMPDIR="$RTE_DATA/tmp"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"

fw_conda () {                                   # source conda.sh
  local sh="${RTE_CONDA_SH:-}"
  [ -n "$sh" ] && [ -f "$sh" ] || sh="$HOME/miniconda3/etc/profile.d/conda.sh"
  source "$sh"
}

fw_build () {                                   # fw_build <name> [python-version]
  local name="$1"; local py="${2:-3.12}"; local prefix="$RTE_DATA/env/fw_$name"
  fw_conda
  # conda extracts into the shared package cache; concurrent creates corrupt it -> serialize with a lock
  [ -x "$prefix/bin/python" ] || flock "/tmp/rte_conda_create_$USER.lock" conda create -y -p "$prefix" "python=$py"
  "$prefix/bin/pip" install --no-input -q -r "$REPO/requirements-frameworks/$name.txt"
  "$prefix/bin/python" -c "import requests; print('$name env OK:', '$prefix')"
}

fw_build_isolated () {                          # fw_build_isolated <name>: python 3.12, create retried, user site cut out
  local name="$1"; local prefix="$RTE_DATA/env/fw_$name"
  export CONDA_PKGS_DIRS="$RTE_DATA/conda_pkgs"
  mkdir -p "$CONDA_PKGS_DIRS"
  fw_conda
  # conda's repodata cache takes an exclusive lock, so parallel builds are serialized via flock + retry.
  for try in 1 2 3 4 5; do
    [ -x "$prefix/bin/python" ] && break
    flock "$CONDA_PKGS_DIRS/.rte.lock" conda create -y -p "$prefix" python=3.12 && break
    sleep $((try * 20))
  done
  # conda envs put ~/.local/lib/python3.12/site-packages on sys.path, which would let pip skip deps that
  # only exist in the user site and leave the env broken. Cut it out before installing anything.
  cat > "$prefix/lib/python3.12/site-packages/zzz_no_user_site.pth" <<'PTH'
import sys, os; sys.path[:] = [p for p in sys.path if not p.startswith(os.path.expanduser('~/.local/'))]
PTH
  "$prefix/bin/pip" install --no-input -q -r "$REPO/requirements-frameworks/$name.txt"
  "$prefix/bin/python" -c "import requests; print('fw_$name OK')"
}
