#!/usr/bin/env bash
# Install the CPU deps the decentralized rivals need into the base python
# (~/miniconda3/bin/python), user site. Per CONTRACT.md this is acceptable until
# $RTE_DATA/env/rte exists; re-run there when it does.
#   usage: bash ~/rte/scripts/03_install_deps.sh
set -euo pipefail
PY="${PY:-$HOME/miniconda3/bin/python}"
# Keep pip's build/cache off the (nearly full) home quota.
export RTE_DATA="${RTE_DATA:-/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte}"
export TMPDIR="$RTE_DATA/tmp"; mkdir -p "$TMPDIR"
export PIP_CACHE_DIR="$RTE_DATA/pipcache"; mkdir -p "$PIP_CACHE_DIR"
"$PY" -m pip install --user --upgrade pip setuptools wheel pybind11
"$PY" -m pip install --user hnswlib scipy
"$PY" - <<'PYEOF'
import hnswlib, scipy, numpy as np
i = hnswlib.Index(space="ip", dim=4); i.init_index(max_elements=8, M=8, ef_construction=16)
i.set_num_threads(1)
i.add_items(np.eye(4, dtype=np.float32)[:4], np.arange(4))
i.set_ef(8)
print("hnswlib", hnswlib.__file__, "query ->", i.knn_query(np.eye(4, dtype=np.float32)[2:3], k=1)[0])
print("scipy", scipy.__version__)
PYEOF
