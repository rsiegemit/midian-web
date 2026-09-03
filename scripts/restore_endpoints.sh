#!/bin/bash
# Re-register endpoints wiped by a fleet's exit handler (pre-990af66 fleets ran a global `clear`). Reads a snapshot dir of
# endpoints.d files and re-adds every entry whose serving job is still running (key "<model>#<jobid>", or the fleet job id).
# Usage: scripts/restore_endpoints.sh <snapshot_dir> [fleet_jobid_for_unaliased_keys]
set -uo pipefail
RTE_DATA="${RTE_DATA:-/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte}"; PY="$RTE_DATA/env/rte/bin/python"; REPO="$(dirname "$0")/.."
for f in "$1"/*.json; do
  key=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['model'])" "$f"); url=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['url'])" "$f")
  job=${key#*#}; [ "$job" = "$key" ] && job="${2:-}"
  [ -n "$job" ] && squeue -h -j "$job" -o %T 2>/dev/null | grep -q RUNNING && "$PY" "$REPO/scripts/_register_endpoint.py" add "$key" "$url" && echo "restored $key"
done
