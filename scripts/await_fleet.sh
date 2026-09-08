#!/bin/bash
# Block until every model in configs/models.yaml actually answers /health, then exit 0.
#
#   scripts/await_fleet.sh [--timeout SECONDS] [--job JOBID] [--quiet]
#
# Why /health and not $RTE_DATA/endpoints.d: the registry is shared mutable state. A second fleet registers the SAME
# bare keys, so cancelling either one runs its cleanup trap and deregisters models the other is still serving; and a
# replica alias ("model#jobid") makes a naive file count read as "ready" while whole models are missing. Both cost us
# runs on 2026-09-06. The servers themselves are the only honest source of truth.
# --job JOBID aborts early (exit 2) if that fleet job disappears while models are still missing.
set -uo pipefail
cd "${RTE_REPO:-$HOME/rte}"; export RTE_DATA="${RTE_DATA:-/scratch/rte}"
TIMEOUT=""; JOB=""; QUIET=""
while [ $# -gt 0 ]; do case "$1" in
  --timeout) TIMEOUT="$2"; shift 2;; --job) JOB="$2"; shift 2;; --quiet) QUIET=1; shift;;
  *) echo "unknown arg $1" >&2; exit 64;; esac; done
PY="${RTE_PYTHON:-$RTE_DATA/env/rte/bin/python}"; START=$(date +%s)
while true; do
  MISSING=$("$PY" - <<'PYEOF'
import json, glob, os, urllib.request, yaml
want = {m["id"]: m["port"] for m in yaml.safe_load(open("configs/models.yaml"))["models"]}
host = None
for f in glob.glob(os.environ["RTE_DATA"] + "/endpoints.d/*.json"):
    try: host = json.load(open(f))["url"].split("//")[1].split(":")[0]
    except Exception: pass
missing = []
for mid, port in want.items():
    ok = False
    if host:
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=5) as r: ok = r.status == 200
        except Exception: ok = False
    if not ok: missing.append(mid)
print(",".join(sorted(missing)))
PYEOF
)
  [ -z "$MISSING" ] && { [ -n "$QUIET" ] || echo "[await_fleet] all $(grep -c 'id:' configs/models.yaml) models answering /health"; exit 0; }
  if [ -n "$JOB" ] && [ -z "$(squeue -j "$JOB" -h -o %T 2>/dev/null)" ]; then
    echo "[await_fleet] fleet job $JOB is gone; still missing: $MISSING" >&2; exit 2; fi
  if [ -n "$TIMEOUT" ] && [ $(( $(date +%s) - START )) -ge "$TIMEOUT" ]; then
    echo "[await_fleet] timed out after ${TIMEOUT}s; still missing: $MISSING" >&2; exit 3; fi
  [ -n "$QUIET" ] || echo "[await_fleet] $(date +%H:%M) waiting on: $MISSING"
  sleep 60
done
