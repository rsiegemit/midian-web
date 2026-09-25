#!/bin/bash
# Cancel the 4-GPU fleets only once the 1-GPU replacements actually SERVE the supervisor, so there is never a window
# with zero supervisor replicas (the 2026-09-19 outage). Polls endpoints.json; gives up after 12 h, leaving old ones up.
. "$(dirname "$(readlink -f "$0")")/../env.sh"
need RTE_DATA RTE_ACCOUNT
NEW="47797868 47797875 47797886"; OLD="47315656 47315663 47626219"
for i in $(seq 1 720); do
  up=$($RTE_DATA/env/rte/bin/python -c "
import json
d = json.load(open('$RTE_DATA/endpoints.json'))
print(sum(1 for k in d if k.startswith('Qwen/Qwen2.5-7B') and k.split('#')[-1] in '$NEW'.split()))" 2>/dev/null || echo 0)
  if [ "${up:-0}" -ge 2 ]; then
    echo "$(date -Is) $up new supervisor replicas serving -> cancelling old 4-GPU fleets $OLD"
    scancel $OLD; echo RETIRED; exit 0
  fi
  sleep 60
done
echo "$(date -Is) TIMEOUT: new replicas never reached 2; old fleets left running on purpose"
