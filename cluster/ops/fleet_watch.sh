#!/bin/bash
# Watch fleet health and EXIT on any material change, so the change surfaces as a task notification rather than
# needing a poll. Material = supervisor replica count changes, or it hits zero (the 2026-09-19 outage: both fleets
# hit their 2-day cap within hours of each other and 13 h of framework jobs failed before anyone noticed).
export RTE_DATA=/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte
sup() { $RTE_DATA/env/rte/bin/python -c "
import json,os
p='$RTE_DATA/endpoints.json'
try: d=json.load(open(p))
except Exception: d={}
print(sum(1 for k in d if k.startswith('Qwen/Qwen2.5-7B')))" 2>/dev/null || echo 0; }
A=$(sup); echo "$(date -Is)  baseline: $A supervisor replica(s), $(squeue -u $USER -h -o '%j %T' | grep rte_serve_fleet | grep -c ' R') fleet job(s) running"
for i in $(seq 1 480); do          # 8 h max
  sleep 60
  B=$(sup)
  if [ "$B" != "$A" ]; then
    echo "$(date -Is)  CHANGE: supervisor replicas $A -> $B"
    squeue -u "$USER" -h -o "  %.10i %.2t %.14P %.10M %R" | grep -E "$(squeue -u $USER -h -o '%i %j' | grep rte_serve_fleet | awk '{printf "%s|",$1}' | sed 's/|$//')" 2>/dev/null
    [ "$B" -eq 0 ] && echo "  *** FLEET DOWN -- framework jobs will fail with 'endpoints.json has []' ***"
    exit 0
  fi
done
echo "$(date -Is)  8 h elapsed, steady at $A supervisor replica(s)"
