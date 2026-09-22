#!/bin/bash
# Restore files deleted out from under the framework conda envs (symlinks survived, their targets did not -- see
# CHANGES_AND_ERRATA erratum 28). Byte-identical: force-reinstall exactly the owning package BUILDS, offline, from the
# package cache, so no framework version changes relative to rows already collected. No flock (hangs on netscratch).
export RTE_DATA=/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte CONDA_PKGS_DIRS=$RTE_DATA/conda_pkgs
source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh 2>/dev/null || source $HOME/miniconda3/etc/profile.d/conda.sh
for e in fw_autogen fw_crewai fw_google_adk fw_langgraph fw_llamaindex fw_maf fw_openai_agents fw_agentscope fw_camel fw_metagpt fw_smolagents; do
  E=$RTE_DATA/env/$e
  specs=$(python3 - "$E" <<'PY'
import glob, json, os, sys
E = sys.argv[1]
missing = set()
for root, _, files in os.walk(E):
    for f in files:
        p = os.path.join(root, f)
        if os.path.islink(p) and not os.path.exists(p):
            missing.add(os.path.relpath(os.path.join(root, os.readlink(p)) if not os.path.isabs(os.readlink(p)) else os.readlink(p), E))
out = set()
for m in glob.glob(f"{E}/conda-meta/*.json"):
    d = json.load(open(m))
    if missing & set(d.get("files", [])): out.add(f"{d['name']}={d['version']}={d['build']}")
print(" ".join(sorted(out)))
PY
)
  before=$(find $E -xtype l | wc -l)
  if [ -z "$specs" ]; then echo "$e: $before dangling, no owning conda package found"; continue; fi
  conda install -y -q -p $E --offline --force-reinstall --no-deps $specs > $RTE_DATA/logs/repair_$e.log 2>&1
  after=$(find $E -xtype l | wc -l)
  echo "$e: dangling $before -> $after   (reinstalled: $specs)"
done
echo REPAIR_DONE
