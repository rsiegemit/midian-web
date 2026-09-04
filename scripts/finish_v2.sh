#!/bin/bash
# Unattended wrap-up of the v2 programme: wait until no rte_ method job remains, then analyze every grid, regenerate the
# figures, and commit the artefacts. Submit with: sbatch -p shared -c 2 --mem=48G -t 2-00:00:00 scripts/finish_v2.sh
#SBATCH --job-name=rte_finish_v2
#SBATCH --output=/scratch/rte/logs/finish_v2.out
#SBATCH --error=/scratch/rte/logs/finish_v2.err
set -uo pipefail
cd ~/rte; export RTE_DATA="${RTE_DATA:-/scratch/rte}" PYTHONPATH=~/rte
source "$RTE_DATA/env/rte/bin/activate"
quiet=0                                   # need 3 consecutive empty polls; a failing squeue counts as "still running"
while [ $quiet -lt 3 ]; do
  out=$(squeue -u "$USER" -h -o %j 2>/dev/null) || { sleep 600; continue; }
  echo "$out" | grep -q finish_v2 || { sleep 600; continue; }               # a listing without this very job is a failed poll, not "all done"
  if echo "$out" | grep -E '^rte_' | grep -vqE 'serve|finish_v2'; then quiet=0; else quiet=$((quiet+1)); fi
  sleep 600
done
GRIDS="fw_live_n100 fw_live_n1000 fw_live_n100_verified fw_live_n1000_verified midian_v_replication internals_v2 variants_f1 midian_r20 stratify churn_n1000 live_n10k_v2 budget_b10_shapes live_f1_core_s6_10 live_f1_n1000"
for g in $GRIDS; do python -m rte.analyze --grid "$g" || echo "analyze $g failed"; done
python -m rte.analyze --grid variants_f1 --grids stratify,churn_n1000,live_n10k_v2,midian_v_replication,budget_b10_shapes,internals_v2,live_f1_n1000,live_f1_core_s6_10 \
  --out "$RTE_DATA/results/v2_targets" || echo "targets_v2 merge failed"
python scripts/extra_figs.py || echo "figures failed"
mkdir -p figures && cp "$RTE_DATA"/results/extra_figs/*.png figures/
date=$(date +%F\ %H:%M)
printf '\n**%s — finish_v2 (unattended): all v2 grids closed (3 consecutive empty polls); analyses, targets_v2 merge (results/v2_targets) and figures regenerated. Next: fill RESULTS_rte_v2.md TODO(grid) markers from the summaries.**\n' "$date" >> STATUS.md
git add figures STATUS.md && git commit -qm "finish_v2: analyses and figures after the v2 grids closed ($date)" && git push -q origin master:main
echo FINISH_V2_DONE
