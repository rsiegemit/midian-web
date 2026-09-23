#!/bin/bash
#SBATCH -p test
#SBATCH -A sompolinsky_lab
#SBATCH -c 1 --mem=2G -t 00:20:00
#SBATCH -J rte_campaign_tick
#SBATCH -o /n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte/logs/campaign_tick.log
#SBATCH --open-mode=append
# Session-independent campaign driver (OPS_RULES H6). On `test`: a 20-min bookkeeping job must not queue behind the
# campaign's own fairshare (it sat at (Priority) on sapphire). Runs for seconds, then re-submits itself 30 min later, so it never
# dies with a conversation (nohup'd session helpers did, 2026-09-22) and never holds a CPU while waiting. Each tick:
#   1. keeps the rerun submitter alive until every unit in quarantine_units.tsv is submitted
#   2. alerts (logs/ALERT_fleet) if fewer than 2 supervisor replicas are serving (OPS_RULES F2)
#   3. stage 1: once all units are submitted and every HEADLINE rerun + lietext_th job has drained -> finalize_stage1
#   4. stage 2: once every rerun job (ablation included) has drained -> finalize_stage2, and stop ticking
#   sbatch scripts/ops/campaign_tick.sh          # start (idempotent: a tick that finds another queued exits)
set -u
export RTE_DATA=/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte
L=$RTE_DATA/logs; REPO=/n/home02/rsiegelmann/rte; PY=$RTE_DATA/env/rte/bin/python; cd $REPO
say() { echo "$(date -Is) $*"; }
ABL='fw_live_n(1000|100)(_lowskill)?_sota'
q() { squeue -u "$USER" -h -o "%i %j %T"; }

# 1. rerun submitter: resumable, so re-launching it is always safe
plan=$($PY -c "import pandas as pd; print(len(pd.read_csv('$RTE_DATA/results/quarantine_units.tsv', sep=chr(9))))")
subd=$(wc -l < $L/rerun_quarantine.txt 2>/dev/null || echo 0)
HB=$L/rerun_submitter.heartbeat                        # a live submitter (SLURM or login-node setsid) touches this
fresh=$([ -e $HB ] && [ $(( $(date +%s) - $(stat -c %Y $HB) )) -lt 900 ] && echo 1 || echo 0)
if [ "$subd" -lt "$plan" ] && [ "$fresh" = 0 ] && ! q | grep -q " rte_rerun_submitter "; then
  sbatch -p test -A sompolinsky_lab -c 1 --mem=4G -t 11:00:00 -J rte_rerun_submitter \
    -o $L/rerun_submitter.log --open-mode=append --wrap="$REPO/scripts/ops/rerun_units.sh" >/dev/null && say "submitter (re)started: $subd/$plan submitted"
fi

# 2. fleet health
rep=$($PY -c "import json; print(sum(k.startswith('Qwen/Qwen2.5-7B') for k in json.load(open('$RTE_DATA/endpoints.json'))))" 2>/dev/null || echo 0)
[ "$rep" -lt 2 ] && { say "ALERT only $rep supervisor replica(s)"; echo "$(date -Is) $rep supervisor replicas" >> $L/ALERT_fleet; }

# 3-4. finalize stages, gated on explicit job IDs from the launcher logs
ids() { cat $L/rerun_quarantine.txt $L/launch_lietext_th.txt $L/resubmit_parallel.txt 2>/dev/null | grep -Ev "$1" | awk '{print $1}' | sort -u; }
live() { comm -12 <(ids "$1") <(squeue -u "$USER" -h -o "%i" | sort -u) | wc -l; }
fin() {
  $PY scripts/ops/build_job_sizing.py
  sbatch -p test -A sompolinsky_lab -c 8 --mem=96G -t 4:00:00 -J "finalize_$1" \
    -o $L/finalize_$1.log -e $L/finalize_$1.err $REPO/scripts/ops/finalize_refresh.sh >/dev/null && touch $L/DONE_$1 && say "finalize_$1 submitted"
}
if [ "$subd" -ge "$plan" ]; then
  h=$(live "$ABL"); a=$(live '^$'); say "all $plan submitted; headline/lietext live=$h, all live=$a, replicas=$rep"
  [ ! -e $L/DONE_stage1 ] && [ "$h" -eq 0 ] && fin stage1
  [ ! -e $L/DONE_stage2 ] && [ "$a" -eq 0 ] && { fin stage2; say "campaign complete; tick stops"; exit 0; }
else
  say "submitted $subd/$plan; replicas=$rep"
fi

# re-schedule (only if no other tick is already queued -- keeps exactly one tick alive)
[ "$(q | grep -c ' rte_campaign_tick PENDING')" -eq 0 ] && sbatch --begin=now+30minutes $REPO/scripts/ops/campaign_tick.sh >/dev/null
