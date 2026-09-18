#!/bin/bash
# Submit the SOTA-retrieval framework campaign, ONE UNIT PER JOB (the documented pattern for llm grids).
#   scripts/launch_sota.sh [--dry]
# Resumable: every submission is appended to $LOG and a re-run skips what is already there, so a killed or
# timed-out launcher is restarted by running it again -- never by starting from scratch (that is how the
# 2026-09-16 campaign produced 937 duplicates). One job covers all four arms of one framework in one cell and
# seed, because --methods filters by NAME and the arms differ only in params.
set -uo pipefail
export RTE_DATA="${RTE_DATA:-/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte}"
PY="$RTE_DATA/env/rte/bin/python"
LOG="$RTE_DATA/logs/launch_sota.txt"
DRY=${1:-}
mkdir -p "$(dirname "$LOG")"; touch "$LOG"
cd "$HOME/rte"; export PYTHONPATH="$PWD"

# (grid, method, only-filter, seed) per line, from the grid loader itself
"$PY" - <<'EOF' > /tmp/sota_units.$$
import sys, yaml
sys.path.insert(0, '/n/home02/rsiegelmann/rte')
from rte.run import blocks, cells, method_specs, seeds
cfg = yaml.safe_load(open('configs/grid.yaml'))
GRIDS = ["fw_live_n100k_sota", "fw_live_n10k_sota", "fw_live_n10k_cartel_sota",      # cheap + highest value first
         "fw_live_n1000_sota", "fw_live_n1000_lowskill_sota", "fw_live_n100_sota", "fw_live_n100_lowskill_sota"]
AXES = ["dist", "beta", "liar_select"]
for g in GRIDS:
    for blk in blocks(cfg, g):
        names = sorted({s["name"] for s in method_specs(blk)})
        for cell in cells(blk):
            only = ",".join(f"{a}={cell[a]}" for a in AXES)
            for seed in seeds(blk["seeds"]):
                for m in names:
                    print(f"{g}\t{m}\t{only}\t{seed}")
EOF
n=$(wc -l < /tmp/sota_units.$$); echo "units x methods to submit: $n  (already logged: $(wc -l < "$LOG"))"
[ "$DRY" = "--dry" ] && { head -3 /tmp/sota_units.$$; rm -f /tmp/sota_units.$$; exit 0; }

sub=0; skip=0
while IFS=$'\t' read -r g m only seed; do
    key="$g|$m|$only|$seed"
    grep -qF "$key" "$LOG" && { skip=$((skip+1)); continue; }
    jid=$(sbatch --parsable -p sapphire,serial_requeue -A sompolinsky_lab \
        --job-name="rte_${g}__${m}" -c 2 --mem=32G --time=1-00:00:00 \
        -o "$RTE_DATA/logs/sota/%x-%j.out" -e "$RTE_DATA/logs/sota/%x-%j.err" \
        --export=ALL,RTE_PYTHON="$PY",RTE_WORKERS=1,RTE_CONSOLIDATE=0 \
        scripts/run_grid.sbatch "$g" --methods "$m" --only "$only" --seeds "$seed" 2>&1)
    case "$jid" in
        ''|*[!0-9]*) echo "FAIL $key :: $jid" >&2; sleep 5;;
        *) echo "$jid $key" >> "$LOG"; sub=$((sub+1));;
    esac
    [ $((sub % 200)) -eq 0 ] && [ $sub -gt 0 ] && echo "  ... $sub submitted" && sleep 2
done < /tmp/sota_units.$$
rm -f /tmp/sota_units.$$
echo "SUBMITTED $sub, SKIPPED $skip -> $LOG"
