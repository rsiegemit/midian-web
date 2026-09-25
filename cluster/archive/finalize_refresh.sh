#!/bin/bash
# Compute-node half of finalize: re-audit, and ONLY if clean, regenerate every reported artefact from the corrected rows.
set -x
. "$(dirname "$(readlink -f "$0")")/../env.sh"
need RTE_DATA RTE_ACCOUNT
export PYTHONPATH="$RTE_REPO" MPLBACKEND=Agg
PY=$RTE_DATA/env/rte/bin/python; cd "$RTE_REPO"
$PY scripts/ops/quarantine_fallback_rows.py > $RTE_DATA/logs/final_audit.log 2>&1
grep -q "WOULD QUARANTINE 0 rows" $RTE_DATA/logs/final_audit.log || { echo "CONTAMINATION RECURRED -- not regenerating"; cat $RTE_DATA/logs/final_audit.log; exit 3; }
echo AUDIT_CLEAN
$PY scripts/paper_numbers.py && echo STEP_NUMBERS_OK
$PY scripts/doc_tables.py --sync && $PY scripts/doc_tables.py --verify && echo STEP_TABLES_OK
for f in bar_figs paper_figs extra_figs v3_figs shortlist_figs; do $PY scripts/$f.py && echo STEP_${f}_OK; done
echo FINALIZE_DONE
