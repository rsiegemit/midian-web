#!/usr/bin/env bash
# Fail if any tracked file that ships in the anonymous export names a person, lab, account, cluster or host.
#   scripts/checks/check_anonymity.sh [REPO]     (default: this repository)
# Patterns: one extended regex per line, case-insensitive, in anonymity_patterns.txt next to this script (override
# with ANON_PATTERNS=...). Paths marked `export-ignore` in .gitattributes (cluster/, the pattern file) are not checked,
# because `git archive` leaves them out. Prints every hit; exit 1 on any hit, 0 when clean.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PATTERNS=${ANON_PATTERNS:-$HERE/anonymity_patterns.txt}
REPO=${1:-$(git -C "$HERE" rev-parse --show-toplevel)}
cd "$REPO"
mapfile -t SKIP < <(git ls-files | git check-attr --stdin export-ignore | awk -F': ' '$3 == "set" {print ":(exclude)" $1}')
if git grep -n -I -i -E -f "$PATTERNS" -- . "${SKIP[@]}"; then
    echo "check_anonymity: FAILED ($(git grep -I -i -E -c -f "$PATTERNS" -- . "${SKIP[@]}" | awk -F: '{s += $NF} END {print s}') hits)" >&2
    exit 1
fi
echo "check_anonymity: clean"
