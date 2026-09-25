#!/usr/bin/env bash
# Build the anonymous review copy of HEAD: git archive (drops export-ignore paths) -> fresh repository with one commit
# by "Anonymous Authors" -> anonymity check. Prints the path; pushes nothing.
#   scripts/export_anonymous.sh [OUT_DIR]      (default: a new temporary directory)
set -euo pipefail
SRC=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
OUT=${1:-$(mktemp -d -t rte_anonymous.XXXXXX)}
mkdir -p "$OUT"
[ -z "$(ls -A "$OUT")" ] || { echo "export_anonymous: $OUT is not empty" >&2; exit 1; }
git -C "$SRC" archive --format=tar HEAD | tar -x -C "$OUT"
cd "$OUT"
git init -q
git add -A
"$SRC/scripts/checks/check_anonymity.sh" "$OUT"
GIT_AUTHOR_NAME="Anonymous Authors" GIT_AUTHOR_EMAIL="anonymous@example.com" \
GIT_COMMITTER_NAME="Anonymous Authors" GIT_COMMITTER_EMAIL="anonymous@example.com" \
GIT_AUTHOR_DATE="2026-01-01T00:00:00Z" GIT_COMMITTER_DATE="2026-01-01T00:00:00Z" \
    git -c commit.gpgsign=false commit -q -m "Anonymous code release"
echo "$OUT"
