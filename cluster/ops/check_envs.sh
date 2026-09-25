#!/bin/bash
# Environment integrity gate (OPS_RULES.md R9). Run before every campaign; exits non-zero on any failure.
#   cluster/ops/check_envs.sh
# Two checks per env, both on a BARE environment (LD_LIBRARY_PATH unset) so a job's result cannot depend on which
# node it lands on:
#   1. no dangling symlinks  -- on 2026-09-22, 7 framework envs had 98 each (libffi, libbz2, libreadline, libsqlite3
#                               targets deleted), and jobs silently degraded to declared-argmax picks on some nodes
#   2. the stdlib C extensions import (ctypes, sqlite3, ssl, bz2, lzma, zlib, decimal, uuid)
. "$(dirname "$(readlink -f "$0")")/../env.sh"
need RTE_DATA
bad=0
for e in "$RTE_DATA"/env/*/; do
  n=$(basename "$e"); P="$e/bin/python"; [ -x "$P" ] || continue
  d=$(find "$e" -xtype l -path "*/lib/*" 2>/dev/null | wc -l)
  imp=$(env -u LD_LIBRARY_PATH "$P" -c "
bad=[]
for m in ('ctypes','sqlite3','ssl','bz2','lzma','zlib','decimal','uuid'):
    try: __import__(m)
    except Exception as x: bad.append(m)
print(','.join(bad) or 'ok')" 2>&1 | tail -1)
  if [ "$d" -eq 0 ] && [ "$imp" = ok ]; then printf "  ok    %-20s\n" "$n"
  else printf "  FAIL  %-20s dangling lib links: %-4s failed imports: %s\n" "$n" "$d" "$imp"; bad=1; fi
done
[ $bad -eq 0 ] && echo "ALL ENVS OK" || { echo "ENV CHECK FAILED -- repair before launching (cluster/ops/repair_fw_envs.sh)"; exit 1; }
