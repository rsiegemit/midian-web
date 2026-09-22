#!/bin/bash
# Restore files deleted out from under the framework conda envs (OPS_RULES D2/D3). The symlinks survived; their
# targets did not -- and the extracted package cache lost the same files (hardlinked, same purge), so the only intact
# source is the compressed .conda ARCHIVE. This extracts exactly the missing members from the owning package's archive:
# byte-identical, no conda solve, no locks (conda's cache lock hangs on netscratch -- a --force-reinstall sat 6+ min).
#   scripts/ops/repair_fw_envs.sh [env ...]        # default: every env under $RTE_DATA/env with dangling lib links
export RTE_DATA="${RTE_DATA:-/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte}"
PKGS="$RTE_DATA/conda_pkgs"; ZSTD=$(command -v zstd || echo "$HOME/miniconda3/bin/zstd")
envs=("$@"); [ ${#envs[@]} -eq 0 ] && for e in "$RTE_DATA"/env/*/; do [ -d "$e/conda-meta" ] && envs+=("$(basename "$e")"); done
for e in "${envs[@]}"; do
  E="$RTE_DATA/env/$e"; before=$(find "$E" -xtype l | wc -l)
  [ "$before" -eq 0 ] && { echo "$e: clean"; continue; }
  python3 - "$E" "$PKGS" "$ZSTD" <<'PY'
import glob, io, json, os, subprocess, sys, tarfile, zipfile
E, PKGS, ZSTD = sys.argv[1:4]
missing = {}
for root, _, files in os.walk(E):
    for f in files:
        p = os.path.join(root, f)
        if os.path.islink(p) and not os.path.exists(p):
            t = os.path.normpath(os.path.join(root, os.readlink(p)))
            if t.startswith(E + "/"): missing[os.path.relpath(t, E)] = t
owner = {}
for m in glob.glob(f"{E}/conda-meta/*.json"):
    d = json.load(open(m))
    for f in d.get("files", []):
        if f in missing: owner.setdefault(f"{d['name']}-{d['version']}-{d['build']}", []).append(f)
done = 0
for pkg, rels in owner.items():
    arc = f"{PKGS}/{pkg}.conda"
    if not os.path.exists(arc): print(f"   no archive for {pkg}"); continue
    z = zipfile.ZipFile(arc)
    raw = z.read(next(n for n in z.namelist() if n.startswith("pkg-")))
    tar = tarfile.open(fileobj=io.BytesIO(subprocess.run([ZSTD, "-dc"], input=raw, capture_output=True, check=True).stdout))
    for rel in rels:
        try: mem = tar.getmember(rel)
        except KeyError: print(f"   {rel} not in {pkg}"); continue
        dst = missing[rel]; os.makedirs(os.path.dirname(dst), exist_ok=True)
        if mem.issym():
            if os.path.lexists(dst): os.remove(dst)
            os.symlink(mem.linkname, dst)
        else:
            with open(dst, "wb") as fh: fh.write(tar.extractfile(mem).read())
            os.chmod(dst, mem.mode)
        done += 1
print(f"   restored {done} of {len(missing)} missing targets from {len(owner)} package archives")
PY
  echo "$e: dangling lib links $before -> $(find "$E" -xtype l | wc -l)"
done
