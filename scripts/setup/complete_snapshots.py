#!/usr/bin/env python3
"""Complete every RTE model snapshot in $HF_HOME without huggingface_hub's file locking.

WHY THIS EXISTS.  vLLM loads a model with `snapshot_download(repo, local_files_only=True)` and NO
patterns, so a snapshot missing even a README refuses to boot (smoke job 43851943). Re-running the
downloader to fetch those files hangs, because `fcntl.flock` BLOCKS FOREVER on this cluster's
/n/netscratch mount -- where HF_HOME lives -- while the same call on $HOME or /tmp returns
instantly. huggingface_hub takes an flock per file, so any download into $HF_HOME wedges.

This script fetches the still-missing files over plain HTTP and writes them into the cache layout
itself (blob keyed by the file's remote etag/oid, snapshot entry symlinked to it), which is exactly
what hf_hub would have produced. Large files already present are left alone -- it only ever adds
what the snapshot lacks. Then it verifies each repo with the same offline call vLLM makes.

    python scripts/complete_snapshots.py [--repo Qwen/Qwen2.5-7B-Instruct]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

RTE_DATA = Path(os.environ.get("RTE_DATA", "/scratch/rte"))
os.environ.setdefault("HF_HOME", str(RTE_DATA / "hf_cache"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rte.backends.population import bands, ladder  # noqa: E402

REPOS = bands(ladder())[0]        # configs/models.yaml is the one source of truth


def token() -> str | None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    for p in (Path.home() / ".huggingface" / "token", Path.home() / ".cache" / "huggingface" / "token"):
        if p.exists():
            return p.read_text().strip()
    return None


def main() -> int:
    import requests
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", default=None)
    args = ap.parse_args()

    tok = token()
    api = HfApi(token=tok)
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    bad = []

    for repo in (args.repo or REPOS):
        info = api.model_info(repo)
        sha = info.sha
        root = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}"
        snap = root / "snapshots" / sha
        blobs = root / "blobs"
        snap.mkdir(parents=True, exist_ok=True)
        blobs.mkdir(parents=True, exist_ok=True)

        added = 0
        for f in info.siblings:
            name = f.rfilename
            dest = snap / name
            if dest.exists() or dest.is_symlink():
                continue
            url = f"https://huggingface.co/{repo}/resolve/{sha}/{name}"
            r = requests.get(url, headers=headers, timeout=120)
            r.raise_for_status()
            oid = r.headers.get("X-Linked-Etag") or r.headers.get("ETag") or ""
            oid = oid.strip('"').replace("W/", "") or f"sha-{abs(hash(name)):016x}"
            blob = blobs / oid
            if not blob.exists():
                tmp = blobs / f".{oid}.partial"
                tmp.write_bytes(r.content)
                os.replace(tmp, blob)
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(os.path.relpath(blob, dest.parent), dest)
            added += 1
            print(f"  + {repo}/{name} ({len(r.content)} bytes)", flush=True)

        try:
            p = snapshot_download(repo, local_files_only=True)
            print(f"COMPLETE {repo} (+{added} files) -> {p}", flush=True)
        except Exception as e:                                   # noqa: BLE001
            print(f"INCOMPLETE {repo}: {type(e).__name__}: {str(e)[:250]}", flush=True)
            bad.append(repo)

    print(f"\n{'ALL SNAPSHOTS COMPLETE' if not bad else 'STILL INCOMPLETE: ' + ', '.join(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
