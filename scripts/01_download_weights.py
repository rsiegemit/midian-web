#!/usr/bin/env python3
"""Download the RTE model ladder into $RTE_DATA/hf_cache (login node only -- compute nodes are offline).

    python scripts/01_download_weights.py                 # all models
    python scripts/01_download_weights.py --only Qwen/Qwen2.5-0.5B-Instruct

Full snapshots minus duplicate weight formats (.bin/.pth/.gguf/...), then verified with the
same offline `snapshot_download` call vLLM makes at load time.
Gemma is `gated=manual` on the Hub; if the token lacks access the model is SKIPPED and printed
in the final summary so the caller can record the fallback in DEVIATIONS.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

RTE_DATA = Path(os.environ.get("RTE_DATA", "/scratch/rte"))
os.environ.setdefault("HF_HOME", str(RTE_DATA / "hf_cache"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rte.backends.population import bands, ladder      # noqa: E402

LADDER = bands(ladder())[0]                            # configs/models.yaml, the only model list

# Download COMPLETE snapshots. An allow_patterns list looks tidy but breaks serving: vLLM calls
# `snapshot_download(repo, local_files_only=True)` with NO patterns, and huggingface_hub then
# raises IncompleteSnapshotError over the files the allow-list skipped -- for these seven repos
# that was .gitattributes / LICENSE / README.md, under 9 MiB in total (smoke job 43851943).
# ignore_patterns still drops the heavy duplicate weight formats, none of which these repos carry.
IGNORE = ["*.bin", "*.pth", "*.pt", "*.gguf", "*.onnx", "original/*", "*.msgpack", "*.h5"]


def _token() -> str | None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    for p in (Path.home() / ".huggingface" / "token", Path.home() / ".cache" / "huggingface" / "token"):
        if p.exists():
            return p.read_text().strip()
    return None


def du_bytes(path: Path) -> int:
    # hub snapshots are symlinks into the blob store: f.stat() follows them, which is what we want.
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=None, help="repo id (repeatable)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    token = _token()
    repos = args.only or LADDER
    done, skipped = [], []
    for repo in repos:
        t0 = time.time()
        print(f"\n=== {repo} ===", flush=True)
        try:
            path = snapshot_download(repo, ignore_patterns=IGNORE,
                                     token=token, max_workers=args.workers)
            # the exact call vLLM makes at load time -- fail here, not on the GPU node
            snapshot_download(repo, local_files_only=True)
            gb = du_bytes(Path(path)) / 2**30
            print(f"OK {repo}  {gb:.2f} GiB  {time.time() - t0:.0f}s  -> {path}", flush=True)
            done.append((repo, gb))
        except Exception as e:                                    # noqa: BLE001
            print(f"SKIP {repo}  {type(e).__name__}: {str(e)[:300]}", flush=True)
            skipped.append((repo, f"{type(e).__name__}: {str(e)[:200]}"))

    print("\n===== SUMMARY =====")
    for repo, gb in done:
        print(f"  downloaded  {repo:34s} {gb:7.2f} GiB")
    for repo, err in skipped:
        print(f"  SKIPPED     {repo:34s} {err}")
    print(f"HF_HOME={os.environ['HF_HOME']}")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
