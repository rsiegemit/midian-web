#!/usr/bin/env python3
"""Register/deregister one served model in the fleet's endpoint list. LOCK-FREE BY DESIGN.

    python _register_endpoint.py add <model_id> <base_url>
    python _register_endpoint.py remove <model_id>
    python _register_endpoint.py clear
    python _register_endpoint.py list

WHY NO flock.  `fcntl.flock` BLOCKS FOREVER on this cluster's /n/netscratch mount -- verified
2026-09-02: an flock on a fresh file under $RTE_DATA never returns, while the same call on $HOME
or /tmp returns instantly. That is also what hangs `huggingface_hub`'s downloader when HF_HOME
lives there. (sqlite is unaffected: it uses POSIX fcntl record locks, which do work here.)

So instead of one shared file guarded by a lock, each server owns ONE file:

    $RTE_DATA/endpoints.d/<slug>.json     {"model": ..., "url": ...}      <- authoritative
    $RTE_DATA/endpoints.json              {model: url, ...}               <- merged, for the contract

Single-writer-per-file means no lock is needed; every write lands via a same-directory temp file
plus os.replace, which is atomic. endpoints.json is regenerated from the directory after each
change -- a concurrent regeneration can briefly publish a stale merge, which the next registration
repairs, and `rte.llm_client` reads the directory in preference to the merged file anyway.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

RTE_DATA = Path(os.environ.get("RTE_DATA", "/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte"))
MERGED = Path(os.environ.get("RTE_ENDPOINTS", RTE_DATA / "endpoints.json"))
ENDPOINT_DIR = Path(os.environ.get("RTE_ENDPOINT_DIR", MERGED.parent / "endpoints.d"))


def slug(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_dir() -> dict[str, str]:
    out: dict[str, str] = {}
    if not ENDPOINT_DIR.is_dir():
        return out
    for p in sorted(ENDPOINT_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text())
            if d.get("model") and d.get("url"):
                out[d["model"]] = d["url"]
        except (OSError, json.JSONDecodeError):
            continue                      # a half-written file cannot happen (atomic replace), but be safe
    return out


def republish() -> dict[str, str]:
    merged = read_dir()
    atomic_write_json(MERGED, merged)
    return merged


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    action = sys.argv[1]
    ENDPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if action == "add":
        model, url = sys.argv[2], sys.argv[3]
        atomic_write_json(ENDPOINT_DIR / f"{slug(model)}.json", {"model": model, "url": url})
    elif action == "remove":
        (ENDPOINT_DIR / f"{slug(sys.argv[2])}.json").unlink(missing_ok=True)
    elif action == "clear":
        for p in ENDPOINT_DIR.glob("*.json"):
            p.unlink(missing_ok=True)
    elif action == "list":
        print(json.dumps(read_dir(), indent=2, sort_keys=True))
        return 0
    else:
        print(f"unknown action {action!r}", file=sys.stderr)
        return 2

    merged = republish()
    print(f"{action}: {len(merged)} endpoint(s) -> {MERGED} (source of truth: {ENDPOINT_DIR})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
