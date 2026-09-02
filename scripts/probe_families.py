#!/usr/bin/env python3
"""Validate every family in rte.backends.llm's lists against the installed reasoning-gym.

For each family: generate instance seed 7, check the *gold* answer scores 1.0, a junk answer
scores < 0.99, and that regenerating from the same seed gives the identical question.
Prints one line per family and a PASS/FAIL summary. No GPU, no endpoints needed.

    python scripts/probe_families.py            # the K=64 list
    python scripts/probe_families.py --all      # every generator reasoning-gym registers
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from rte.backends import families   # noqa: E402


def check(name: str) -> tuple[bool, str]:
    try:
        e = families.entry(name, 7)
        if e.get("answer") is None:
            return False, "answer=None (no gold answer; unsolvable)"
        gold = families.correct(name, 7, str(e["answer"]))
        junk = families.correct(name, 7, "zzz_not_an_answer")
        det = families.question(name, 7) == families.question(name, 7)
        return bool(gold == 1 and junk == 0 and det), f"gold={gold} junk={junk} deterministic={det}"
    except Exception as ex:                                   # noqa: BLE001
        return False, f"{type(ex).__name__}: {str(ex)[:120]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        from reasoning_gym.factory import DATASETS
        names = sorted(DATASETS)
    else:
        names = families.FAMILIES_64
    bad = []
    for name in names:
        ok, why = check(name)
        tag = "OK  " if ok else "FAIL"
        mark = " [K=16]" if name in families.FAMILIES_16 else ""
        print(f"{tag} {name:32s} {why}{mark}", flush=True)
        if not ok:
            bad.append(name)
    print(f"\n{len(names) - len(bad)}/{len(names)} families usable")
    if bad:
        print("unusable:", ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
