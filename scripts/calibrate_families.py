#!/usr/bin/env python3
"""Measure one served model's accuracy per family, unhandicapped vs handicapped.

The population only works if expertise EXISTS to be discovered: SPEC §3 wants a `specialist`
agent near 0.70-0.95 on its specialty families and 0.05-0.30 elsewhere. That needs the family
generators to sit at a difficulty the ladder can partly solve, and it needs the handicap to be
MONOTONE. This answers both for whatever is served, before a full measurement is paid for.

    $RTE_DATA/env/rte/bin/python scripts/calibrate_families.py --probes 20 [--tools]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                                          # noqa: E402

from rte import llm_client                                                  # noqa: E402
from rte.backends import families, llm, population, tools                   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", type=int, default=20)
    ap.add_argument("--families", type=int, default=8)
    ap.add_argument("--model", default=None)
    ap.add_argument("--tools", action="store_true", help="compare tools instead of the handicap")
    a = ap.parse_args()

    model = a.model or sorted(llm_client.endpoints())[0]
    fams = families.names(16)[:a.families]
    cfg = population.pinned_cfg([model])
    llm.ladder = population.ladder = lambda: cfg
    print(f"model: {model}\nfamilies: {fams}\nprobes per cell: {a.probes}\n")

    b = llm.LLMBackend(n=3, K=len(fams), dist="specialist", seed=1, families=fams, concurrency=32)
    if a.tools:                       # unhandicapped everywhere, one arm per tool
        arms = list(tools.NAMES)
        for i, t in enumerate(arms):
            b.profiles[i] = dict(id=i, model=model, specialty=list(range(len(fams))), tool=t)
    else:                             # the two ends of the profile space bound every profile
        arms = ["unhandicapped", "handicapped"]
        b.profiles[0] = dict(id=0, model=model, specialty=list(range(len(fams))), tool="none")
        b.profiles[1] = dict(id=1, model=model, specialty=[], tool="none")

    print(f"{'family':24s}" + "".join(f"{x:>14s}" for x in arms))
    acc = np.zeros((len(fams), len(arms)))
    for f, fam in enumerate(fams):
        out = b.execute_many(np.arange(len(arms)), np.full(len(arms), f), a.probes,
                             np.random.default_rng(1000 + f))
        acc[f] = out.mean(axis=1)
        print(f"{fam:24s}" + "".join(f"{v:14.2f}" for v in acc[f]), flush=True)

    print(f"\n{'MEAN':24s}" + "".join(f"{v:14.2f}" for v in acc.mean(axis=0)))
    solvable = [f for f, v in zip(fams, acc[:, 0]) if v >= 0.3]
    print(f"\nfamilies at >=0.30 on the first arm ({len(solvable)}/{len(fams)}): {solvable}")
    if not a.tools:
        gap = acc[:, 0] - acc[:, 1]
        print(f"handicap gap (unhandicapped - handicapped): mean {gap.mean():+.3f}, "
              f"min {gap.min():+.3f} -- a NEGATIVE min means the handicap helps somewhere")
    print(f"stats: {b.stats()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
