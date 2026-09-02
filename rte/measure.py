"""Build one population, MEASURE its true skill, write both declared channels, apply the gate.

    python -m rte.backends.llm --measure --dist specialist --n 100 --K 16 --seed 1
    python -m rte.measure                --dist specialist --n 100 --K 16 --seed 1

Writes S.npy, profiles.json, D_programmatic.npy, D_self_described.npy, descriptions.json and
S_summary.json under $RTE_DATA/populations/<dist>_n<n>_K<K>_seed<seed>/. Exits 3 if the population
fails SPEC §1's `skill_excess_ratio_family >= 1.5` gate — there is no expertise to discover, so
running the grid on it would prove nothing.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .backends.llm import LLMBackend
from .world import skill_summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rte.measure")
    ap.add_argument("--dist", default="specialist")
    for flag, default in (("--n", 100), ("--K", 16), ("--seed", 1), ("--measure-probes", 200),
                          ("--measure-probes-large", 60), ("--concurrency", 64)):
        ap.add_argument(flag, type=int, default=default)
    ap.add_argument("--measure", action="store_true", help="accepted for symmetry; this IS measuring")
    ap.add_argument("--skip-self-described", action="store_true")
    a = ap.parse_args(argv)

    b = LLMBackend(n=a.n, K=a.K, dist=a.dist, seed=a.seed, measure_probes=a.measure_probes,
                   measure_probes_large=a.measure_probes_large, concurrency=a.concurrency)
    summary = skill_summary(b.true_skill(), a.measure_probes)
    np.save(b.dir / "D_programmatic.npy", b.declared("programmatic"))
    if not a.skip_self_described:
        b.declared("self_described")
    b.descriptions()
    (b.dir / "S_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n===== measured S =====")
    for k, v in summary.items():
        print(f"  {k:26s} {v:.4f}")
    gate = summary["skill_excess_ratio_family"]
    print(f"\ngate skill_excess_ratio_family >= 1.5 : {gate:.3f} -> {'PASS' if gate >= 1.5 else 'FAIL'}")
    print(f"population dir: {b.dir}\nbackend stats: {b.stats()}")
    return 0 if gate >= 1.5 else 3


if __name__ == "__main__":
    sys.exit(main())
