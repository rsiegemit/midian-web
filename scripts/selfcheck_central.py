"""Self-check for the seven centralized/trusted-observer rivals.

For each of {0.0, 0.5} liar fractions on a bernoulli World(100, 16,
'specialist', beta, seed=1): build each method with Budget(3), verify the
build-phase probe count is within budget, run 500 tasks from world.tasks(500)
through world.execute + method.observe, and report success, success over the
last 250 tasks, oracle success on the same stream, and the per-fetch ledger
delta (ledger cost of running the 500-task stream, build cost excluded).

Run: cd ~/rte && PYTHONPATH=. ~/miniconda3/bin/python scripts/selfcheck_central.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rte.world import World
from rte.budget import Budget
from rte.methods import load_method

METHODS = [
    "flat_probe_argmax",
    "ucb_per_family",
    "thompson_per_family",
    "sequential_halving",
    "verify_on_claim",
    "warm_start_bandit",
    "trueskill_per_family",
]

N, K, DIST, SEED, Q = 100, 16, "specialist", 1, 500


def run_one(method_name: str, beta: float):
    world = World(N, K, DIST, beta, seed=SEED)
    cls = load_method(method_name)
    method = cls()
    view = world.view(needs=cls.needs)

    before = world.ledger.snapshot()
    budget = Budget(3)
    method.build(view, budget)
    build_diff = world.ledger.diff(before)
    max_probes = budget.total_probes(N, K)
    assert build_diff["probes"] <= max_probes, (
        f"{method_name}: build spent {build_diff['probes']} probes > budget {max_probes}")

    world.ledger.reset()
    tasks = world.tasks(Q)
    outcomes = []
    fetch_before = world.ledger.snapshot()
    for task in tasks:
        a = method.fetch(task)
        o = world.execute(a, task)
        method.observe(task, a, o)
        outcomes.append(o)
    fetch_diff = world.ledger.diff(fetch_before)

    oracle_outcomes = [world.execute(world.oracle(t), t) for t in tasks]

    success = sum(outcomes) / len(outcomes)
    success_late = sum(outcomes[-250:]) / len(outcomes[-250:])
    oracle_success = sum(oracle_outcomes) / len(oracle_outcomes)

    return {
        "method": method_name, "beta": beta,
        "success": success, "success_late": success_late, "oracle_success": oracle_success,
        "build_probes": build_diff["probes"], "budget_probes": max_probes,
        "fetch_ledger": fetch_diff,
    }


def main():
    rows = []
    for beta in (0.0, 0.5):
        for m in METHODS:
            try:
                rows.append(run_one(m, beta))
            except NotImplementedError as e:
                print(f"SKIP {m} beta={beta}: {e}")

    print("\n=== success table (method x beta) ===")
    header = f"{'method':24s} {'beta':>5s} {'success':>9s} {'succ_late':>10s} {'oracle':>8s}"
    print(header)
    for r in rows:
        print(f"{r['method']:24s} {r['beta']:5.2f} {r['success']:9.3f} {r['success_late']:10.3f} {r['oracle_success']:8.3f}")

    print("\n=== build-probe counts ===")
    for r in rows:
        if r["beta"] == 0.0:
            print(f"{r['method']:24s} build_probes={r['build_probes']:>8d}  budget={r['budget_probes']:>8d}")

    print("\n=== per-fetch ledger (500-task stream, build excluded) ===")
    for r in rows:
        print(f"{r['method']:24s} beta={r['beta']:.2f}  {r['fetch_ledger']}")


if __name__ == "__main__":
    main()
