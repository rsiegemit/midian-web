"""Quick self-check for the declared-channel + CNP/disrouter/cluster/route-to-k
methods. Not a pytest file -- run directly:

    cd ~/rte && PYTHONPATH=. ~/miniconda3/bin/python scripts/selfcheck_declared.py

For each method: build with Budget(), run 200 tasks from World(100, 16,
'specialist', 0.25, seed=1).tasks(200), execute via world.execute (majority
for list-returning fetches), print success + ledger diff, and assert the
View never raised AccessError. Then time build + 100 fetches for
declared_argmax and cluster_head_router at n=1e6 (iid_uniform, beta=0).
"""
from __future__ import annotations

import time

import numpy as np

from rte.world import World, AccessError
from rte.budget import Budget
from rte.methods import load_method

METHODS = [
    "random",
    "declared_argmax",
    "declared_softmax",
    "cnp_self_bid",
    "disrouter_cascade",
    "cluster_head_router",
    "route_to_k_majority",
]


def majority(outcomes) -> int:
    s = sum(outcomes)
    return 1 if s > len(outcomes) / 2 else 0


def run_check(world: World, name: str, **params) -> tuple[float, dict]:
    cls = load_method(name)
    m = cls(**params)
    view = world.view(needs=m.needs)
    try:
        m.build(view, Budget())
        view.ledger.reset()
        before = view.ledger.snapshot()
        tasks = world.tasks(200)
        n_success = 0
        for t in tasks:
            a = m.fetch(t)
            if isinstance(a, list):
                assert len(a) >= 1
                outs = [world.execute(ai, t) for ai in a]
                o = majority(outs)
            else:
                o = world.execute(int(a), t)
            m.observe(t, a, o)
            n_success += o
    except AccessError as e:
        raise AssertionError(f"{name}: View raised AccessError unexpectedly: {e}") from e
    diff = view.ledger.diff(before)
    success = n_success / len(tasks)
    print(f"{name:24s} success={success:.3f}  ledger_diff={diff}")
    return success, diff


def scale_check(name: str, world: World, n_fetch: int = 100) -> None:
    cls = load_method(name)
    m = cls()
    view = world.view(needs=m.needs)
    t0 = time.time()
    m.build(view, Budget())
    t_build = time.time() - t0
    tasks = world.tasks(n_fetch)
    t0 = time.time()
    for t in tasks:
        m.fetch(t)
    t_fetch = time.time() - t0
    print(f"{name:24s} n={world.n:.0e}  build={t_build:.2f}s  "
          f"{n_fetch}-fetch={t_fetch:.2f}s  ({1000 * t_fetch / n_fetch:.3f}ms/task)")


def main() -> None:
    world = World(100, 16, "specialist", 0.25, seed=1)
    print(f"--- 100 tasks, n=100, specialist, beta=0.25 (world stats: {world.stats()}) ---")
    for name in METHODS:
        run_check(world, name)

    print("\n--- scale check @ n=1e6, iid_uniform, beta=0 ---")
    world_big = World(1_000_000, 16, "iid_uniform", 0.0, seed=1)
    for name in ["declared_argmax", "cluster_head_router"]:
        scale_check(name, world_big)


if __name__ == "__main__":
    main()
