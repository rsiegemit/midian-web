"""Selfcheck for the decentralized, non-hierarchical rivals (SPEC §6).

    cd ~/rte && PYTHONPATH=. ~/miniconda3/bin/python scripts/selfcheck_decentral.py

Quality: bernoulli World(1000, 16, 'specialist', beta, seed=1, collude=True) for
beta in {0, 0.25, 0.5}; each method built with Budget(3) and run on the same 500
paired tasks. Reports success, oracle success on those tasks, misroute-to-liar
fraction and per-fetch ledger costs. `world.liars` is read HERE ONLY (runner-side
metric); no method ever sees it.

Scale: build + 100 fetches at n=1e5, K=16, b=1 for referral_network and
flat_nsw_router.
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "/n/home02/rsiegelmann/rte")

from rte.budget import Budget                     # noqa: E402
from rte.methods import load_method               # noqa: E402
from rte.world import World                       # noqa: E402

METHODS = ["referral_network", "gossip_reputation_greedy", "flat_nsw_router"]
BETAS = [0.0, 0.25, 0.5]
N, K, Q, B = 1000, 16, 500, 3


def run_one(world, name, budget, tasks):
    cls = load_method(name)
    m = cls()
    view = world.view(m.needs)
    world.ledger.reset()
    t0 = time.time()
    m.build(view, budget)
    build = world.ledger.snapshot()
    t_build = time.time() - t0
    cap = budget.total_probes(world.n, world.K)
    assert build["probes"] <= cap, f"{name} spent {build['probes']} probes > {cap}"
    world.ledger.reset()
    t0 = time.time()
    picks, outs = [], []
    for t in tasks:
        a = m.fetch(t)
        o = world.execute(a, t)
        m.observe(t, a, o)
        picks.append(a); outs.append(o)
    fetch = world.ledger.snapshot()
    return {"name": name, "build": build, "t_build": t_build, "cap": cap,
            "success": float(np.mean(outs)),
            "misroute": float(world.liars[np.asarray(picks)].mean()),
            "hops": fetch["hops"] / len(tasks), "cmp": fetch["comparisons"] / len(tasks),
            "msg": fetch["messages"] / len(tasks), "t_fetch": (time.time() - t0) / len(tasks)}


def quality():
    print(f"=== quality: bernoulli n={N} K={K} specialist collude=True seed=1 b={B} Q={Q} ===")
    rows = {}
    for beta in BETAS:
        w = World(N, K, "specialist", beta, seed=1, collude=True)
        tasks = w.tasks(Q)
        w.ledger.reset()
        orc = float(np.mean([w.execute(w.oracle(t), t) for t in tasks]))
        rnd = np.random.default_rng(0)
        rand = float(np.mean([w.execute(int(rnd.integers(0, w.n)), t) for t in tasks]))
        print(f"\n-- beta={beta}  liars={int(w.liars.sum())}  oracle={orc:.3f}  random={rand:.3f}")
        print(f"{'method':>26} {'succ':>6} {'oracle':>7} {'misrt':>6} {'probes':>8} {'reports':>8} "
              f"{'hops/f':>7} {'cmp/f':>7} {'msg/f':>7} {'build_s':>8} {'ms/fetch':>9}")
        for name in METHODS:
            r = run_one(w, name, Budget(B), tasks)
            rows[(name, beta)] = (r["success"], orc, r["misroute"])
            print(f"{name:>26} {r['success']:6.3f} {orc:7.3f} {r['misroute']:6.3f} "
                  f"{r['build']['probes']:8d} {r['build']['reports']:8d} "
                  f"{r['hops']:7.1f} {r['cmp']:7.1f} {r['msg']:7.1f} "
                  f"{r['t_build']:8.2f} {1000 * r['t_fetch']:9.3f}")
    print("\n=== success table (method x beta), oracle in the last row ===")
    print(f"{'method':>26} " + " ".join(f"{b:>8}" for b in BETAS))
    for name in METHODS:
        print(f"{name:>26} " + " ".join(f"{rows[(name, b)][0]:8.3f}" for b in BETAS))
    print(f"{'oracle':>26} " + " ".join(f"{rows[(METHODS[0], b)][1]:8.3f}" for b in BETAS))
    print(f"{'misroute_to_liar':>26}")
    for name in METHODS:
        print(f"{name:>26} " + " ".join(f"{rows[(name, b)][2]:8.3f}" for b in BETAS))


def scale(n=100_000, K_=16, b=1, q=100):
    print(f"\n=== scale: bernoulli n={n} K={K_} b={b} beta=0.25 specialist ===")
    w = World(n, K_, "specialist", 0.25, seed=1, collude=True)
    tasks = w.tasks(q)
    for name in ["referral_network", "flat_nsw_router"]:
        m = load_method(name)()
        view = w.view(m.needs)
        w.ledger.reset()
        t0 = time.time(); m.build(view, Budget(b)); tb = time.time() - t0
        assert w.ledger.probes <= Budget(b).total_probes(w.n, w.K)
        w.ledger.reset()
        t0 = time.time()
        outs = [w.execute(m.fetch(t), t) for t in tasks]
        tf = (time.time() - t0) / q
        print(f"{name:>26}  build {tb:8.2f}s  probes {Budget(b).total_probes(n, K_):>10d}  "
              f"fetch {1000 * tf:7.3f} ms  success {np.mean(outs):.3f}")


if __name__ == "__main__":
    quality()
    scale()
