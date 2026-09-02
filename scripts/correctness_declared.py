"""Correctness property checks (lead directive) for the 7 declared-channel methods, at n in {100, 1000},
bernoulli 'specialist', beta=0: valid agent ids, exact ledger accounting against the documented per-fetch
message/hop/compare formula, build messages == n (declared collection), success >= random's on the same
paired task stream, and declared_argmax returns the true argmax of D.

    cd ~/rte && PYTHONPATH=. ~/miniconda3/bin/python scripts/correctness_declared.py
"""
import numpy as np

from rte.world import World, AccessError
from rte.budget import Budget
from rte.methods import load_method

Q = 200


def check(world, name, tasks):
    cls = load_method(name)
    m = cls()
    view = world.view(needs=m.needs)
    before = view.ledger.snapshot()
    m.build(view, Budget())
    build_diff = view.ledger.diff(before)
    view.ledger.reset()

    outs, per_fetch = [], []
    for t in tasks:
        b = view.ledger.snapshot()
        a = m.fetch(t)
        d = view.ledger.diff(b)
        ids = a if isinstance(a, list) else [a]
        assert all(0 <= i < world.n for i in ids), f"{name}: invalid agent id {a}"
        results_i = [world.execute(i, t) for i in ids]
        o = (1 if sum(results_i) > len(results_i) / 2 else 0) if isinstance(a, list) else results_i[0]
        outs.append(o)
        per_fetch.append(d)
    return build_diff, per_fetch, float(np.mean(outs))


def assert_formula(name, n, build_diff, per_fetch):
    assert build_diff["messages"] == (0 if name == "random" else n), (name, "build messages", build_diff)
    for d in per_fetch:
        if name in ("declared_argmax", "declared_softmax", "route_to_k_majority"):
            assert d["messages"] == 0 and d["comparisons"] == n and d["hops"] == 0, (name, d)
        elif name == "cnp_self_bid":
            assert d["messages"] == 2 * n and d["comparisons"] == n and d["hops"] == 0, (name, d)
        elif name == "disrouter_cascade":
            assert d["messages"] == d["hops"] and 0 <= d["messages"] < n and d["comparisons"] == 0, (name, d)
        elif name == "cluster_head_router":
            assert d["messages"] == 4 and d["hops"] == 2, (name, d)
        elif name == "random":
            assert d == {"probes": 0, "reports": 0, "messages": 0, "hops": 0, "comparisons": 0, "tasks": 0}, (name, d)


def check_argmax_exact(world):
    cls = load_method("declared_argmax")
    m = cls()
    view = world.view(needs=m.needs)
    m.build(view, Budget())
    D = view.declared
    for f in range(min(world.K, 8)):
        a = m.fetch(type(world.tasks(1)[0])(0, f, 0))
        assert D[a, f] == D[:, f].max(), ("declared_argmax not exact argmax", f, a)


METHODS = ["random", "declared_argmax", "declared_softmax", "cnp_self_bid",
           "disrouter_cascade", "cluster_head_router", "route_to_k_majority"]


def main():
    for n in (100, 1000):
        world = World(n, 16, "specialist", 0.0, seed=1)
        tasks = world.tasks(Q)
        results = {}
        for name in METHODS:
            try:
                build_diff, per_fetch, success = check(world, name, tasks)
            except AccessError as e:
                raise AssertionError(f"{name}: View raised unexpectedly: {e}") from e
            assert_formula(name, n, build_diff, per_fetch)
            results[name] = success
            print(f"n={n:5d} {name:22s} success={success:.3f}  build={build_diff}")
        assert results["random"] <= min(v for k, v in results.items() if k != "random") + 1e-9, \
            f"n={n}: some method scored below random: {results}"
        print(f"n={n}: PASS -- all methods >= random ({results['random']:.3f})")
    check_argmax_exact(World(1000, 16, "specialist", 0.0, seed=1))
    print("declared_argmax exact-argmax check: PASS")


if __name__ == "__main__":
    main()
