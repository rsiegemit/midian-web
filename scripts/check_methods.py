"""Correctness + ledger-accounting checker (CONTRACT 'Correctness checks'). Run on any method names:
    PYTHONPATH=. python scripts/check_methods.py flat_probe_argmax ucb_per_family ...
Per method, at n in {100, 1000} on bernoulli specialist beta=0: valid ids; build probes <= budget; build/per-fetch ledger vs the
documented formula (EXPECT); success >= random; exact-estimate argmax check for methods with an `est`/`best` table."""
import math, sys
import numpy as np
from rte.world import World
from rte.budget import Budget
from rte.methods import load_method

K, B, Q = 16, 3, 400
# documented formulas: build -> dict, fetch -> dict (per task).  n = agents, r = 10 default
EXPECT = {
    "flat_probe_argmax":   (lambda n: dict(probes=n*K*B, messages=0), lambda n: dict(comparisons=n, messages=0)),
    "ucb_per_family":      (lambda n: dict(probes=n*K*B, messages=0), lambda n: dict(comparisons=n, messages=0)),
    "thompson_per_family": (lambda n: dict(probes=n*K*B, messages=0), lambda n: dict(comparisons=n, messages=0)),
    "warm_start_bandit":   (lambda n: dict(probes=n*K*B, messages=n), lambda n: dict(comparisons=n, messages=0)),
    "sequential_halving":  (lambda n: dict(messages=0),              lambda n: dict(comparisons=1, messages=0)),
    "verify_on_claim":     (lambda n: dict(probes=0, messages=n),    lambda n: dict(messages=0)),
    "trueskill_per_family":(lambda n: dict(messages=0),              lambda n: dict(comparisons=n, messages=0)),
    "midian":              (lambda n: dict(probes=n*K*B, reports=n*K*B*9), lambda n: dict(hops=math.ceil(math.log10(n)), comparisons=10*math.ceil(math.log10(n)), messages=2*math.ceil(math.log10(n)))),
    "cnp_self_bid":        (lambda n: dict(messages=n),              lambda n: dict(messages=2*n, comparisons=n)),
    "declared_argmax":     (lambda n: dict(messages=n),              lambda n: dict(comparisons=n, messages=0)),
    "random":              (lambda n: dict(),                        lambda n: dict(probes=0, messages=0)),
    "referral_network":    (lambda n: dict(probes=n*K*B, reports=n*K*B, messages=n*10),
                            lambda n: dict(hops=4, comparisons=40, messages=80)),
    "gossip_reputation_greedy": (lambda n: dict(probes=n*K*B, reports=n*K*B),
                            lambda n: dict(hops=6, comparisons=60, messages=120)),
    "flat_nsw_router":     (lambda n: dict(probes=n*K*B, messages=0),
                            lambda n: dict(hops=math.ceil(math.log2(n)), comparisons=50, messages=0)),
}


def run(name, n, **kw):
    w = World(n, K, "specialist", 0.0, seed=1); M = load_method(name)(**kw); v = w.view(M.needs)
    s0 = w.ledger.snapshot(); M.build(v, Budget(B)); build = w.ledger.diff(s0)
    ts = w.tasks(Q); s1 = w.ledger.snapshot(); ok = []
    for t in ts:
        a = M.fetch(t); a = a if isinstance(a, (int, np.integer)) else a[0]
        assert 0 <= a < n, (name, a)
        o = w.execute(a, t); M.observe(t, a, o); ok.append(o)
    fetch = {k: val / Q for k, val in w.ledger.diff(s1).items()}
    rnd = np.mean([w.execute(int(v.rng.integers(n)), t) for t in ts])
    return M, w, build, fetch, float(np.mean(ok)), float(rnd)


def exact_argmax(name, n):
    """Make probes return S exactly; an argmax-type method must then return argmax S for every family."""
    w = World(n, K, "specialist", 0.0, seed=1); M = load_method(name)(); v = w.view(M.needs)
    v.probe_many = lambda agents, fams, reps: np.broadcast_to(w.S[np.asarray(agents), np.asarray(fams)][..., None], np.broadcast_arrays(np.asarray(agents), np.asarray(fams))[0].shape + (reps,)).astype(np.float64)
    v.report_many = lambda R, A, O: np.broadcast_arrays(R, A, O)[2].astype(np.float64)   # honest pass-through, float
    M.build(v, Budget(B)); truth = w.oracle_all()
    hits = sum(M.fetch(t) == truth[t.family] for t in w.tasks(200))
    return hits / 200


if __name__ == "__main__":
    for name in sys.argv[1:]:
        for n in (100, 1000):
            M, w, build, fetch, succ, rnd = run(name, n)
            eb, ef = EXPECT.get(name, (lambda n: {}, lambda n: {}))
            bad = [f"build.{k}={build[k]}!={val}" for k, val in eb(n).items() if build[k] != val]
            bad += [f"fetch.{k}={fetch[k]:.3g}!={val}" for k, val in ef(n).items() if abs(fetch[k] - val) > 1e-9]
            bad += [f"probes {build['probes']} > budget {n*K*B}"] if build["probes"] > n*K*B else []
            bad += [f"success {succ:.3f} < random {rnd:.3f}"] if succ < rnd else []
            print(f"{name:22s} n={n:5d} succ={succ:.3f} rnd={rnd:.3f} build={ {k:v for k,v in build.items() if v} } "
                  f"fetch={ {k:round(v,3) for k,v in fetch.items() if v} }  {'OK' if not bad else 'FAIL: ' + '; '.join(bad)}")
        if hasattr(M, "est") or hasattr(M, "best"):
            print(f"{name:22s} exact-estimate argmax hit rate: {exact_argmax(name, 100):.2f} (expect 1.00)")
