"""MIDIAN tree invariants (SPEC §9) plus the CONTRACT correctness checks: partition, depth,
the max-tree property under exact estimates, exact ledger accounting, trimming under collusion,
success >= random, and cost at n=1e5."""
import time

import numpy as np
import pytest

from rte.budget import Budget
from rte.methods import load_method
from rte.methods.midian import Midian
from rte.methods.random import RandomMethod
from rte.world import Task, World

NS, RS = [100, 1000, 1234], [5, 10]


def depth_of(n, r):
    """ceil(log_r n), computed exactly (float logs round the wrong way at n = r^L)."""
    d = 0
    while n > 1:
        n, d = -(-n // r), d + 1
    return max(d, 1)


def built(n=100, K=16, r=10, b=3, beta=0.0, delta=1 / 3, seed=1, exact=None, **wkw):
    """A built Midian, its world, and the ledger delta of build. `exact` mocks the estimate stage."""
    w = World(n, K, "specialist", beta, seed=seed, **wkw)
    m = Midian(r=r, delta=delta)
    if exact is not None:
        exact.setattr("rte.methods.midian.peer_reported_estimates", lambda *a, **k: w.S.copy())
    before = w.ledger.snapshot()
    m.build(w.view(m.needs), Budget(b))
    return w, m, w.ledger.diff(before)


def children_values(m, l):
    """The (nodes, r, K) block of child values one level of the tree reduced."""
    ch = m.children[l]
    return np.where(ch[:, :, None] >= 0, (m.est if l == 0 else m.summary[l - 1])[ch], -np.inf)


# ------------------------------------------------------------------ (a) partition, (b) depth
@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("r", RS)
def test_every_agent_in_exactly_one_leaf_cohort(n, r):
    _, m, _ = built(n=n, r=r, b=1)
    ok = m.leaves >= 0
    assert np.array_equal(np.sort(m.leaves[ok]), np.arange(n))          # each agent exactly once
    rows = np.broadcast_to(np.arange(len(m.leaves))[:, None], m.leaves.shape)
    assert np.array_equal(m.leaf_of[m.leaves[ok]], rows[ok])


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("r", RS)
def test_depth_is_ceil_log_r_n(n, r):
    _, m, _ = built(n=n, r=r, b=1)
    assert m.depth == depth_of(n, r) == len(m.summary) == len(m.children)
    assert len(m.summary[-1]) == 1                                       # one root


# ------------------------------------------------------------------ (c) max-tree under exact estimates
@pytest.mark.parametrize("n,r", [(100, 10), (1000, 5), (1234, 10), (1234, 5)])
def test_exact_estimates_make_a_max_tree(n, r, monkeypatch):
    w, m, _ = built(n=n, r=r, b=1, exact=monkeypatch)
    assert np.array_equal(m.est, w.S)
    for l in range(m.depth):
        v = children_values(m, l)
        assert np.array_equal(m.summary[l], v.max(1))                    # summary == max over children
        assert np.array_equal(np.take_along_axis(v, m.best[l][:, None, :], 1)[:, 0], m.summary[l])
    assert np.array_equal([m.fetch(Task(0, f, 0)) for f in range(w.K)], w.S.argmax(0))


def test_observe_keeps_the_tree_consistent(monkeypatch):
    w, m, _ = built(n=1000, r=10, b=1, exact=monkeypatch)
    best = int(w.S[:, 0].argmax())
    for _ in range(300):
        m.observe(Task(0, 0, 0), best, 0)                                # drive the leader's estimate to 0
    for l in range(m.depth):
        assert np.allclose(m.summary[l][:, 0], children_values(m, l)[:, :, 0].max(1))
    assert m.fetch(Task(0, 0, 0)) != best                                # the descent now avoids it


# ------------------------------------------------------------------ (d) ledger accounting
@pytest.mark.parametrize("n,r,K,b", [(100, 10, 16, 3), (1000, 10, 16, 1), (1234, 10, 16, 3), (1234, 5, 16, 2)])
def test_build_and_fetch_ledger_is_exact(n, r, K, b):
    w, m, d = built(n=n, K=K, r=r, b=b)
    mm = n % r                                                           # at most one short cohort
    peers = (n - mm) * (r - 1) + mm * (mm - 1) if mm else n * (r - 1)
    assert d["probes"] == n * K * b == Budget(b).total_probes(n, K)
    assert d["reports"] == peers * K * b
    if mm == 0:
        assert d["reports"] == n * K * b * (r - 1)                       # SPEC §5 (iii), exactly
    assert d["messages"] == n - len(m.leaves) + sum(len(s) for s in m.summary) - 1
    assert (d["hops"], d["comparisons"], d["tasks"]) == (0, 0, 0)

    before = w.ledger.snapshot()
    for q in range(50):
        assert 0 <= m.fetch(Task(q, q % K, q)) < n
    d2 = w.ledger.diff(before)
    assert (d2["hops"], d2["comparisons"], d2["messages"]) == (50 * m.depth, 50 * m.depth * r, 50 * 2 * m.depth)
    assert (d2["probes"], d2["reports"], d2["tasks"]) == (0, 0, 0)


# ------------------------------------------------------------------ correctness: success >= random
@pytest.mark.parametrize("n", [100, 1000])
def test_success_beats_random_at_beta_zero(n, capsys):
    w, m, _ = built(n=n, K=16, r=10, b=3)
    rnd = RandomMethod()
    rnd.build(w.view(rnd.needs), Budget(3))
    stream = w.tasks(1000)
    got = {}
    for name, meth in (("midian", m), ("random", rnd)):
        got[name] = float(np.mean([w.execute(meth.fetch(t), t) for t in stream]))
    got["oracle"] = float(np.mean([w.execute(w.oracle(t), t) for t in stream]))
    print(f"\n[success] n={n} K=16 b=3 beta=0 specialist: " +
          "  ".join(f"{k}={v:.3f}" for k, v in got.items()))
    assert got["midian"] > got["random"]


# ------------------------------------------------------------------ (e) trimming under collusion
def test_trimming_vs_no_trimming_on_liar_cohorts(capsys):
    """Measured, not forced. delta=1/3 trims floor(delta*(r-1))=3 of 27 reports each side; SPEC §8.3
    predicts it stops paying once expected liars per cohort exceeds 3 (beta > 0.3)."""
    n, K, r, b, seed = 2000, 16, 10, 3, 7
    out = {}
    for beta in (0.25, 0.5):
        for delta in (0.0, 1 / 3):
            w, m, _ = built(n=n, K=K, r=r, b=b, beta=beta, delta=delta, seed=seed, collude=True)
            cohort = np.isin(m.leaf_of, np.unique(m.leaf_of[w.liars]))
            err = np.abs(m.est - w.S)
            out[beta, delta] = (float(err[cohort].mean()), float(err[cohort & ~w.liars].mean()),
                                float(err[w.liars].mean()))
    print(f"\n[trimming] collude=True n={n} K={K} r={r} b={b} seed={seed}: "
          "mean |est-S| over cohorts holding >=1 liar\n" +
          "\n".join(f"  beta={be}  delta={de:.3f}   all={v[0]:.4f}  honest={v[1]:.4f}  liars={v[2]:.4f}"
                    for (be, de), v in out.items()))
    for beta in (0.25, 0.5):                    # holds at every beta: trimming cleans up the honest
        assert out[beta, 1 / 3][1] <= out[beta, 0.0][1] + 0.002
    assert out[0.25, 1 / 3][0] <= out[0.25, 0.0][0] + 0.002    # below the crossover, the cohort too


# ------------------------------------------------------------------ (f) cost at n = 1e5
def test_timing_at_1e5(capsys):
    n, K, b = 100_000, 16, 1
    w = World(n, K, "specialist", 0.0, seed=1)
    m = Midian(r=10)
    t0 = time.perf_counter()
    m.build(w.view(m.needs), Budget(b))
    t_build = time.perf_counter() - t0
    stream = [Task(q, q % K, q) for q in range(2000)]
    t0 = time.perf_counter()
    for t in stream:
        m.fetch(t)
    t_fetch = (time.perf_counter() - t0) / len(stream)
    t0 = time.perf_counter()
    for t in stream:
        m.observe(t, 0, 1)
    t_obs = (time.perf_counter() - t0) / len(stream)
    print(f"\n[timing] n={n} K={K} b={b} r=10 depth={m.depth}: build {t_build:.2f} s, "
          f"fetch {t_fetch*1e3:.3f} ms/task, observe {t_obs*1e3:.3f} ms/task, est {m.est.nbytes/2**20:.0f} MiB")
    assert t_build < 60.0 and t_fetch < 1e-3


# ------------------------------------------------------------------ the LLM-descent ablation
def test_llm_descent_falls_back_to_the_arithmetic_argmax(monkeypatch):
    """Same tree, same estimates, same ledger; on an unparseable answer it routes as plain MIDIAN."""
    from rte.methods.midian_llm_descent import MidianLLMDescent
    w1, m1, _ = built(n=500, r=10, b=1, seed=3)
    w2 = World(500, 16, "specialist", 0.0, seed=3)
    m2 = MidianLLMDescent(r=10)
    m2.build(w2.view(m2.needs), Budget(1))
    monkeypatch.setattr("rte.llm_client.complete", lambda *a, **k: "no idea")
    assert np.array_equal(m1.est, m2.est) and np.array_equal(m1.leaves, m2.leaves)
    assert [m1.fetch(Task(0, f, 0)) for f in range(16)] == [m2.fetch(Task(0, f, 0)) for f in range(16)]
    assert w1.ledger.snapshot() == w2.ledger.snapshot()          # identical cost, quality-only ablation
    assert m2.stats["fallbacks"] == m2.stats["calls"] == 16 * m2.depth


def test_llm_descent_follows_a_parseable_answer(monkeypatch):
    from rte.methods.midian_llm_descent import MidianLLMDescent
    w = World(500, 16, "specialist", 0.0, seed=3)
    m = MidianLLMDescent(r=10)
    m.build(w.view(m.needs), Budget(1))
    monkeypatch.setattr("rte.llm_client.complete", lambda *a, **k: "1")   # always answer "child 1"
    for f in range(16):
        node, a = 0, m.fetch(Task(0, f, 0))
        for l in range(m.depth - 1, -1, -1):
            v = m._values(l, node, f)
            ok = np.flatnonzero(np.isfinite(v))
            node = int(m.children[l][node, 1 if 1 in ok else ok[0]])
        assert a == node
    assert m.stats["fallbacks"] == 0


# ---------------------------------------------------------------- v2 (2026-09-03): per-probe reports, stratify, churn, midian_v
def _build(name, n, seed=1, beta=0.0, **kw):
    w = World(n, 16, "specialist", beta, seed=seed); M = load_method(name)(**kw); v = w.view(M.needs)
    s0 = w.ledger.snapshot(); M.build(v, Budget(3)); return w, M, v, w.ledger.diff(s0)


@pytest.mark.parametrize("n", [100, 1000])
def test_midian_v_reports_one_per_reporter_per_probe(n):
    """0.3: every arm charges one report per (peer, member, family, probe): reports == probes * (r-1) for V too."""
    _, _, _, d = _build("midian_v", n)
    assert d["reports"] == d["probes"] * 9 and d["probes"] <= n * 16 * 3


def test_midian_v_equals_midian_verify_cached():
    wa, A, _, da = _build("midian_v", 300, seed=4, beta=0.25); wb, B_, _, db = _build("midian", 300, seed=4, beta=0.25, verify=True, cached=True)
    assert da == db and [A.fetch(t) for t in wa.tasks(50)] == [B_.fetch(t) for t in wb.tasks(50)]


def test_stratified_cohorts_take_one_member_per_stratum(monkeypatch):
    """1.5: with exact probes the key is S.mean(1); every full cohort holds exactly one agent from each of the r deciles."""
    w = World(100, 16, "specialist", 0.0, seed=3); M = load_method("midian")(stratify=True); v = w.view(M.needs)
    monkeypatch.setattr("rte.methods.midian.probe_outcomes", lambda view, b: np.broadcast_to(w.S[:, :, None], (100, 16, b)).astype(np.float32))
    s0 = w.ledger.snapshot(); M.build(v, Budget(3)); d = w.ledger.diff(s0)
    assert d["probes"] == 0 and d["reports"] == 100 * 16 * 3 * 9            # (mocked probes; reports still charged)
    rest = np.setdiff1d(np.arange(100), M.leaves[-1])                     # the last cohort is the random (short) one
    stratum = np.empty(100, int); stratum[rest[np.argsort(w.S[rest].mean(1), kind="stable")]] = np.arange(90) // 9
    for cohort in M.leaves[:-1]:
        assert sorted(stratum[cohort]) == list(range(10))


def test_churn_repairs_only_the_arrived_agents():
    w, M, v, _ = _build("midian", 200, seed=2)
    arrived = np.array([3, 50, 77]); before = M.est.copy(); s0 = w.ledger.snapshot()
    M.churn(arrived, arrived); d = w.ledger.diff(s0)
    assert d["probes"] == 3 * 16 * 3 and d["reports"] == 3 * 16 * 3 * 9 and d["messages"] == 3 * (9 + M.depth)
    untouched = np.setdiff1d(np.arange(200), arrived)
    assert np.array_equal(M.est[untouched], before[untouched])
    for l in range(M.depth):                                               # tree still consistent after the repair
        for node in range(len(M.children[l])):
            for f in (0, 7):
                vals = M._values(l, node, f); assert M.summary[l][node, f] == vals.max() and M.best[l][node, f] == vals.argmax()
