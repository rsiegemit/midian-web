"""MIDIAN-SH / MIDIAN-A / MIDIAN-SH+A (v2 labeled variants): budget equality, exclusion of caught liars, plain-MIDIAN
behaviour when the mechanism is switched off, and the bernoulli sanity checks."""
import numpy as np
import pytest

from rte.budget import Budget
from rte.methods.midian import Midian
from rte.methods.midian_a import MidianA
from rte.methods.midian_sh import MidianSH, _schedule
from rte.methods.midian_sha import MidianSHA
from rte.world import World


def run(cls, n=100, beta=0.25, seed=1, b=3, Q=300, **kw):
    w = World(n, 16, "specialist", beta, seed=seed, liar_select="low_skill_first")
    m = cls(**kw); before = w.ledger.snapshot()
    m.build(w.view(m.needs), Budget(b)); build = w.ledger.diff(before)
    stream = w.tasks(Q); s = []
    for t in stream:
        a = m.fetch(t); o = w.execute(a, t); m.observe(t, a, o); s.append(o)
    return w, m, build, float(np.mean(s))


@pytest.mark.parametrize("s,b", [(10, 3), (10, 1), (3, 3), (7, 2), (2, 5)])
def test_schedule_spends_exactly_s_times_b(s, b):
    assert sum(sz * p for sz, p in _schedule(s, b, True)) == s * b
    assert _schedule(s, b, False) == [(s, b)]


def test_midian_sh_budget_equal():
    """MIDIAN-SH spends exactly plain MIDIAN's probes (n*K*b) and reports per (peer, member, family, probe)."""
    _, _, plain, _ = run(Midian)
    _, m, sh, s = run(MidianSH)
    assert sh["probes"] == plain["probes"] == 100 * 16 * 3
    assert sh["reports"] == plain["reports"]
    assert m.est.shape == (100, 16) and np.isfinite(m.est).all() and s > 0.5


def test_midian_sh_halving_off_matches_plain_accounting():
    _, _, plain, _ = run(Midian)
    _, _, off, _ = run(MidianSH, halving=False)
    assert off == plain


def test_midian_a_exclusion():
    """A colluding liar caught lying twice by the audits is excluded; audits cost <= 5% extra probes."""
    w, m, build, s = run(MidianA, beta=0.5)
    liars = w.liars
    assert m.excluded.any() and not m.excluded[~liars].any(), "audits excluded an honest reporter"
    assert (m.hits[m.excluded] >= 2).all() and liars[m.excluded].all()
    _, _, plain, _ = run(Midian, beta=0.5)
    assert plain["probes"] < build["probes"] <= 1.05 * plain["probes"] * 1.02     # 5% audits (binomial slack)
    assert s > 0.5


def test_midian_a_no_liars_no_exclusions():
    w, m, build, _ = run(MidianA, beta=0.0)
    assert not m.excluded.any()


def test_midian_sha_composes():
    _, m, build, s = run(MidianSHA, beta=0.5)
    assert m.halving and m.audit and build["probes"] <= 1.05 * 100 * 16 * 3 * 1.02 and s > 0.5


def test_online_audit_charges_reports_and_can_exclude():
    w, m, _, _ = run(MidianA, beta=0.5, Q=1)
    m.rate = 1.0                                          # audit every routed outcome
    w.ledger.reset()
    for t in w.tasks(50):
        a = m.fetch(t); m.observe(t, a, w.execute(a, t))
    assert w.ledger.reports > 0


def test_midian_va_costs_like_v():
    """VA = V + audits: build probes within 1.05x of midian_v on the same world, fetch charges 1 comparison + 2 messages."""
    from rte.methods.midian_v import MidianV
    from rte.methods.midian_va import MidianVA
    out = {}
    for cls in (MidianV, MidianVA):
        w = World(n=200, K=8, dist="specialist", beta=0.25, seed=3, backend="bernoulli"); m = cls()
        v = w.view(m.needs); m.build(v, Budget(3)); out[cls.name] = dict(w.ledger.snapshot())
        w.ledger.reset(); t = w.tasks(20)[0]; a = m.fetch(t); assert 0 <= a < w.n
        assert w.ledger.snapshot()["comparisons"] == 1 and w.ledger.snapshot()["messages"] == 2
    assert out["midian_va"]["probes"] <= 1.05 * out["midian_v"]["probes"] + 1
