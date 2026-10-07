"""The 2026-10-07 review attacks (lie_mode options q / fam / d) and sequential halving with MIDIAN's audits."""
import numpy as np
import pytest

from midian.budget import Budget
from midian.methods.sequential_halving import SequentialHalving
from midian.world import World, apply_lying, parse_lie


def world(lie_mode="inflate", beta=0.5, n=60, **kw):
    return World(n, 8, "specialist", beta, seed=5, lie_mode=lie_mode, **kw)


def batch(w, size=4000):
    rng = np.random.default_rng(1)
    return rng.integers(0, w.n, size), rng.integers(0, w.n, size), rng.integers(0, 2, size)


def test_bare_mode_is_the_standard_attack():
    assert parse_lie("inflate") == ("inflate", {})
    a, b = world(), world("inflate:q=1")
    rep, ag, out = batch(a)
    assert np.array_equal(a.report_many(rep, ag, out), b.report_many(rep, ag, out))
    assert np.array_equal(a.D, b.D)


def test_partial_lie_tells_a_fraction_q_of_the_lies():
    full = world()
    rep, ag, out = batch(full)
    lies = full.report_many(rep, ag, out) != out
    assert np.array_equal(world("inflate:q=0").report_many(rep, ag, out), out.astype(np.int8))
    told = world("inflate:q=0.25").report_many(rep, ag, out) != out
    assert not (told & ~lies).any()                              # only lies the full attack tells
    assert 0.15 < told.sum() / lies.sum() < 0.35


def test_single_family_cartel_lies_about_that_family_only():
    w = world("inflate:fam=2")
    rep, ag, out = batch(w)
    fams = np.arange(rep.size) % 8
    changed = w.report_many(rep, ag, out, fams) != out
    assert changed.any() and (fams[changed] == 2).all()
    with pytest.raises(ValueError):
        w.report_many(rep, ag, out)
    D0 = World(60, 8, "specialist", 0.0, seed=5).D
    moved = w.D != D0
    assert moved.any() and (np.flatnonzero(moved.any(0)) == [2]).all()


def test_declared_delta_option():
    D = np.full((10, 4), 0.3, np.float32)
    liars = np.arange(10) < 5
    assert np.allclose(apply_lying(D, liars, "inflate:d=0.1")[liars], 0.4)
    assert np.allclose(apply_lying(D, liars, "inflate:d=0.1")[~liars], 0.3)
    with pytest.raises(ValueError):
        parse_lie("inflate:x=1")


def run_halving(w, **p):
    m = SequentialHalving(**p)
    w.reset()
    m.build(w.view(m.needs), Budget(3))
    return m


def test_halving_audits_stay_in_budget_and_catch_the_cartel():
    w = World(400, 4, "specialist", 0.5, liar_select="low_skill_first", seed=3)
    m = run_halving(w, peer_reported=True, audit=True)
    assert w.ledger.snapshot()["probes"] <= Budget(3).total_probes(w.n, w.K)
    assert m.stats["excluded"] > 0
    honest = World(400, 4, "specialist", 0.0, seed=3)
    assert run_halving(honest, peer_reported=True, audit=True).stats["excluded"] == 0


def test_audit_needs_peer_reports():
    with pytest.raises(ValueError):
        SequentialHalving(audit=True)
