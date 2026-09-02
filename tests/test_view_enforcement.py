"""View access control (a method may touch only what it declares) and the lying model.

Two halves:
  1. `View` raises AccessError on anything outside `needs` -- including a *mutation* test where a
     method that quietly reads `view.declared` without declaring it blows up.
  2. `select_liars` / `apply_lying` / the report channel behave as SPEC section 4 says, and the
     scalar (`report`) and vectorized (`report_many`) paths agree on the same inputs.
"""
from __future__ import annotations

import numpy as np
import pytest

from rte.budget import Budget
from rte.methods.base import Method
from rte.world import (AccessError, DELTA_INFLATE, World, apply_lying, select_liars)


def _world(**kw):
    kw.setdefault("beta", 0.25)
    return World(60, 8, "specialist", seed=5, **kw)


# =========================================================== 1. view enforcement
PROBE_ONLY = {"probe"}


def test_probe_only_view_raises_on_declared():
    v = _world().view(PROBE_ONLY)
    with pytest.raises(AccessError):
        _ = v.declared


def test_probe_only_view_raises_on_report_channel():
    v = _world().view(PROBE_ONLY)
    with pytest.raises(AccessError):
        v.report_channel(0, 1, 1)


def test_probe_only_view_raises_on_report_many():
    v = _world().view(PROBE_ONLY)
    with pytest.raises(AccessError):
        v.report_many([0], [1], [1])


def test_probe_only_view_raises_on_bus():
    v = _world().view(PROBE_ONLY)
    with pytest.raises(AccessError):
        _ = v.bus


@pytest.mark.parametrize("attr", ["S", "liars", "D", "backend", "oracle", "beta", "seed", "world"])
def test_view_raises_on_any_unknown_attribute(attr):
    """S and the liar set are never exposed, and neither is anything else undeclared."""
    v = _world().view(PROBE_ONLY)
    with pytest.raises(AccessError):
        getattr(v, attr)


def test_empty_needs_raises_on_probe():
    v = _world().view(set())
    with pytest.raises(AccessError):
        v.probe(0, 0)
    with pytest.raises(AccessError):
        v.probe_many([0], [0], 1)


def test_declared_view_allows_declared_and_nothing_else():
    w = _world()
    v = w.view({"declared"})
    D = v.declared
    assert D.shape == (w.n, w.K)
    with pytest.raises(AccessError):
        v.probe(0, 0)
    with pytest.raises(AccessError):
        _ = v.bus


def test_always_available_fields():
    w = _world()
    v = w.view(set())
    assert v.n == w.n and v.K == w.K
    assert len(v.families) == w.K
    assert v.ledger is w.ledger
    assert v.needs == frozenset()
    assert isinstance(v.rng, np.random.Generator)


def test_declared_array_is_read_only():
    v = _world().view({"declared"})
    D = v.declared
    assert D.flags.writeable is False
    with pytest.raises(ValueError):
        D[0, 0] = 0.5


def test_declared_attribute_cannot_be_rebound():
    """`.declared` is a read-only property: assigning to it raises."""
    v = _world().view({"declared"})
    with pytest.raises(AttributeError):
        v.declared = np.zeros((v.n, v.K))


def test_declared_view_does_not_leak_true_skill():
    """The declared channel is a *copy*; poking at it cannot reach S."""
    w = _world()
    v = w.view({"declared"})
    assert v.declared is not w.S
    assert np.shares_memory(v.declared, w.S) is False


def test_unknown_need_name_raises_value_error():
    w = _world()
    with pytest.raises(ValueError):
        w.view({"telepathy"})
    with pytest.raises(ValueError):
        w.view({"probe", "S"})


def test_view_rng_differs_across_seeds():
    w = _world()
    a = w.view(PROBE_ONLY, seed=1).rng.random(16)
    b = w.view(PROBE_ONLY, seed=2).rng.random(16)
    assert not np.array_equal(a, b)
    # ... and is reproducible at a fixed seed
    assert np.array_equal(a, w.view(PROBE_ONLY, seed=1).rng.random(16))


def test_view_rng_differs_across_needs():
    w = _world()
    a = w.view({"probe"}, seed=1).rng.random(8)
    b = w.view({"probe", "reports"}, seed=1).rng.random(8)
    assert not np.array_equal(a, b)


# ---------------------------------------------------------- mutation test
class _SneakyMethod(Method):
    """Declares only `probe` but reads the declared channel in build. Must fail."""
    name = "_sneaky"
    needs = frozenset({"probe"})

    def build(self, view, budget: Budget) -> None:
        self.D = np.asarray(view.declared)          # undeclared access

    def fetch(self, task) -> int:
        return 0


class _HonestMethod(Method):
    name = "_honest"
    needs = frozenset({"probe", "declared"})

    def build(self, view, budget: Budget) -> None:
        self.D = np.asarray(view.declared)
        self.est = view.probe_many(np.arange(view.n), np.zeros(view.n, int), 1)

    def fetch(self, task) -> int:
        return int(np.argmax(self.D[:, task.family]))


def test_mutation_undeclared_declared_read_fails():
    w = _world()
    m = _SneakyMethod()
    with pytest.raises(AccessError):
        m.build(w.view(m.needs), Budget(1))


def test_mutation_same_method_with_the_need_declared_succeeds():
    """The control arm: the only thing that changed is the declaration."""
    w = _world()
    m = _SneakyMethod()
    m.needs = frozenset({"probe", "declared"})
    m.build(w.view(m.needs), Budget(1))
    assert m.D.shape == (w.n, w.K)


def test_honest_method_runs_and_dropping_a_need_breaks_it():
    w = _world()
    m = _HonestMethod()
    m.build(w.view(m.needs), Budget(1))
    task = w.tasks(1)[0]
    assert 0 <= m.fetch(task) < w.n
    for dropped in sorted(m.needs):
        with pytest.raises(AccessError):
            _HonestMethod().build(w.view(m.needs - {dropped}), Budget(1))


# =========================================================== 2. the lying model
@pytest.mark.parametrize("beta", [0.0, 0.1, 0.25, 0.5, 1.0])
def test_select_liars_count(beta):
    S = np.random.default_rng(0).random((200, 8))
    liars = select_liars(S, beta, "random", np.random.default_rng(1))
    assert liars.dtype == bool and liars.shape == (200,)
    assert int(liars.sum()) == int(round(beta * 200))


def test_select_liars_low_skill_first_picks_the_lowest_mean_agents():
    rng = np.random.default_rng(2)
    S = rng.random((100, 8))
    liars = select_liars(S, 0.25, "low_skill_first", rng)
    means = S.mean(axis=1)
    assert int(liars.sum()) == 25
    assert means[liars].max() <= means[~liars].min()
    assert set(np.flatnonzero(liars)) == set(np.argsort(means, kind="stable")[:25])


def test_select_liars_random_is_not_low_skill_first():
    rng = np.random.default_rng(3)
    S = rng.random((200, 8))
    a = select_liars(S, 0.25, "random", np.random.default_rng(7))
    b = select_liars(S, 0.25, "low_skill_first", np.random.default_rng(7))
    assert not np.array_equal(a, b)


def test_select_liars_rejects_unknown_mode():
    with pytest.raises(ValueError):
        select_liars(np.zeros((10, 2)), 0.5, "sneaky", np.random.default_rng(0))


def test_apply_lying_inflate():
    S = np.array([[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]], dtype=np.float32)
    liars = np.array([True, False, True])
    D = apply_lying(S, liars, "inflate")
    assert np.allclose(D[1], S[1])                                     # honest untouched
    assert np.allclose(D[0], np.clip(S[0] + DELTA_INFLATE, 0, 1))
    assert np.allclose(D[2], np.clip(S[2] + DELTA_INFLATE, 0, 1))
    assert D.max() <= 1.0 and D.min() >= 0.0
    assert not np.shares_memory(D, S)                                   # input is not mutated


def test_apply_lying_inflate_is_a_noop_with_no_liars():
    S = np.random.default_rng(0).random((10, 4))
    D = apply_lying(S, np.zeros(10, dtype=bool), "inflate")
    assert np.allclose(D, S)


def test_apply_lying_squat_sets_top_demand_families_to_one():
    S = np.full((4, 6), 0.3)
    liars = np.array([True, False, False, True])
    demand = np.array([0.05, 0.30, 0.10, 0.25, 0.20, 0.10])
    D = apply_lying(S, liars, "squat", demand)
    top3 = set(np.argsort(-demand, kind="stable")[:3].tolist())          # {1, 3, 4}
    assert top3 == {1, 3, 4}
    for a in (0, 3):
        for f in range(6):
            assert D[a, f] == (1.0 if f in top3 else 0.3)
    assert np.allclose(D[1], S[1]) and np.allclose(D[2], S[2])


def test_apply_lying_squat_needs_demand():
    with pytest.raises(ValueError):
        apply_lying(np.zeros((3, 4)), np.array([True, False, False]), "squat", None)


def test_apply_lying_rejects_unknown_mode():
    with pytest.raises(ValueError):
        apply_lying(np.zeros((3, 4)), np.array([True, False, False]), "shrink")


def test_world_wires_lying_into_declared():
    w = World(200, 8, "specialist", 0.25, seed=11)
    assert w.liars.sum() == 50
    honest = ~w.liars
    # liars declare higher than they are; honest declare within the declaration noise
    assert (w.D[w.liars] - w.S[w.liars]).mean() > 0.3
    assert abs(float((w.D[honest] - w.S[honest]).mean())) < 0.02


# ---------------------------------------------------------- report channel
def _liar_and_honest(w):
    liars = np.flatnonzero(w.liars)
    honest = np.flatnonzero(~w.liars)
    assert liars.size and honest.size
    return int(liars[0]), liars, honest


def test_honest_reporter_passes_outcomes_through():
    w = World(60, 8, "specialist", 0.5, seed=5)
    _, liars, honest = _liar_and_honest(w)
    j = int(honest[0])
    for a in list(honest[:5]) + list(liars[:5]):
        for o in (0, 1):
            assert w.report(j, int(a), o) == o


def test_colluding_liar_vouches_for_liars():
    w = World(60, 8, "specialist", 0.5, seed=5)
    j, liars, _ = _liar_and_honest(w)
    for a in liars[1:6]:
        assert w.report(j, int(a), 0) == 1          # reports 1 about a liar it saw fail


def test_colluding_liar_zeroes_the_top_honest_it_has_observed():
    w = World(60, 8, "specialist", 0.5, seed=5)
    j, _, honest = _liar_and_honest(w)
    top = int(honest[0])
    assert w.report(j, top, 1) == 0                 # first honest agent seen is trivially the top
    # a later, worse honest agent is reported truthfully
    other = int(honest[1])
    assert w.report(j, other, 1) == 1


def test_collude_false_disables_the_report_lie():
    w = World(60, 8, "specialist", 0.5, seed=5, collude=False)
    j, liars, honest = _liar_and_honest(w)
    for a in list(honest[:5]) + list(liars[1:5]):
        for o in (0, 1):
            assert w.report(j, int(a), o) == o


def test_report_many_collude_false_is_the_identity():
    w = World(60, 8, "specialist", 0.5, seed=5, collude=False)
    rng = np.random.default_rng(0)
    rep = rng.integers(0, w.n, size=200)
    ag = rng.integers(0, w.n, size=200)
    out = rng.integers(0, 2, size=200)
    assert np.array_equal(w.report_many(rep, ag, out), out.astype(np.int8))


def test_report_many_vouches_for_liars_and_zeroes_top_honest():
    w = World(60, 8, "specialist", 0.5, seed=5)
    j, liars, honest = _liar_and_honest(w)
    a_liar = int(liars[1])
    hs = [int(x) for x in honest[:5]]
    reporters = np.array([j] * 6)
    agents = np.array([a_liar] + hs)
    outcomes = np.array([0, 1, 1, 0, 0, 0])
    got = w.report_many(reporters, agents, outcomes)
    assert got[0] == 1                                   # vouched for the liar
    assert got[1] == 0                                   # top-20% honest (mean 1, lowest id) zeroed
    assert list(got[2:]) == [1, 0, 0, 0]                 # the rest pass through


def test_report_many_leaves_honest_reporters_alone():
    w = World(60, 8, "specialist", 0.5, seed=5)
    _, liars, honest = _liar_and_honest(w)
    j = int(honest[0])
    agents = np.array([int(liars[0]), int(honest[1]), int(honest[2])])
    outcomes = np.array([0, 1, 0])
    assert np.array_equal(w.report_many(np.array([j, j, j]), agents, outcomes), outcomes.astype(np.int8))


def test_report_and_report_many_agree_on_the_same_inputs():
    """The scalar path folds observations in one at a time and the vectorized path in one batch.
    Fed the same (reporter, agent, outcome) sequence in descending-mean order they must agree."""
    for seed in (5, 6, 9):
        w1 = World(60, 8, "specialist", 0.5, seed=seed)
        w2 = World(60, 8, "specialist", 0.5, seed=seed)
        j, liars, honest = _liar_and_honest(w1)
        a_liar = int(liars[1])
        hs = [int(x) for x in honest[:5]]
        agents = [a_liar] + hs
        outcomes = [0, 1, 1, 0, 0, 0]                 # honest means descend: 1,1,0,0,0
        scalar = [w1.report(j, a, o) for a, o in zip(agents, outcomes)]
        batch = w2.report_many(np.array([j] * len(agents)), np.array(agents), np.array(outcomes))
        assert scalar == [int(x) for x in batch]


def test_report_many_handles_multiple_reporters_independently():
    w = World(60, 8, "specialist", 0.5, seed=5)
    _, liars, honest = _liar_and_honest(w)
    j1, j2 = int(liars[0]), int(liars[1])
    hs = [int(x) for x in honest[:4]]
    reporters = np.array([j1] * 4 + [j2] * 4)
    agents = np.array(hs * 2)
    outcomes = np.array([1, 0, 0, 0] * 2)
    got = w.report_many(reporters, agents, outcomes)
    # each liar independently zeroes its own top honest agent (hs[0], mean 1)
    assert got[0] == 0 and got[4] == 0
    assert list(got[1:4]) == [0, 0, 0] and list(got[5:8]) == [0, 0, 0]


def test_no_liars_means_no_corruption():
    w = World(60, 8, "specialist", 0.0, seed=5)
    assert w.liars.sum() == 0
    rng = np.random.default_rng(0)
    rep, ag = rng.integers(0, w.n, size=50), rng.integers(0, w.n, size=50)
    out = rng.integers(0, 2, size=50)
    assert np.array_equal(w.report_many(rep, ag, out), out.astype(np.int8))
    for r, a, o in zip(rep, ag, out):
        assert w.report(int(r), int(a), int(o)) == int(o)
