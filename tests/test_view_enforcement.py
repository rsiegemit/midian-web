"""View access control and the lying model.

Two halves. (1) A View raises AccessError on anything outside `needs` -- including the mutation
test, where the only difference between a method that works and one that blows up is the
declaration itself. (2) `select_liars` / `apply_lying` / the report channel behave as SPEC §4 says,
and the scalar (`report`) and vectorized (`report_many`) paths agree on the same inputs.
"""
from __future__ import annotations

import numpy as np
import pytest

from rte.budget import Budget
from rte.methods.base import Method
from rte.world import AccessError, DELTA_INFLATE, World, apply_lying, select_liars

PROBE_ONLY = {"probe"}


def world(beta=0.25, n=60, **kw):
    return World(n, 8, "specialist", beta, seed=5, **kw)


# =========================================================== 1. view enforcement
@pytest.mark.parametrize("what,call", [
    ("declared", lambda v: v.declared),
    ("report_channel", lambda v: v.report_channel(0, 1, 1)),
    ("report_many", lambda v: v.report_many([0], [1], [1])),
    ("bus", lambda v: v.bus),
])
def test_probe_only_view_raises_on_everything_else(what, call):
    with pytest.raises(AccessError):
        call(world().view(PROBE_ONLY))


@pytest.mark.parametrize("attr", ["S", "liars", "D", "backend", "oracle", "beta", "seed", "world"])
def test_view_raises_on_any_unknown_attribute(attr):
    """S and the liar set are never exposed, and neither is anything else undeclared."""
    with pytest.raises(AccessError):
        getattr(world().view(PROBE_ONLY), attr)


@pytest.mark.parametrize("call", [lambda v: v.probe(0, 0), lambda v: v.probe_many([0], [0], 1)])
def test_empty_needs_raises_on_probe(call):
    with pytest.raises(AccessError):
        call(world().view(set()))


def test_declared_view_allows_declared_and_nothing_else():
    w = world()
    v = w.view({"declared"})
    assert v.declared.shape == (w.n, w.K)
    for call in (lambda: v.probe(0, 0), lambda: v.bus):
        with pytest.raises(AccessError):
            call()


def test_always_available_fields():
    w = world()
    v = w.view(set())
    assert (v.n, v.K, len(v.families), v.ledger, v.needs) == (w.n, w.K, w.K, w.ledger, frozenset())
    assert isinstance(v.rng, np.random.Generator)


def test_declared_is_a_read_only_copy_that_cannot_reach_true_skill():
    w = world()
    v = w.view({"declared"})
    D = v.declared
    assert D.flags.writeable is False
    with pytest.raises(ValueError):
        D[0, 0] = 0.5
    with pytest.raises(AttributeError):                 # the property itself cannot be rebound
        v.declared = np.zeros((v.n, v.K))
    assert D is not w.S and not np.shares_memory(D, w.S)


@pytest.mark.parametrize("needs", [{"telepathy"}, {"probe", "S"}])
def test_unknown_need_name_raises_value_error(needs):
    with pytest.raises(ValueError):
        world().view(needs)


def test_view_rng_is_reproducible_and_differs_across_seeds_and_needs():
    w = world()
    a = w.view(PROBE_ONLY, seed=1).rng.random(16)
    assert np.array_equal(a, w.view(PROBE_ONLY, seed=1).rng.random(16))
    assert not np.array_equal(a, w.view(PROBE_ONLY, seed=2).rng.random(16))
    assert not np.array_equal(a[:8], w.view({"probe", "reports"}, seed=1).rng.random(8))


# ---------------------------------------------------------- mutation test
class _Reader(Method):
    """Reads the declared channel and probes. `needs` is set per test: that is the mutation."""
    name = "_reader"
    needs = frozenset({"probe", "declared"})

    def build(self, view, budget):
        self.D = np.asarray(view.declared)
        self.est = view.probe_many(np.arange(view.n), np.zeros(view.n, int), 1)

    def fetch(self, task):
        return int(np.argmax(self.D[:, task.family]))


@pytest.mark.parametrize("dropped", ["declared", "probe"])
def test_mutation_dropping_a_need_the_method_uses_raises(dropped):
    w = world()
    m = _Reader()
    with pytest.raises(AccessError):
        m.build(w.view(_Reader.needs - {dropped}), Budget(1))


def test_mutation_control_the_same_method_with_the_needs_declared_works():
    """The only thing that changed from the tests above is the declaration."""
    w = world()
    m = _Reader()
    m.build(w.view(_Reader.needs), Budget(1))
    assert m.D.shape == (w.n, w.K)
    assert 0 <= m.fetch(w.tasks(1)[0]) < w.n


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
    assert not np.array_equal(liars, select_liars(S, 0.25, "random", np.random.default_rng(7)))


def test_select_liars_rejects_unknown_mode():
    with pytest.raises(ValueError):
        select_liars(np.zeros((10, 2)), 0.5, "sneaky", np.random.default_rng(0))


def test_apply_lying_inflate():
    S = np.array([[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]], dtype=np.float32)
    liars = np.array([True, False, True])
    D = apply_lying(S, liars, "inflate")
    assert np.allclose(D[1], S[1])                                      # honest untouched
    assert np.allclose(D[[0, 2]], np.clip(S[[0, 2]] + DELTA_INFLATE, 0, 1))
    assert 0.0 <= D.min() and D.max() <= 1.0
    assert not np.shares_memory(D, S)                                    # the input is not mutated
    assert np.allclose(apply_lying(S, np.zeros(3, bool), "inflate"), S)  # no liars -> no change


def test_apply_lying_squat_sets_the_top_demand_families_to_one():
    S = np.full((4, 6), 0.3)
    liars = np.array([True, False, False, True])
    demand = np.array([0.05, 0.30, 0.10, 0.25, 0.20, 0.10])
    D = apply_lying(S, liars, "squat", demand)
    top3 = {1, 3, 4}
    assert set(np.argsort(-demand, kind="stable")[:3].tolist()) == top3
    for a in (0, 3):
        assert [D[a, f] for f in range(6)] == [1.0 if f in top3 else 0.3 for f in range(6)]
    assert np.allclose(D[[1, 2]], S[[1, 2]])


@pytest.mark.parametrize("mode,demand", [("squat", None), ("shrink", np.ones(4))])
def test_apply_lying_rejects_bad_arguments(mode, demand):
    with pytest.raises(ValueError):
        apply_lying(np.zeros((3, 4)), np.array([True, False, False]), mode, demand)


def test_world_wires_lying_into_the_declared_channel():
    w = World(200, 8, "specialist", 0.25, seed=11)
    assert w.liars.sum() == 50
    assert (w.D[w.liars] - w.S[w.liars]).mean() > 0.3                    # liars overclaim
    assert abs(float((w.D[~w.liars] - w.S[~w.liars]).mean())) < 0.02     # honest declare truthfully


# ---------------------------------------------------------- report channel
def parties(w):
    liars, honest = np.flatnonzero(w.liars), np.flatnonzero(~w.liars)
    assert liars.size and honest.size
    return int(liars[0]), liars, honest


def test_honest_reporter_passes_every_outcome_through():
    w = world(0.5)
    _, liars, honest = parties(w)
    j = int(honest[0])
    for a in list(honest[:5]) + list(liars[:5]):
        assert [w.report(j, int(a), o) for o in (0, 1)] == [0, 1]


def test_colluding_liar_vouches_for_liars_and_zeroes_the_top_honest_it_has_seen():
    w = world(0.5)
    j, liars, honest = parties(w)
    for a in liars[1:6]:
        assert w.report(j, int(a), 0) == 1              # reports 1 about a liar it saw fail
    assert w.report(j, int(honest[0]), 1) == 0          # the first honest agent seen is its top
    assert w.report(j, int(honest[1]), 1) == 1          # a later, worse one is reported truthfully


def test_collude_false_disables_the_report_lie_on_both_paths():
    w = world(0.5, collude=False)
    j, liars, honest = parties(w)
    for a in list(honest[:5]) + list(liars[1:5]):
        assert [w.report(j, int(a), o) for o in (0, 1)] == [0, 1]
    rng = np.random.default_rng(0)
    rep, ag, out = (rng.integers(0, w.n, 200), rng.integers(0, w.n, 200), rng.integers(0, 2, 200))
    assert np.array_equal(w.report_many(rep, ag, out), out.astype(np.int8))


def test_report_many_vouches_for_liars_and_zeroes_the_top_honest():
    w = world(0.5)
    j, liars, honest = parties(w)
    hs = [int(x) for x in honest[:5]]
    got = w.report_many(np.array([j] * 6), np.array([int(liars[1])] + hs),
                        np.array([0, 1, 1, 0, 0, 0]))
    assert got[0] == 1                                   # vouched for the liar
    assert got[1] == 0                                   # top-20% honest (mean 1, lowest id) zeroed
    assert list(got[2:]) == [1, 0, 0, 0]                 # the rest pass through


def test_report_many_leaves_honest_reporters_alone():
    w = world(0.5)
    _, liars, honest = parties(w)
    j = int(honest[0])
    agents, outcomes = np.array([int(liars[0]), int(honest[1]), int(honest[2])]), np.array([0, 1, 0])
    assert np.array_equal(w.report_many(np.array([j] * 3), agents, outcomes), outcomes.astype(np.int8))


@pytest.mark.parametrize("seed", [5, 6, 9])
def test_report_and_report_many_agree_on_the_same_inputs(seed):
    """The scalar path folds observations in one at a time and the batch path all at once. Fed the
    same (reporter, agent, outcome) sequence in descending-mean order they must agree."""
    w1, w2 = World(60, 8, "specialist", 0.5, seed=seed), World(60, 8, "specialist", 0.5, seed=seed)
    j, liars, honest = parties(w1)
    agents = [int(liars[1])] + [int(x) for x in honest[:5]]
    outcomes = [0, 1, 1, 0, 0, 0]                        # honest means descend: 1, 1, 0, 0, 0
    scalar = [w1.report(j, a, o) for a, o in zip(agents, outcomes)]
    batch = w2.report_many(np.array([j] * 6), np.array(agents), np.array(outcomes))
    assert scalar == [int(x) for x in batch]


def test_report_many_treats_each_liar_reporter_independently():
    w = world(0.5)
    _, liars, honest = parties(w)
    j1, j2 = int(liars[0]), int(liars[1])
    hs = [int(x) for x in honest[:4]]
    got = w.report_many(np.array([j1] * 4 + [j2] * 4), np.array(hs * 2),
                        np.array([1, 0, 0, 0] * 2))
    assert got[0] == 0 and got[4] == 0                   # each zeroes its own top honest agent
    assert list(got[1:4]) == [0, 0, 0] and list(got[5:8]) == [0, 0, 0]


def test_no_liars_means_no_corruption_anywhere():
    w = world(0.0)
    assert w.liars.sum() == 0
    rng = np.random.default_rng(0)
    rep, ag, out = (rng.integers(0, w.n, 50), rng.integers(0, w.n, 50), rng.integers(0, 2, 50))
    assert np.array_equal(w.report_many(rep, ag, out), out.astype(np.int8))
    assert [w.report(int(r), int(a), int(o)) for r, a, o in zip(rep, ag, out)] == list(map(int, out))
