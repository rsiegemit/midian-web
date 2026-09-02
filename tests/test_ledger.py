"""Ledger conservation: counters start at zero, each increment site adds exactly k,
snapshot/diff/reset behave, and the world/bus charge the amounts the contract promises."""
from __future__ import annotations

import numpy as np
import pytest

from rte.ledger import COUNTERS, Ledger
from rte.world import World

SITES = {"probes": "probe", "reports": "report", "messages": "message",
         "hops": "hop", "comparisons": "compare", "tasks": "task"}


def test_counters_start_at_zero():
    led = Ledger()
    assert led.snapshot() == {c: 0 for c in COUNTERS}


@pytest.mark.parametrize("counter,site", sorted(SITES.items()))
@pytest.mark.parametrize("k", [1, 2, 7, 1000])
def test_increment_site_adds_exactly_k(counter, site, k):
    led = Ledger()
    getattr(led, site)(k)
    snap = led.snapshot()
    assert snap[counter] == k
    assert all(snap[c] == 0 for c in COUNTERS if c != counter)


def test_default_increment_is_one():
    for counter, site in SITES.items():
        led = Ledger()
        getattr(led, site)()
        assert led.snapshot()[counter] == 1


def test_snapshot_is_a_copy_not_a_live_view():
    led = Ledger()
    before = led.snapshot()
    led.hop(5)
    assert before["hops"] == 0
    assert led.snapshot()["hops"] == 5


def test_diff_conservation():
    led = Ledger()
    led.probe(3); led.hop(2)
    before = led.snapshot()
    led.probe(4); led.compare(9); led.task(1)
    d = led.diff(before)
    assert d == {"probes": 4, "reports": 0, "messages": 0, "hops": 0, "comparisons": 9, "tasks": 1}
    # after = before + diff, componentwise
    after = led.snapshot()
    assert all(after[c] == before[c] + d[c] for c in COUNTERS)


def test_reset_zeroes_everything():
    led = Ledger()
    for site in SITES.values():
        getattr(led, site)(11)
    led.reset()
    assert led.snapshot() == {c: 0 for c in COUNTERS}


def test_split_increments_sum():
    """Charging k in pieces equals charging k at once (no per-call overhead)."""
    a, b = Ledger(), Ledger()
    for i in range(1, 11):
        a.probe(i)
    b.probe(sum(range(1, 11)))
    assert a.probes == b.probes


# --------------------------------------------------------------- world charge sites
def _world(**kw):
    kw.setdefault("beta", 0.25)
    return World(50, 8, "specialist", seed=3, **kw)


def test_world_probe_charges_exactly_one_probe():
    w = _world()
    before = w.ledger.snapshot()
    w.probe(0, 0)
    d = w.ledger.diff(before)
    assert d["probes"] == 1
    assert all(d[c] == 0 for c in COUNTERS if c != "probes")


@pytest.mark.parametrize("m,reps", [(1, 1), (5, 3), (50, 2), (7, 1)])
def test_probe_many_charges_len_times_reps(m, reps):
    w = _world()
    agents = np.arange(m) % w.n
    fams = np.zeros(m, dtype=int)
    before = w.ledger.snapshot()
    out = w.probe_many(agents, fams, reps)
    d = w.ledger.diff(before)
    assert out.shape == (m, reps)
    assert d["probes"] == m * reps
    assert all(d[c] == 0 for c in COUNTERS if c != "probes")


def test_probe_many_charges_broadcast_size():
    """agents and families are broadcast; the charge is the broadcast size * reps."""
    w = _world()
    agents = np.arange(4)
    fams = 2                                  # scalar broadcasts against 4 agents
    before = w.ledger.snapshot()
    w.probe_many(agents, np.asarray(fams), 3)
    assert w.ledger.diff(before)["probes"] == 4 * 3


def test_report_charges_exactly_one():
    w = _world()
    before = w.ledger.snapshot()
    w.report(1, 2, 1)
    d = w.ledger.diff(before)
    assert d["reports"] == 1
    assert all(d[c] == 0 for c in COUNTERS if c != "reports")


@pytest.mark.parametrize("size", [1, 4, 37])
def test_report_many_charges_size(size):
    w = _world()
    rng = np.random.default_rng(0)
    reporters = rng.integers(0, w.n, size=size)
    agents = rng.integers(0, w.n, size=size)
    outcomes = rng.integers(0, 2, size=size)
    before = w.ledger.snapshot()
    out = w.report_many(reporters, agents, outcomes)
    d = w.ledger.diff(before)
    assert out.size == size
    assert d["reports"] == size
    assert all(d[c] == 0 for c in COUNTERS if c != "reports")


def test_report_many_charges_2d_size():
    w = _world()
    reporters = np.arange(6).reshape(3, 2) % w.n
    agents = np.arange(6).reshape(3, 2) % w.n
    outcomes = np.ones((3, 2), dtype=int)
    before = w.ledger.snapshot()
    w.report_many(reporters, agents, outcomes)
    assert w.ledger.diff(before)["reports"] == 6


def test_bus_broadcast_charges_n():
    w = _world()
    view = w.view({"bus"})
    before = w.ledger.snapshot()
    view.bus.broadcast(0, "hi")
    d = w.ledger.diff(before)
    assert d["messages"] == w.n
    assert all(d[c] == 0 for c in COUNTERS if c != "messages")


def test_bus_send_and_send_many():
    w = _world()
    view = w.view({"bus"})
    before = w.ledger.snapshot()
    view.bus.send(0, 1, "x")
    assert w.ledger.diff(before)["messages"] == 1
    before = w.ledger.snapshot()
    view.bus.send_many(13)
    assert w.ledger.diff(before)["messages"] == 13


def test_world_execute_charges_one_task():
    w = _world()
    task = w.tasks(1)[0]
    before = w.ledger.snapshot()
    o = w.execute(0, task)
    d = w.ledger.diff(before)
    assert o in (0, 1)
    assert d["tasks"] == 1
    assert all(d[c] == 0 for c in COUNTERS if c != "tasks")


def test_oracle_is_free():
    """The oracle line is runner-only bookkeeping; it must not charge anything but its task calls."""
    w = _world()
    task = w.tasks(1)[0]
    before = w.ledger.snapshot()
    a = w.oracle(task)
    w.oracle_all()
    assert w.ledger.diff(before) == {c: 0 for c in COUNTERS}
    assert 0 <= a < w.n


def test_view_shares_the_world_ledger():
    w = _world()
    view = w.view({"probe"})
    assert view.ledger is w.ledger
    view.probe(0, 0)
    assert w.ledger.probes == 1
