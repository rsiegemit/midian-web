"""Ledger conservation, and that the world/bus charge exactly what the CONTRACT promises.

`charges(world, fn, **expect)` is the whole idea: run `fn`, assert the ledger diff is exactly
`expect` and every other counter is untouched. One increment site per counter means a charge that
lands on the wrong counter is a bug, not a rounding difference.
"""
from __future__ import annotations

import numpy as np
import pytest

from rte.ledger import COUNTERS, Ledger
from rte.world import World

SITES = {"probes": "probe", "reports": "report", "messages": "message",
         "hops": "hop", "comparisons": "compare", "tasks": "task"}
ZERO = dict.fromkeys(COUNTERS, 0)


def charges(world, fn, **expect):
    """Assert `fn` moves exactly the named counters by the named amounts. Returns fn's value."""
    before = world.ledger.snapshot()
    out = fn()
    assert world.ledger.diff(before) == {**ZERO, **expect}
    return out


@pytest.fixture
def w():
    return World(50, 8, "specialist", 0.25, seed=3)


# --------------------------------------------------------------------- the counters themselves
def test_counters_start_at_zero():
    assert Ledger().snapshot() == ZERO


@pytest.mark.parametrize("counter,site", sorted(SITES.items()))
@pytest.mark.parametrize("k", [None, 1, 2, 7, 1000])
def test_increment_site_adds_exactly_k(counter, site, k):
    led = Ledger()
    getattr(led, site)(*([] if k is None else [k]))
    assert led.snapshot() == {**ZERO, counter: 1 if k is None else k}


def test_snapshot_is_a_copy_not_a_live_view():
    led = Ledger()
    before = led.snapshot()
    led.hop(5)
    assert before["hops"] == 0 and led.snapshot()["hops"] == 5


def test_diff_conservation():
    led = Ledger()
    led.probe(3); led.hop(2)
    before = led.snapshot()
    led.probe(4); led.compare(9); led.task(1)
    d = led.diff(before)
    assert d == {**ZERO, "probes": 4, "comparisons": 9, "tasks": 1}
    after = led.snapshot()
    assert all(after[c] == before[c] + d[c] for c in COUNTERS)      # after = before + diff


def test_reset_zeroes_everything():
    led = Ledger()
    for site in SITES.values():
        getattr(led, site)(11)
    led.reset()
    assert led.snapshot() == ZERO


def test_charging_k_in_pieces_equals_charging_k_at_once():
    a, b = Ledger(), Ledger()
    for i in range(1, 11):
        a.probe(i)
    b.probe(sum(range(1, 11)))
    assert a.probes == b.probes


# --------------------------------------------------------------------- the world's charge sites
def test_probe_charges_exactly_one_probe(w):
    charges(w, lambda: w.probe(0, 0), probes=1)


@pytest.mark.parametrize("m,reps", [(1, 1), (5, 3), (50, 2), (7, 1)])
def test_probe_many_charges_len_times_reps(w, m, reps):
    out = charges(w, lambda: w.probe_many(np.arange(m) % w.n, np.zeros(m, int), reps),
                  probes=m * reps)
    assert out.shape == (m, reps)


def test_probe_many_charges_the_broadcast_size(w):
    """A scalar family broadcasts against the agent list; the charge follows the broadcast."""
    charges(w, lambda: w.probe_many(np.arange(4), np.asarray(2), 3), probes=12)


def test_report_charges_exactly_one(w):
    charges(w, lambda: w.report(1, 2, 1), reports=1)


@pytest.mark.parametrize("shape", [(1,), (4,), (37,), (3, 2)])
def test_report_many_charges_one_per_report(w, shape):
    rng = np.random.default_rng(0)
    size = int(np.prod(shape))
    args = (rng.integers(0, w.n, shape), rng.integers(0, w.n, shape), rng.integers(0, 2, shape))
    out = charges(w, lambda: w.report_many(*args), reports=size)
    assert out.shape == shape


def test_bus_charges_per_message(w):
    bus = w.view({"bus"}).bus
    charges(w, lambda: bus.broadcast(0, "hi"), messages=w.n)
    charges(w, lambda: bus.send(0, 1, "x"), messages=1)
    charges(w, lambda: bus.send_many(13), messages=13)


def test_execute_charges_one_task(w):
    task = w.tasks(1)[0]
    assert charges(w, lambda: w.execute(0, task), tasks=1) in (0, 1)


def test_the_oracle_line_is_free(w):
    """The ceiling is runner-only bookkeeping: picking the oracle agent costs nothing."""
    task = w.tasks(1)[0]
    a = charges(w, lambda: (w.oracle(task), w.oracle_all())[0])
    assert 0 <= a < w.n


def test_view_shares_the_world_ledger(w):
    view = w.view({"probe"})
    assert view.ledger is w.ledger
    charges(w, lambda: view.probe(0, 0), probes=1)
