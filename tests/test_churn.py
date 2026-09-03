"""Churn (v2): in-place replacement of agents, epoch scoring, deterministic per event, restored by reset, row ids of
non-churn cells unchanged, halving rebuild cost, runner integration."""
import numpy as np
from rte.budget import Budget
from rte.methods import load_method
from rte.run import CELL, row_id, run_method
from rte.world import World

W = dict(n=100, K=8, dist="specialist", beta=0.25, seed=3)


def test_churn_replaces_in_place_and_reset_restores():
    w = World(**W); S0, D0, L0 = w.S.copy(), w.D.copy(), w.liars.copy()
    w.probe_many(np.arange(100), 0, 2)
    ids = w.churn(0.1)
    keep = np.setdiff1d(np.arange(100), ids)
    assert ids.size == 10 and w.n == 100
    assert not np.allclose(w.S[ids], S0[ids]) and np.allclose(w.S[keep], S0[keep])
    assert np.allclose(w.D[keep], D0[keep]) and (w.liars[keep] == L0[keep]).all()
    assert (w._probe_idx[ids] == 0).all() and (w._probe_idx[keep, 0] == 2).all()
    assert (w.epoch[ids] == 1).all() and (w.epoch[keep] == 0).all()
    ids2 = World(**W).churn(0.1); assert (ids2 == ids).all()                 # same draw for every method
    w.reset()
    assert np.allclose(w.S, S0) and np.allclose(w.D, D0) and (w.liars == L0).all() and w.churn_events == 0


def test_epoch_rule_stale_route_scores_zero_once():
    w = World(**W); t = w.tasks(1)[0]; ids = w.churn(0.1); a = int(ids[0])
    w.S[a] = 1.0; w.backend._S[a] = 1.0                                          # would succeed if it were known
    assert w.execute(a, t) == 0 and w.execute(a, t) == 1                         # stale once, then real
    b = int(ids[1]); w.backend._S[b] = 1.0; w.probe_many(np.array([b]), 0, 1)
    assert w.execute(b, t) == 1                                                  # probing marks it seen


def test_row_id_unchanged_without_churn():
    cell = {**{k: v for k, v in zip(CELL, ("bernoulli", 100, 8, "specialist", 0.25, "random", True, "programmatic", "inflate", "uniform", 3, 100))},
            "backend_kwargs": {}}
    assert row_id(cell, "midian", {}, 1) == row_id({**cell, "churn": None}, "midian", {}, 1) != row_id({**cell, "churn": {"frac": 0.1, "every": 200}}, "midian", {}, 1)


def test_halving_rebuild_charges_full_budget_and_flat_reprobes():
    w = World(**W); ids = w.churn(0.1); w.reset()
    for name, params, probes in [("sequential_halving", {"churn_mode": "rebuild"}, None), ("sequential_halving", {"churn_mode": "stale"}, 0),
                                 ("flat_probe_argmax", {"online": True}, 10 * 8 * 3), ("warm_start_bandit", {}, 10 * 8 * 3)]:
        m = load_method(name)(**params); v = w.view(m.needs); w.reset(); m.build(v, Budget(3)); w.ledger.reset()
        ids = w.churn(0.1); m.churn(ids, ids); spent = w.ledger.snapshot()["probes"]
        assert spent == (probes if probes is not None else spent) and (probes is not None or 0 < spent <= 100 * 8 * 3), (name, spent)
        assert 0 <= m.fetch(w.tasks(1)[0]) < 100


def test_runner_churn_row():
    w = World(**W); stream = w.tasks(200)
    row = run_method(w, stream, {"name": "flat_probe_argmax", "params": {"online": True}}, 3, {"frac": 0.1, "every": 50})
    assert row["repair_probes_per_event"] == 10 * 8 * 3 and 0 <= row["success"] <= 1
    assert "repair_probes_per_event" not in run_method(w, stream, {"name": "random", "params": {}}, 3, None)


def test_llm_backend_redraw_is_offline_and_restorable():
    from rte.backends.llm import LLMBackend
    be = LLMBackend(n=20, K=16, dist="specialist", seed=1)               # no fleet needed until S is asked for
    snap = be.snapshot(); ids = np.array([2, 5]); rng = np.random.default_rng(0)
    be.redraw(ids, rng)
    assert be.profiles[2]["id"] == 2 and be.dir != snap[1] and be._S is None and len(be.profiles) == 20
    be.restore(snap); assert be.profiles == snap[0] and be.dir == snap[1]
