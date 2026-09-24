"""MIDIAN's defenses as parameters (audit, verify; both on by default) and the old-key mapping (rte.methods.keys):
budgets, exclusion of caught liars, the undefended accounting when the defenses are off, the cost of the full method
against MIDIAN w/o audits, and the bernoulli sanity checks."""
import json

import numpy as np
import pandas as pd
import pytest

from rte.budget import Budget
from rte.methods import keys
from rte.methods.midian import Midian
from rte.world import World

OFF = dict(audit=False, verify=False)                     # MIDIAN w/o defenses


def run(n=100, beta=0.25, seed=1, b=3, Q=300, **kw):
    w = World(n, 16, "specialist", beta, seed=seed, liar_select="low_skill_first")
    m = Midian(**kw); before = w.ledger.snapshot()
    m.build(w.view(m.needs), Budget(b)); build = w.ledger.diff(before)
    stream = w.tasks(Q); s = []
    for t in stream:
        a = m.fetch(t); o = w.execute(a, t); m.observe(t, a, o); s.append(o)
    return w, m, build, float(np.mean(s))


def test_defaults_are_the_full_method():
    m = Midian()
    assert m.audit and m.rate == 0.05 and m.verify and m.cached
    assert Midian(audit=0.1).rate == 0.1 and not Midian(**OFF).audit and not Midian(**OFF).cached


def test_wo_defenses_spends_the_plain_budget():
    """Defenses off: exactly n*K*b probes and one report per (peer, member, family, probe)."""
    _, m, plain, s = run(**OFF)
    assert plain["probes"] == 100 * 16 * 3 and plain["reports"] == plain["probes"] * 9
    assert m.est.shape == (100, 16) and np.isfinite(m.est).all() and s > 0.5


def test_audit_exclusion():
    """MIDIAN w/o verification: a colluding liar caught lying twice by the audits is excluded; audits cost <= 5% extra probes."""
    w, m, build, s = run(beta=0.5, verify=False)
    liars = w.liars
    assert m.excluded.any() and not m.excluded[~liars].any(), "audits excluded an honest reporter"
    assert (m.hits[m.excluded] >= 2).all() and liars[m.excluded].all()
    _, _, plain, _ = run(beta=0.5, **OFF)
    assert plain["probes"] < build["probes"] <= 1.05 * plain["probes"] * 1.02     # 5% audits (binomial slack)
    assert s > 0.5


def test_audit_no_liars_no_exclusions():
    _, m, _, _ = run(beta=0.0, verify=False)
    assert not m.excluded.any()


def test_online_audit_charges_reports():
    w, m, _, _ = run(beta=0.5, Q=1, verify=False)
    m.rate = 1.0                                          # audit every routed outcome
    w.ledger.reset()
    for t in w.tasks(50):
        a = m.fetch(t); m.observe(t, a, w.execute(a, t))
    assert w.ledger.reports > 0


def test_full_method_costs_like_wo_audit():
    """MIDIAN = MIDIAN w/o audits + audits: build probes within 1.05x of w/o audits on the same world; with verify (and its
    cached root pick) a fetch charges 1 comparison + 2 messages."""
    out = {}
    for tag, kw in (("wo_audit", dict(audit=False)), ("full", {})):
        w = World(n=200, K=8, dist="specialist", beta=0.25, seed=3, backend="bernoulli"); m = Midian(**kw)
        v = w.view(m.needs); m.build(v, Budget(3)); out[tag] = dict(w.ledger.snapshot())
        w.ledger.reset(); t = w.tasks(20)[0]; a = m.fetch(t); assert 0 <= a < w.n
        assert w.ledger.snapshot()["comparisons"] == 1 and w.ledger.snapshot()["messages"] == 2
    assert out["full"]["probes"] <= 1.05 * out["wo_audit"]["probes"] + 1


# ---------------------------------------------------------------- rte.methods.keys: the old keys
OLD = [("midian_va", {}, ("midian", {})),
       ("midian_a", {}, ("midian", {"verify": False})),
       ("midian_v", {}, ("midian", {"audit": False})),
       ("midian", {}, ("midian", {"audit": False, "verify": False})),
       ("midian", {"r": 5}, ("midian", {"audit": False, "verify": False, "r": 5})),
       ("midian_v", {"r": 5}, ("midian", {"audit": False, "r": 5})),
       ("midian_va", {"cohort": "block"}, ("midian", {"cohort": "block"})),
       ("fw_langgraph", {"retrieval": "midian_va", "r": 10}, ("fw_langgraph", {"retrieval": "midian", "r": 10})),
       ("fw_langgraph", {"retrieval": "midian", "r": 5}, ("fw_langgraph", {"retrieval": "midian_wo_audit", "r": 5}))]


@pytest.mark.parametrize("old,params,new", OLD)
def test_keys_round_trip(old, params, new):
    got = keys.to_new(old, params)
    assert got == new and keys.legacy(*got) == (old, params)


def test_keys_old_verify_cached_is_wo_audit():
    """midian{verify, cached} spelled MIDIAN w/o audits before the rename: it maps to w/o audits (whose legacy key is midian_v)."""
    assert keys.to_new("midian", {"verify": True, "cached": True}) == ("midian", {"audit": False})


@pytest.mark.parametrize("old", ["midian_sh", "midian_sha"])
def test_keys_withdrawn(old):
    assert keys.to_new(old, {}) is None


def test_normalize_translates_and_drops_withdrawn(tmp_path):
    df = pd.DataFrame({"method": ["midian_va", "midian", "midian_v", "midian_a", "midian_sh", "midian_sha", "random"],
                       "params": ["{}", '{"r":5}', "{}", "{}", "{}", "{}", "{}"], "success": np.arange(7.0)})
    out = keys.normalize(df, str(tmp_path))
    assert list(out.success) == [0.0, 1.0, 2.0, 3.0, 6.0]                                   # SH / SHA rows removed
    got = [(m, json.loads(p)) for m, p in zip(out.method, out.params)]
    assert got == [("midian", {}), ("midian", {"audit": False, "r": 5, "verify": False}), ("midian", {"audit": False}),
                   ("midian", {"verify": False}), ("random", {})]


def test_normalize_checks_a_migrated_directory(tmp_path):
    (tmp_path / keys.SENTINEL).touch()
    new = pd.DataFrame({"method": ["midian", "midian"], "params": ["{}", '{"audit":false,"verify":false}']})
    assert keys.normalize(new, str(tmp_path)) is new                                         # current keys: untouched
    with pytest.raises(AssertionError):
        keys.normalize(pd.DataFrame({"method": ["midian_va"], "params": ["{}"]}), str(tmp_path))
