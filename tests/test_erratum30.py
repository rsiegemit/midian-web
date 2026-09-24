"""Erratum 30: the opt-in `calibrated` declared source, replay `split`, routereval `no_repeat`.

Regression: every default cell must stay bit-identical. GOLDEN holds fingerprints (S, both declared sources, liars, the
task stream, oracle outcomes, a probe sweep and four methods' success) computed with the code BEFORE erratum 30."""
from __future__ import annotations

import hashlib, os

import numpy as np
import pytest

from rte.backends import CAL_VALUES, calibrated_declared
from rte.run import run_method
from rte.world import World, apply_lying

DATA = os.environ.get("RTE_DATA", os.path.expanduser("~/rte_data"))
REPLAY_NPZ = f"{DATA}/data/routerbench_cells.npz"
LRB = f"{DATA}/data/llmrouterbench/perf_matrix.npz"
MMLU = f"{DATA}/data/routereval/router_dataset/mmlu_router_dataset.pkl"
LIVE_S = f"{DATA}/populations/specialist_n1000_K16_seed1/S.npy"
METHODS = [{"name": "declared_argmax", "params": {}}, {"name": "flat_probe_argmax", "params": {"online": True}},
           {"name": "warm_start_bandit", "params": {}}, {"name": "midian_va", "params": {}}]


def fake_cells(path, K=8, M=4, prompts=40):
    rng = np.random.default_rng(0)
    bias = rng.uniform(0.2, 0.9, M)
    np.savez(path, model_names=np.array([f"m{m}" for m in range(M)]), category_names=np.array([f"c{k}" for k in range(K)]),
             offsets=np.arange(K + 1, dtype=np.int64) * prompts, n_prompts=np.full(K, prompts, np.int64),
             outcomes=(rng.random((K * prompts, M)) < bias).astype(np.int8))
    return path


def fingerprint(kw, Q=200):
    w = World(**kw)
    h = hashlib.sha256()
    for x in (w.S, np.asarray(w.backend.declared("programmatic")), np.asarray(w.backend.declared("self_described")), w.D, w.liars):
        h.update(np.ascontiguousarray(x).tobytes())
    stream = w.tasks(Q)
    h.update(np.array([(t.family, t.instance) for t in stream], np.int64).tobytes())
    h.update(np.array([w.execute(w.oracle(t), t) for t in stream], np.int8).tobytes())
    w.reset()
    h.update(w.probe_many(np.arange(w.n)[:, None], np.arange(w.K)[None, :], 3).tobytes())
    for s in METHODS:
        h.update(repr(round(run_method(w, stream, s, 3)["success"], 12)).encode())
    return h.hexdigest()[:16]


def worlds(tmp):
    base = dict(beta=0.5, liar_select="low_skill_first", seed=3)
    out = {"bernoulli": dict(n=60, K=8, dist="specialist", backend="bernoulli", **base),
           "replay_fake": dict(n=60, K=8, dist="specialist", backend="replay", backend_kwargs={"cells_path": fake_cells(f"{tmp}/c.npz")}, **base)}
    if os.path.exists(LIVE_S):
        out["bernoulli_cal"] = dict(n=60, K=16, dist="specialist", backend="bernoulli", backend_kwargs={"calibrate_from": LIVE_S}, **base)
    if os.path.exists(REPLAY_NPZ):
        out["replay_real"] = dict(n=60, K=64, dist="heavy_tail", backend="replay", **base)
    if os.path.exists(LRB):
        out["llmrouterbench"] = dict(n=20, K=15, dist="all", backend="routereval", backend_kwargs={"dataset": "llmrouterbench"}, **base)
    if os.path.exists(MMLU):
        out["mmlu10"] = dict(n=10, K=16, dist="strong_to_weak", backend="routereval", backend_kwargs={"dataset": "mmlu"}, **base)
    return out


GOLDEN = {"bernoulli": "5bd553e81e85e757", "replay_fake": "a33078be53c1df6e", "bernoulli_cal": "69c40b9ca984e638",
          "replay_real": "e8ae02c33b542ebd", "llmrouterbench": "bd277439fbc57d68", "mmlu10": "834a58cc27147de8"}


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return str(tmp_path_factory.mktemp("e30"))


# ------------------------------------------------------------------ defaults are bit-identical
@pytest.mark.parametrize("name", ["bernoulli", "replay_fake", "bernoulli_cal", "replay_real", "llmrouterbench", "mmlu10"])
def test_default_bit_identical(tmp, name):
    kw = worlds(tmp).get(name)
    if kw is None:
        pytest.skip("data not present")
    assert fingerprint(kw) == GOLDEN[name]


def test_explicit_off_equals_default(tmp):
    kw = worlds(tmp)["replay_fake"]
    off = {**kw, "backend_kwargs": {**kw["backend_kwargs"], "split": False}}
    assert fingerprint(off) == fingerprint(kw)


def test_options_change_the_fingerprint(tmp):
    """The fingerprint is sensitive: each option does change the world it applies to."""
    w = worlds(tmp)
    kw = w["replay_fake"]
    assert fingerprint({**kw, "backend_kwargs": {**kw["backend_kwargs"], "split": True}}) != GOLDEN["replay_fake"]
    assert fingerprint({**w["bernoulli"], "declared_source": "calibrated"}) != GOLDEN["bernoulli"]
    if "llmrouterbench" in w:
        kw = w["llmrouterbench"]
        assert fingerprint({**kw, "backend_kwargs": {**kw["backend_kwargs"], "no_repeat": True}}) != GOLDEN["llmrouterbench"]


# ------------------------------------------------------------------ FIX A: calibrated declarations
def test_calibrated_reproduces_live_stats():
    if not os.path.exists(LIVE_S):
        pytest.skip("live population not present")
    S = np.load(LIVE_S)
    live = np.load(LIVE_S.replace("S.npy", "D_self_described.npy"))
    D = calibrated_declared(S, 1)
    assert set(np.unique(D)) <= set(CAL_VALUES)
    assert abs(D.mean() - live.mean()) < 0.01 and abs(D.std() - live.std()) < 0.01
    assert abs(np.corrcoef(S.ravel(), D.ravel())[0, 1] - np.corrcoef(S.ravel(), live.ravel())[0, 1]) < 0.03


def test_calibrated_deterministic_per_seed_and_chunk_safe():
    S = np.random.default_rng(0).random((600_000, 4)).astype(np.float32)     # > CAL_CHUNK rows: crosses a chunk boundary
    a, b, c = calibrated_declared(S, 7), calibrated_declared(S, 7), calibrated_declared(S, 8)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a.dtype == np.float32 and a.shape == S.shape


@pytest.mark.parametrize("name", ["bernoulli", "replay_fake", "llmrouterbench"])
def test_calibrated_on_every_backend_and_lying_unchanged(tmp, name):
    kw = worlds(tmp).get(name)
    if kw is None:
        pytest.skip("data not present")
    w = World(**{**kw, "declared_source": "calibrated"})
    honest = w.backend.declared("calibrated")
    assert np.array_equal(honest, calibrated_declared(w.S, w.seed))
    assert w.liars.sum() == round(0.5 * w.n)
    assert np.array_equal(w.D, apply_lying(honest, w.liars, "inflate"))
    assert np.array_equal(w.D[w.liars], np.clip(honest[w.liars] + 0.4, 0, 1)) and np.array_equal(w.D[~w.liars], honest[~w.liars])
    # the liar set is chosen on S, which the declared source does not touch
    assert np.array_equal(w.liars, World(**kw).liars)


# ------------------------------------------------------------------ FIX B: replay split
@pytest.mark.parametrize("real", [False, True])
def test_replay_split_disjoint_rows(tmp, real):
    if real and not os.path.exists(REPLAY_NPZ):
        pytest.skip("real cells not present")
    kw = dict(n=60, K=64 if real else 8, dist="specialist", beta=0.5, liar_select="low_skill_first", seed=3, backend="replay",
              backend_kwargs={"split": True, **({} if real else {"cells_path": fake_cells(f"{tmp}/c.npz")})})
    w = World(**kw); b = w.backend
    d = np.load(REPLAY_NPZ if real else kw["backend_kwargs"]["cells_path"])
    O, off, npr = d["outcomes"], d["offsets"], d["n_prompts"]
    sel = np.sort(np.argsort(-npr, kind="stable")[:b.K])
    for f, c in enumerate(sel):
        p, t = b.probe_rows[f], b.task_rows[f]
        assert not set(p) & set(t) and sorted([*p, *t]) == list(range(off[c], off[c] + npr[c]))
        assert len(p) == int(0.7 * npr[c])
        assert np.allclose(b._model_cat_acc[:, f], O[p].mean(0))          # S reads the probe rows only
    # execute -> task rows; execute_many (probes) -> probe rows
    for t in w.tasks(300):
        a = int(np.argmax(w.S[:, t.family])); f = t.family
        m = b._weakest_model[f] if b.mask[a, f] else b.model_id[a]
        assert w.backend.execute(a, t) == O[b.task_rows[f][t.instance % len(b.task_rows[f])], m]
    a, f, inst = np.arange(w.n) % w.n, np.arange(w.n) % w.K, np.arange(w.n) * 7919
    m = np.where(b.mask[a, f], b._weakest_model[f], b.model_id[a])
    rows = np.array([b.probe_rows[ff][i % len(b.probe_rows[ff])] for ff, i in zip(f, inst)])
    assert np.array_equal(b.execute_many(a, f, inst), O[rows, m])


# ------------------------------------------------------------------ FIX C: no repeated test prompts
class _Pool:
    no_repeat = True
    def __init__(self, sizes): self.sizes = np.array(sizes)
    def task_pool_sizes(self): return self.sizes


def test_no_repeat_stream_unit(tmp):
    w = World(**worlds(tmp)["bernoulli"])
    w.backend = _Pool([3, 5, 40, 40, 40, 40, 40, 40])
    s = w.tasks(200)
    pairs = [(t.family, t.instance) for t in s]
    assert len(set(pairs)) == 200 and all(0 <= i < w.backend.sizes[f] for f, i in pairs)
    assert pairs == [(t.family, t.instance) for t in w.tasks(200)]            # one deterministic stream per (seed, cell)
    counts = np.bincount([f for f, _ in pairs], minlength=8)
    assert counts[0] == 3 and counts[1] == 5                                  # small families exhausted, never repeated
    first = [f for f, _ in pairs[:24]]                                        # demand is uniform while all have prompts
    assert len(set(first)) >= 6
    assert len(w.tasks(248)) == 248                                         # the whole pool, each prompt once
    with pytest.raises(ValueError, match="exceeds the test pool"):
        w.tasks(249)


def test_no_repeat_llmrouterbench():
    if not os.path.exists(LRB):
        pytest.skip("LLMRouterBench not present")
    kw = dict(n=20, K=15, dist="all", beta=0.5, liar_select="low_skill_first", seed=3, backend="routereval",
              backend_kwargs={"dataset": "llmrouterbench", "no_repeat": True})
    w = World(**kw); b = w.backend
    s = w.tasks(390)
    rows = [b._te[t.family][t.instance] for t in s]
    assert len(set(rows)) == 390 and all(t.instance < len(b._te[t.family]) for t in s)
    assert [w.execute(0, t) for t in s[:50]] == [int(b._Yte[r, 0]) for r in rows[:50]]
    assert b.text(s[0].family, s[0].instance) == str(b._Pte[rows[0]])
    with pytest.raises(ValueError):
        w.tasks(int(b.task_pool_sizes().sum()) + 1)


# ------------------------------------------------------------------ probe index with repeated cells in one call (audit F4)
def test_probe_call_without_duplicates_unchanged(tmp):
    from rte.world import probe_seed
    w = World(**worlds(tmp)["bernoulli"])
    w._probe_idx[3, 2] = 7
    a, f = np.array([3, 4, 5]), np.array([2, 2, 0])
    _, inst = w._probe(a, f, 3)
    k = np.array([7, 0, 0])[:, None] + np.arange(3)                    # the pre-fix formula
    assert np.array_equal(inst, probe_seed(w._probe_salt, a[:, None], f[:, None], k))
    assert w._probe_idx[3, 2] == 10 and w._probe_idx[4, 2] == 3 and w._probe_idx[5, 0] == 3


def test_probe_call_with_duplicates_advances_like_sequential_calls(tmp):
    w = World(**worlds(tmp)["bernoulli"])
    a, f = np.array([2, 5, 2, 2]), np.array([1, 1, 1, 0])
    out, inst = w._probe(a, f, 2)
    assert w._probe_idx[2, 1] == 4 and w._probe_idx[5, 1] == 2 and w._probe_idx[2, 0] == 2
    assert len({*inst[0], *inst[2]}) == 4                               # the repeated cell got fresh instances
    w.reset()
    seq = [w._probe(np.array([x]), np.array([y]), 2) for x, y in zip(a, f)]
    assert np.array_equal(inst, np.concatenate([s[1] for s in seq])) and np.array_equal(out, np.concatenate([s[0] for s in seq]))


# ------------------------------------------------------------------ routereval shuffle: random agent order, so index tie-breaks are random
def test_shuffle_permutes_pool_deterministically():
    if not os.path.exists(LRB):
        pytest.skip("LLMRouterBench not present")
    kw = dict(n=20, K=15, dist="all", beta=0.5, liar_select="low_skill_first", seed=3, backend="routereval", declared_source="calibrated")
    base = World(**kw, backend_kwargs={"dataset": "llmrouterbench"}).backend
    a, b = (World(**kw, backend_kwargs={"dataset": "llmrouterbench", "shuffle": True}).backend for _ in range(2))
    assert a.model_names == b.model_names and np.array_equal(a._Yte, b._Yte)                  # deterministic per seed
    perm = [base.model_names.index(m) for m in a.model_names]
    assert sorted(perm) == list(range(20)) and perm != list(range(20))
    assert np.array_equal(a._Ytr, base._Ytr[:, perm]) and np.array_equal(a._Yte, base._Yte[:, perm]) and np.allclose(a._S, base._S[perm])
    c = World(**{**kw, "seed": 4}, backend_kwargs={"dataset": "llmrouterbench", "shuffle": True}).backend
    assert c.model_names != a.model_names                                                    # a new order per seed


def test_shuffle_makes_declared_argmax_ties_random():
    if not os.path.exists(MMLU):
        pytest.skip("RouterEval mmlu not present")
    kw = dict(n=1000, K=16, dist="strong_to_weak", beta=0, seed=1, backend="routereval", declared_source="calibrated")
    pick = lambda w: w.S[np.argmax(w.D, 0), np.arange(w.K)].mean()
    assert pick(World(**kw, backend_kwargs={"dataset": "mmlu"})) < 0.3                       # stored weak -> strong: ties pick weak
    assert 0.55 < pick(World(**kw, backend_kwargs={"dataset": "mmlu", "shuffle": True})) < 0.7


def _probe_index_reference(idx, K, agents, families, reps):
    """The first post-fix bookkeeping (np.add.at on every call): the fast path must match it exactly."""
    from rte.world import _occurrence
    agents, families = np.broadcast_arrays(np.asarray(agents, np.int64), np.asarray(families, np.int64))
    k0 = idx[agents, families].astype(np.int64)
    np.add.at(idx, (agents, families), reps)
    if (idx[agents, families] - k0 != reps).any():
        k0 = k0 + reps * _occurrence(agents * K + families)
    return k0[..., None] + np.arange(reps)


def test_probe_fast_path_matches_reference(tmp):
    from rte.world import probe_seed
    w = World(**worlds(tmp)["bernoulli"])
    rng = np.random.default_rng(0)
    ref = w._probe_idx.copy()
    for _ in range(200):
        shape = [(7,), (5, 3), (4, 1)][rng.integers(3)]
        a = rng.integers(0, w.n if rng.random() < .5 else 3, shape)            # sometimes few agents: many repeats
        f = rng.integers(0, w.K, shape[-1:]) if rng.random() < .5 else rng.integers(0, 2, shape)
        reps = int(rng.integers(1, 4))
        k = _probe_index_reference(ref, w.K, a, f, reps)
        _, inst = w._probe(a, f, reps)
        A, F = np.broadcast_arrays(a, f)
        assert np.array_equal(inst, probe_seed(w._probe_salt, A[..., None], F[..., None], k))
        assert np.array_equal(w._probe_idx, ref)


# TrueSkill is the only arm that repeats a cell in one probe call, so the fix changes its rows (and only its rows).
# (success, build_probes) at b = 1, 3, 5 on bernoulli n = 60 and LLMRouterBench, seed 3, cartel. Pre-fix: 9831034b2e2f39c6.
TRUESKILL_FIXED = "0d559758b7279e32"


def test_trueskill_golden_after_probe_fix():
    if not os.path.exists(LRB):
        pytest.skip("LLMRouterBench not present")
    out = []
    for backend, kw in [("bernoulli", dict(n=60, K=8, dist="specialist")),
                        ("routereval", dict(n=20, K=15, dist="all", backend_kwargs={"dataset": "llmrouterbench"}))]:
        for b in (1, 3, 5):
            w = World(beta=0.5, liar_select="low_skill_first", seed=3, backend=backend, **kw)
            r = run_method(w, w.tasks(200), {"name": "trueskill_per_family", "params": {}}, b)
            out += [round(r["success"], 12), r["build_probes"]]
    assert hashlib.sha256(repr(out).encode()).hexdigest()[:16] == TRUESKILL_FIXED
