"""The SOTA retrieval stack: BM25 alone, BM25+dense reciprocal-rank fusion, and fusion + cross-encoder rerank.
The dense half is MiniLM here (the strong Qwen3 embedder is a parameter, not a code path) and the reranker is
stubbed, so the test asserts the RANKING LOGIC without downloading 23 GB of weights."""
from __future__ import annotations

import numpy as np
import pytest

from rte.budget import Budget
from rte.methods._learned import embed
from rte.methods.frameworks._common import FrameworkMethod, _bm25, _rrf
from rte.world import World

N, K, Q = 300, 16, 8


class _Fw(FrameworkMethod):
    name, env, worker = "fw_test", "rte", "echo_worker.py"

    def _texts(self, view):                       # 3 templates per family -> clones dominate, as on the live backend
        D, fams = view.declared, list(view.families)
        desc = [f"expert in {fams[int(np.argmax(D[a]))]} level {int(D[a].max() * 3)}" for a in range(view.n)]
        return desc, [f"Tasks of family {f}" for f in fams], (lambda task: f"task {task.family}")


def _built(**kw):
    m = _Fw(base_url="http://127.0.0.1:1/v1", embed_model="all-MiniLM-L6-v2", **kw)
    m.build(World(N, K, "specialist", 0.0, seed=1).view(m.needs), Budget(1))
    m.bridge.select = lambda *a, **k: {"choice": None, "error": None, "raw": None}
    return m


def _tasks(q=Q):
    return World(N, K, "specialist", 0.0, seed=1).tasks(q)


def test_bm25_scores_match_a_direct_okapi_computation():
    """The inverted index must agree with the textbook formula on the descriptions the adapter actually builds."""
    m = _built(dedup=True, retrieval="bm25")
    k1, b = 1.5, 0.75
    toks = [t.lower().split() for t in m.desc]
    L = np.array([len(t) for t in toks], float); avg = L.mean()
    for f in range(4):
        want = np.zeros(len(m.desc))
        for w in set(m.fdesc[f].lower().split()):
            df = sum(w in t for t in toks)
            if not df: continue
            idf = np.log(1 + (len(toks) - df + 0.5) / (df + 0.5))
            tf = np.array([t.count(w) for t in toks], float)
            want += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * L / avg))
        assert np.allclose(m._B[f], want, atol=1e-4)


def test_bm25_returns_k_distinct_descriptions_ranked_by_bm25():
    m = _built(dedup=True, retrieval="bm25")
    for task in _tasks():
        top = m.retrieve(task)
        assert len({m.desc[a] for a in top}) == len(top) == min(m.k, len(set(m.desc)))
        ref = list(dict.fromkeys(m.desc[a] for a in m._pool[np.argsort(-m._B[task.family][m._pool], kind="stable")]))
        assert [m.desc[a] for a in top] == ref[:len(top)]


def test_hybrid_is_reciprocal_rank_fusion_of_bm25_and_dense():
    m = _built(dedup=True, retrieval="hybrid")
    Ea, Ef = embed(m.desc), embed(m.fdesc)
    for task in _tasks(4):
        f = task.family
        fused = _rrf(m._B[f][m._pool], (Ea @ Ef[f])[m._pool])
        ref = m._pool[np.argsort(-fused, kind="stable")]
        assert list(m.retrieve(task)) == list(ref[:m.k])


def test_hybrid_differs_from_either_half_alone():
    """Fusion is only worth running if it is not just one of its inputs; assert it moves the list on some family."""
    h, bm, em = _built(dedup=True, retrieval="hybrid"), _built(dedup=True, retrieval="bm25"), _built(dedup=True, retrieval="embed")
    tasks = list(_tasks())
    assert any(list(h.retrieve(t)) != list(bm.retrieve(t)) for t in tasks)
    assert any(list(h.retrieve(t)) != list(em.retrieve(t)) for t in tasks)


def test_sota_reranks_the_fused_pool_once_per_family_at_build_time():
    """sota takes the fused top rerank_pool, reranks with the cross-encoder and keeps k -- and calls the reranker
    exactly once per family while building, never per task (the shortlist depends only on the population)."""
    calls, order = [], {}

    def fake(query, docs):                                   # deterministic: exactly reverses the fused pool
        calls.append(query); order[query] = list(docs)
        return np.arange(len(docs), dtype=np.float32)

    class _S(_Fw):
        def _rerank(self, q, d): return fake(q, d)

    m = _S(base_url="http://127.0.0.1:1/v1", embed_model="all-MiniLM-L6-v2", dedup=True, retrieval="sota", rerank_pool=25)
    m.build(World(N, K, "specialist", 0.0, seed=1).view(m.needs), Budget(1))
    assert len(calls) == len(set(calls)) == K                # one call per family, at build time
    n_before = len(calls)
    for task in _tasks(Q):
        top = m.retrieve(task)
        assert len(top) == min(m.k, len(m._pool))
        assert [m.desc[a] for a in top] == order[m.fdesc[int(task.family)]][::-1][:len(top)]   # reranker order, not fusion order
    assert len(calls) == n_before                            # retrieve() never reruns the reranker


def test_sota_pool_handed_to_the_reranker_is_the_hybrid_ranking():
    """Whatever the reranker does, the pool it scores must be the fused top-rerank_pool."""
    got = {}

    class _S(_Fw):
        def _rerank(self, q, d):
            got[q] = list(d); return np.arange(len(d), dtype=np.float32)

    m = _S(base_url="http://127.0.0.1:1/v1", embed_model="all-MiniLM-L6-v2", dedup=True, retrieval="sota", rerank_pool=25)
    m.build(World(N, K, "specialist", 0.0, seed=1).view(m.needs), Budget(1))
    h = _built(dedup=True, retrieval="hybrid")
    for f in range(4):
        fused = _rrf(h._B[f][h._pool], (h._Xa @ h._Xf[f])[h._pool])
        want = h._pool[np.argsort(-fused, kind="stable")][:25]
        assert got[m.fdesc[f]] == [m.desc[a] for a in want]


@pytest.mark.parametrize("mode", ["tfidf", "embed"])
def test_preregistered_and_embed_paths_are_untouched(mode):
    """The new branches must not perturb the two modes already reported."""
    m = _built(dedup=True, retrieval=mode)
    X = m._Xa
    for task in _tasks():
        sims = X @ m._Xf[task.family]
        assert list(m.retrieve(task)) == list(m._pool[np.argsort(-sims[m._pool], kind="stable")][:m.k])
    assert m._B is None and m._sota is None      # the new blocks are not even built for these modes


def test_embed_instruct_changes_only_the_query_side_cache_names():
    """The instruction is a QUERY-side prefix: it must change the family and sota cache names (so a probe cannot read
    a stock-instruction cache) and must NOT change the document cache name (the n descriptions are reused as-is)."""
    from rte.methods.frameworks._common import sota_cache_name
    E, R2 = "Qwen/Qwen3-Embedding-8B", "Qwen/Qwen3-Reranker-4B"
    assert sota_cache_name(E, R2, 10, 50, True) != sota_cache_name(E, R2, 10, 50, True, "some task instruction")
    a = _built(dedup=True, retrieval="embed")
    b = _built(dedup=True, retrieval="embed", embed_instruct="some task instruction")
    assert a._itag() == "" and b._itag().startswith("_i")
    assert a._slug(a.embed_model) == b._slug(b.embed_model)          # document block is shared, never re-embedded


def test_declared_retrieval_ranks_by_the_declared_claim():
    """retrieval='declared' is the cheap baseline: top-k by the declared matrix, no text retrieval at all.
    It reads the same channel liars control, which is the point of running it."""
    m = _built(dedup=True, retrieval="declared")
    D = m.view.declared
    for task in _tasks():
        f = int(task.family)
        top = m.retrieve(task)
        assert list(top) == list(m._pool[np.argsort(-D[m._pool, f], kind="stable")][:m.k])
        assert m._B is None and m._sota is None              # no BM25 block, no reranked table


def test_shuffle_permutes_the_midian_cohort_without_changing_its_members():
    """The position control: shuffle must keep the SAME shortlist (so only ordering differs from the reported arm),
    move MIDIAN's pick off position 1 for most families, and be deterministic across calls and instances."""
    import numpy as np
    from rte.budget import Budget
    from rte.world import World

    class _M(_Fw):
        needs = frozenset({"declared", "probe", "reports"})

    def built(**kw):
        m = _M(base_url="http://127.0.0.1:1/v1", retrieval="midian", r=10, **kw)
        m.build(World(N, K, "specialist", 0.0, seed=1).view(m.needs), Budget(3))
        m.bridge.select = lambda *a, **k: {"choice": None, "error": None, "raw": None}
        return m

    plain, shuf, shuf2 = built(), built(shuffle=True), built(shuffle=True)
    first_is_pick = 0
    for task in _tasks(K):
        a, b, c = plain.retrieve(task), shuf.retrieve(task), shuf2.retrieve(task)
        assert sorted(a.tolist()) == sorted(b.tolist())       # identical members, ordering only
        assert b.tolist() == c.tolist()                       # deterministic: two instances agree
        assert b.tolist() == shuf.retrieve(task).tolist()     # and stable across repeated calls
        first_is_pick += int(b[0] == a[0])
    assert first_is_pick < len(list(_tasks(K)))               # the pick is not always first any more


def test_lie_text_rewrites_the_declared_clause_from_the_declared_channel():
    """erratum 27: with lie_text the 'Declared areas:' clause must state what the agent DECLARES, not its true
    specialty -- applied to every agent, so the method never learns who lies. The prose is left untouched."""
    class _T(_Fw):
        def _texts(self, view):
            fams = list(view.families)
            return ([f"Prose about agent {a}. Declared areas: {fams[0]}." for a in range(view.n)],
                    [f"Tasks of family {f}" for f in fams], (lambda task: "t"))

    honest = _T(base_url="http://127.0.0.1:1/v1", dedup=True, retrieval="tfidf")
    honest.build(World(N, K, "specialist", 0.0, seed=1).view(honest.needs), Budget(1))
    lied = _T(base_url="http://127.0.0.1:1/v1", dedup=True, retrieval="tfidf", lie_text=True)
    lied.build(World(N, K, "specialist", 0.5, seed=1, liar_select="low_skill_first").view(lied.needs), Budget(1))
    assert all(d.startswith("Prose about agent ") for d in lied.desc)          # prose preserved
    assert all(" Declared areas: " in d and d.endswith(".") for d in lied.desc)
    assert all(d.count("Declared areas:") == 1 for d in lied.desc)             # clause replaced, not appended twice
    D, fams = lied.view.declared, list(lied.view.families)
    for a in (0, 7, 42):
        want = ", ".join(fams[f] for f in np.argsort(-D[a], kind="stable")[:3])
        assert lied.desc[a].endswith(f"Declared areas: {want}.")
    assert honest.desc != lied.desc                                            # the condition actually changes the text


def test_lie_text_changes_the_embedding_cache_identity():
    """A dense arm must never reuse honest-text vectors under lie_text. The tag is a hash of the DOCUMENTS -- the view
    does not expose beta or liar_select (adversary knowledge), and content-hashing is what the cache actually needs."""
    plain = _built(dedup=True, retrieval="embed")
    assert plain._ltag() == ""
    lied = _Fw(base_url="http://127.0.0.1:1/v1", embed_model="all-MiniLM-L6-v2", dedup=True,
               retrieval="embed", lie_text=True)
    lied.build(World(N, K, "specialist", 0.5, seed=1, liar_select="low_skill_first").view(lied.needs), Budget(1))
    assert lied._ltag().startswith("_lt") and lied._ltag() != ""
    other = _Fw(base_url="http://127.0.0.1:1/v1", embed_model="all-MiniLM-L6-v2", dedup=True,
                retrieval="embed", lie_text=True)
    other.build(World(N, K, "specialist", 0.25, seed=1).view(other.needs), Budget(1))
    assert other._ltag() != lied._ltag()          # different regime -> different text -> different cache


def test_claim_threshold_lists_every_family_above_it():
    """claim_threshold: the clause states every family the agent's DECLARED rating exceeds, so an inflating liar claims
    far more families than an honest agent -- the lie the top-3 form cannot carry (it preserves the ranking)."""
    class _T(_Fw):
        def _texts(self, view):
            fams = list(view.families)
            return ([f"Prose. Declared areas: {fams[0]}." for a in range(view.n)], [f"Tasks of family {f}" for f in fams], (lambda task: "t"))
    m = _T(base_url="http://127.0.0.1:1/v1", dedup=True, retrieval="tfidf", lie_text=True, claim_threshold=0.5)
    m.build(World(N, K, "specialist", 0.5, seed=1, liar_select="low_skill_first").view(m.needs), Budget(1))
    D, fams = m.view.declared, list(m.view.families)
    for a in (0, 7, 42):
        want = [fams[f] for f in np.argsort(-D[a], kind="stable") if D[a, f] > 0.5] or [fams[int(np.argmax(D[a]))]]
        assert m.desc[a].endswith("Declared areas: " + ", ".join(want) + ".")


def test_infrastructure_errors_fail_the_unit_instead_of_writing_a_fallback_row():
    """A worker that errors (crash, missing .so, dead endpoint) must not silently become declared-argmax picks. A few
    errors are tolerated and counted as infra_errors; past 2 % of calls fetch raises, so the unit writes no row."""
    m = _built(dedup=True, retrieval="tfidf")
    m.bridge.select = lambda *a, **k: {"choice": None, "error": "ImportError: libffi.so.8", "raw": None}
    tasks = list(_tasks(40))
    with pytest.raises(RuntimeError, match="refusing to write"):
        for t in tasks: m.fetch(t)
    assert m.stats["infra_errors"] >= 4 and m.stats["fallbacks"] == 0      # never counted as framework behaviour


def test_a_framework_naming_nobody_is_still_a_fallback_not_an_error():
    """choice=None WITHOUT an error is genuine framework behaviour (it named no candidate): it falls back and counts."""
    m = _built(dedup=True, retrieval="tfidf")
    m.bridge.select = lambda *a, **k: {"choice": None, "error": None, "raw": "I would pick the arithmetic expert"}
    for t in _tasks(10): m.fetch(t)
    assert m.stats["fallbacks"] == 10 and m.stats.get("infra_errors", 0) == 0


@pytest.mark.parametrize("err", ["ValueError: Tool 'agent_000030' not found.\nAvailable tools: transfer_to_agent",
                                 "ModelBehaviorError: Tool transfer_to_agent_00128 not found in agent triage",
                                 "ValueError: next_speaker must be provided if not terminating the conversation.",
                                 "KeyError: 'agent_005044'"])
def test_a_supervisor_invalid_action_is_a_non_pick_not_an_infrastructure_error(err):
    """Every error class that ever failed a unit (2026-09-23) was the supervisor LLM's own invalid action raised by the
    framework (ADK / OpenAI Agents: a tool named after the agent; MAF: no next speaker). That is the framework failing to
    route: a non-pick with the usual declared-argmax fallback, counted in fallback_rate, never retried, never fatal."""
    m = _built(dedup=True, retrieval="tfidf"); calls = []
    m.bridge.select = lambda *a, **k: calls.append(1) or {"choice": None, "error": err, "raw": None}
    tasks = list(_tasks(40))
    for t in tasks:
        D = m.view.declared; cand = m.retrieve(t)
        assert m.fetch(t) == int(cand[np.argmax(D[cand, t.family])]) and m._picked is False
    assert len(calls) == 40                                              # no retry: one sample, like any other non-pick
    assert m.stats["invalid_action"] == 40 and m.stats.get("infra_errors", 0) == 0
    for t in tasks: m.observe(t, 0, 0)
    assert m.stats["fallback_rate"] == 1.0


@pytest.mark.parametrize("retrieval", ["tfidf", "bm25", "declared"])
def test_parallel_prefetch_gives_exactly_the_sequential_picks(monkeypatch, retrieval):
    """RTE_FW_PARALLEL runs the (stateless) framework requests concurrently before the run loop; fetch must return the
    same agent for every task, with the same stats and ledger, and never touch the sequential bridge."""
    tasks = list(_tasks(60)); runs = {}
    for par in ("1", "8"):
        monkeypatch.setenv("RTE_FW_PARALLEL", par)
        m = _Fw(base_url="http://127.0.0.1:1/v1", dedup=True, retrieval=retrieval)
        m.build(World(N, K, "specialist", 0.5, seed=1, liar_select="low_skill_first").view(m.needs), Budget(1))
        m.prefetch(tasks)
        if par != "1": m.bridge.select = lambda *a, **k: pytest.fail("prefetched run called the sequential bridge")
        runs[par] = ([m.fetch(t) for t in tasks], dict(m.stats), m.view.ledger.snapshot())
        m.bridge.close()
    assert runs["1"] == runs["8"] and runs["1"][1]["picks"] == len(tasks)


def test_midian_cohorts_are_never_prefetched(monkeypatch):
    """The cohort shortlists learn online, so their requests must stay in order."""
    monkeypatch.setenv("RTE_FW_PARALLEL", "8")
    m = _built(retrieval="midian", r=10)
    m.prefetch(list(_tasks(5)))
    assert m._pre == {}


def test_content_keyed_cache_is_opt_in_and_keyed_on_the_texts(monkeypatch, tmp_path):
    """Non-live backends: no cache unless RTE_EMBED_CACHE_DIR is set; then one directory per exact (agent, family) text set."""
    m = _built(dedup=True, retrieval="tfidf")
    monkeypatch.delenv("RTE_EMBED_CACHE_DIR", raising=False)
    assert m._popdir(m.view) is None
    monkeypatch.setenv("RTE_EMBED_CACHE_DIR", str(tmp_path))
    d1 = m._popdir(m.view); assert d1 is not None and d1.parent == tmp_path and d1.is_dir()
    assert m._popdir(m.view) == d1                                   # same texts -> same directory
    m.desc = list(m.desc); m.desc[0] = m.desc[0] + " changed"
    assert m._popdir(m.view) != d1                                   # any text change misses
