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
