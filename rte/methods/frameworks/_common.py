"""Shared base for framework rivals: needs={"declared"} only; retrieval adapter (top-k by description
similarity, SPEC §6A "common scaling adapter"); agent-name <-> id mapping; ledger accounting
(compare(k) for the k descriptions read, hop(1) for the one supervisor call); fallback to declared
argmax among the k when the framework fails to pick (counted in stats). Two accountings per task: lenient
(the fallback pick is routed and scored) and strict (`success_strict`: a task the framework did not delegate
-- worker answered "FAILURE: ..." -- or named nothing/not a candidate scores 0)."""
from __future__ import annotations

import os
import re

import numpy as np

from .._learned import MINILM, resolve as _resolve
from ..base import Method
from ._bridge import Bridge, RTE_DATA

SUPERVISOR = "Qwen/Qwen2.5-7B-Instruct"
STRONG_EMBED = "Qwen/Qwen3-Embedding-8B"      # top of MTEB; the SOTA stack's dense half
RERANKER = "Qwen/Qwen3-Reranker-4B"          # cross-encoder over the fused pool
DENSE = ("embed", "hybrid", "sota")          # modes that need the dense block
LEXICAL = ("bm25", "hybrid", "sota")        # modes that need the BM25 block
_TOK = re.compile(r"[a-z0-9]+")


def _hash_tfidf(texts: list[str], dim: int = 4096) -> np.ndarray:
    """Pure-numpy hashed TF-IDF (no sklearn dependency); rows L2-normalized."""
    import hashlib
    X = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in _TOK.findall(t.lower()):
            X[i, int(hashlib.blake2b(w.encode(), digest_size=4).hexdigest(), 16) % dim] += 1.0
    df = (X > 0).sum(axis=0) + 1.0
    X *= np.log((len(texts) + 1.0) / df)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X


def _bm25(texts: list[str], queries: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Okapi BM25 scores, (len(queries), len(texts)); pure numpy over an inverted index, the lexical half of the hybrid.
    Only the query terms are ever touched, so the (n x vocab) matrix is never built (it is 2 GB at n = 10^5)."""
    post: dict[str, dict[int, float]] = {}
    L = np.zeros(len(texts), dtype=np.float32)
    for i, t in enumerate(texts):
        ws = _TOK.findall(t.lower()); L[i] = len(ws)
        for w in ws: post.setdefault(w, {}); post[w][i] = post[w].get(i, 0.0) + 1.0
    avg = float(L.mean()) or 1.0
    norm = k1 * (1.0 - b + b * (L / avg))                     # per-document length normalisation
    out = np.zeros((len(queries), len(texts)), dtype=np.float32)
    for j, q in enumerate(queries):
        for w in set(_TOK.findall(q.lower())):
            p = post.get(w)
            if not p: continue
            idf = np.log(1.0 + (len(texts) - len(p) + 0.5) / (len(p) + 0.5))
            d = np.fromiter(p.keys(), np.int64, len(p)); f = np.fromiter(p.values(), np.float32, len(p))
            out[j, d] += (idf * f * (k1 + 1.0) / (f + norm[d])).astype(np.float32)
    return out


def _rrf(*ranks: np.ndarray, k: float = 60.0) -> np.ndarray:
    """Reciprocal rank fusion over score rows (higher score = better); returns a fused score per column."""
    out = np.zeros_like(ranks[0], dtype=np.float32)
    for r in ranks:
        order = np.argsort(-r, kind="stable"); pos = np.empty_like(order); pos[order] = np.arange(len(order))
        out += (1.0 / (k + 1.0 + pos)).astype(np.float32)
    return out


def sota_shortlist(B, Xa, Xf, pool, desc, fdesc, rerank, k: int = 10, rerank_pool: int = 50) -> np.ndarray:
    """(K, k) SOTA shortlist: reciprocal-rank fusion of BM25 with the dense scores, the top `rerank_pool` reranked by
    the cross-encoder, the top k kept (-1 pads a family with fewer than k distinct candidates). Shared by the method
    and by scripts/embed_populations.py, which pre-warms the cache -- they must never drift apart."""
    rows = []
    for f in range(len(fdesc)):
        fused = _rrf(B[f][pool], (Xa @ Xf[f])[pool])
        cand = pool[np.argsort(-fused, kind="stable")][:min(rerank_pool, len(pool))]
        cand = cand[np.argsort(-rerank(fdesc[f], [desc[a] for a in cand]), kind="stable")]
        rows.append(np.pad(cand[:k], (0, max(0, k - len(cand))), constant_values=-1))
    return np.stack(rows).astype(np.int64)


def sota_cache_name(embed_model: str, rerank_model: str, k: int, rerank_pool: int, dedup: bool) -> str:
    """The cache filename carries every input that changes the table, so a changed setting can never read a stale one."""
    sl = lambda m: "minilm" if m == MINILM else re.sub(r"[^a-z0-9]+", "_", m.lower()).strip("_")
    return f"shortlist_sota_{sl(embed_model)}_{sl(rerank_model)}_k{k}p{rerank_pool}{'_dd' if dedup else ''}.npy"


def _endpoint(model: str) -> str:
    import json
    p = os.path.join(RTE_DATA, "endpoints.json")
    if not os.path.exists(p):
        raise RuntimeError(f"no vLLM endpoints configured at {p}")
    ep = json.load(open(p))
    urls = [u for k, u in ep.items() if k == model or k.startswith(model + "#")]   # replicas register as "<model>#<job>"
    if not urls:
        raise RuntimeError(f"model {model!r} not served; endpoints.json has {list(ep)}")
    return np.random.default_rng().choice(urls)


class FrameworkMethod(Method):
    """Subclass sets: name, env (venv name), worker (file in workers/). Optionally override `describe_task`."""
    needs = frozenset({"declared"})
    requires_llm = True
    env: str = ""
    worker: str = ""

    def __init__(self, k: int = 10, supervisor: str = SUPERVISOR, base_url: str | None = None,
                 retrieval: str = "tfidf", r: int = 10, dedup: bool = False,
                 embed_model: str = MINILM, rerank_model: str = RERANKER, rerank_pool: int = 50, **params):
        super().__init__(k=k, supervisor=supervisor, retrieval=retrieval, r=r, **params)
        self.k, self.supervisor, self._base_url = int(k), supervisor, base_url
        self.retrieval, self.r = retrieval, int(r)
        # The SOTA retrieval stack (2026-09-17, labeled variant): what a production system would actually put in front
        # of a framework, rather than the pre-registered hashed TF-IDF. retrieval="bm25" is the lexical half alone;
        # "hybrid" is reciprocal-rank fusion of BM25 with a strong dense embedder; "sota" reranks the fused top
        # `rerank_pool` with a cross-encoder. Shortlists are per FAMILY (K = 16 live), so the reranker costs
        # K * rerank_pool pairs per population -- the embedder is the only real cost, and it is cached on disk.
        self.embed_model, self.rerank_model, self.rerank_pool = embed_model, rerank_model, int(rerank_pool)
        self._rr = None
        # dedup (2026-09-14, labeled variant): rank DISTINCT description texts and offer one agent per text (the lowest
        # id). Agents sharing a prompt signature share a memoized self-description AND memoized answers, so the plain
        # top-k fills with clones once the population exceeds the number of distinct prompts (~3,900 specialist, 5
        # heavy_tail, 2 bimodal) and the framework's pick stops mattering. See CHANGES_AND_ERRATA.
        self.dedup = bool(dedup)
        # retrieval="embed" (2026-09-15, labeled variant): rank by cosine over all-MiniLM-L6-v2 embeddings of the same
        # descriptions (the encoder knn_router uses) instead of hashed TF-IDF -- the dense retriever a deployed stack
        # would put in front of a framework. Embeddings are cached per population dir (descriptions_minilm.npy).
        if retrieval in ("midian", "midian_va"):             # verified shortlist: MIDIAN-V's (or MIDIAN-VA's) leaf cohort (k = r)
            self.needs = self.needs | {"probe", "reports"}
        self.stats = {"picks": 0, "fallbacks": 0, "failures": 0, "bad_name": 0, "success_strict": 0.0, "fallback_rate": 0.0}
        self._picked, self._n, self._strict = False, 0, 0

    # ---- world accessors (llm backend provides real text; bernoulli/replay get synthesized descriptions)
    def _texts(self, view):
        try:
            from rte.backends import llm as L
            be = L.current_backend()
            if be is not None and be.n == view.n:
                return list(be.descriptions()), list(be.family_descriptions()), be.task_text
        except Exception:
            pass
        D = view.declared
        fams = list(view.families)
        desc = ["Self-rated competence: " + ", ".join(f"{fams[f]} {D[a, f]:.2f}" for f in np.argsort(-D[a])[:5])
                for a in range(view.n)]                  # no agent id in the text: id tokens collide with real words in the hash
        fdesc = [f"Tasks of family {fn}" for fn in fams]
        return desc, fdesc, (lambda task: f"A task of family {fams[task.family]} (instance {task.instance}).")

    def _popdir(self, view):
        """The population directory when the live backend is behind this view, else None (no cache: bernoulli/replay)."""
        try:
            from rte.backends import llm as L
            be = L.current_backend()
            return be.dir if be is not None and be.n == view.n else None
        except Exception:
            return None

    def _slug(self, model): return "minilm" if model == MINILM else re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")

    def _sota_table(self, view):
        """The (K, k) shortlist for retrieval='sota'. It depends only on the population, so it is computed once and
        cached beside it; scripts/embed_populations.py pre-warms it, which is what keeps routing jobs off the GPU."""
        d = self._popdir(view)
        path = (d / sota_cache_name(self.embed_model, self.rerank_model, self.k, self.rerank_pool, self.dedup)) if d is not None else None
        if path is not None and path.exists():
            T = np.load(path)
            if T.shape[0] == len(self.fdesc): return T
        pool = self._pool if self.dedup else np.arange(view.n)
        T = sota_shortlist(self._B, self._Xa, self._Xf, pool, self.desc, self.fdesc, self._rerank, self.k, self.rerank_pool)
        if path is not None:                                 # atomic: concurrent jobs may race to write the same file
            tmp = path.with_suffix(f".{os.getpid()}.tmp.npy"); np.save(tmp, T); os.replace(tmp, path)
        return T

    def _embeddings(self, view):
        """Dense rows for the agent and family descriptions; the agent block is cached next to the population, under a
        filename carrying the model so MiniLM and the strong embedder never share a cache. Asymmetric models (Qwen3)
        want the retrieval prompt on the QUERY side only -- here the family descriptions."""
        from .._learned import embed
        slug = self._slug(self.embed_model)
        d = self._popdir(view)
        cache = (d / f"descriptions_{slug}.npy") if d is not None else None
        fcache = cache.with_name(f"families_{slug}.npy") if cache is not None else None
        qp = None if self.embed_model == MINILM else "query"

        def cached(path, texts, rows, **kw):
            E = np.load(path) if path is not None and path.exists() else None
            if E is not None and E.shape[0] == rows: return E
            if self.embed_model != MINILM:                   # a strong embedder on a CPU routing node is a 100x stall,
                import torch                                 # not a slow path worth taking silently
                if not torch.cuda.is_available():
                    raise RuntimeError(f"{self.embed_model} has no cached block for {d} and no GPU is visible; "
                                       f"run: python scripts/embed_populations.py --model {self.embed_model}")
            E = embed(texts, self.embed_model, **kw)
            if path is not None:                             # atomic: concurrent jobs may race to write the same file
                tmp = path.with_suffix(f".{os.getpid()}.tmp.npy"); np.save(tmp, E); os.replace(tmp, path)
            return E

        # both blocks are cached, so a routing job on a precomputed population never loads the embedder at all
        return cached(cache, self.desc, view.n), cached(fcache, self.fdesc, len(self.fdesc), prompt_name=qp)

    def _rerank(self, query: str, docs: list[str]) -> np.ndarray:
        """Cross-encoder relevance for one family against the fused pool. Qwen3-Reranker ships modules.json and a
        1_LogitScore module, so sentence-transformers drives its yes/no scoring itself. Do NOT hand-build the chat
        prompt: the model's template owns the <Instruct>/<Query>/<Document> scaffold and silently drops content
        passed any other way, which scores every document as empty and returns one constant for the whole pool.
        Higher is better; only the induced order is used."""
        if self._rr is None:
            import torch
            from sentence_transformers import CrossEncoder
            self._rr = CrossEncoder(_resolve(self.rerank_model), device="cuda" if torch.cuda.is_available() else "cpu")
        return np.asarray(self._rr.predict([(query, d) for d in docs]), dtype=np.float32)

    def build(self, view, budget):
        super().build(view, budget)
        self.desc, self.fdesc, self._task_text = self._texts(view)
        self.names = [f"agent_{a:06d}" for a in range(view.n)]
        self._name2id = {nm: a for a, nm in enumerate(self.names)}
        if self.retrieval in DENSE:
            self._Xa, self._Xf = self._embeddings(view)
        else:
            X = _hash_tfidf(self.desc + self.fdesc)
            self._Xa, self._Xf = X[:view.n], X[view.n:]
        self._B = _bm25(self.desc, self.fdesc) if self.retrieval in LEXICAL else None
        first = {}
        for a, t in enumerate(self.desc): first.setdefault(t, a)
        self._pool = np.array(sorted(first.values()), dtype=np.int64)   # lowest id per distinct description (dedup)
        self._sota = self._sota_table(view) if self.retrieval == "sota" else None   # (K, k) -- needs _pool, so after it
        self.bridge = Bridge(self.env, self.worker)
        self.base_url = self._base_url or _endpoint(self.supervisor)
        view.ledger.message(view.n)                         # every agent sends its description to the registry once
        if self.retrieval in ("midian", "midian_va"):
            from ..midian import Midian; from ..midian_va import MidianVA
            self.mid = (MidianVA(r=self.r) if self.retrieval == "midian_va" else Midian(verify=True, cached=True, r=self.r)); self.mid.build(view, budget)

    def retrieve(self, task) -> np.ndarray:
        if self.retrieval in ("midian", "midian_va"):         # MIDIAN's pick first, then the rest of its leaf cohort
            a = self.mid.fetch(task)
            coh = self.mid.leaves[self.mid.leaf_of[a]]
            return np.concatenate([[a], coh[(coh >= 0) & (coh != a)]])
        f = int(task.family)
        if self.retrieval == "sota":
            row = self._sota[f]
            return row[row >= 0]
        pool = self._pool if self.dedup else np.arange(self.view.n)   # dedup: one representative per distinct text
        if self.retrieval == "bm25": sims = self._B[f][pool]
        else:
            sims = (self._Xa @ self._Xf[f])[pool]            # cosine: rows are L2-normalised in every dense mode
            if self.retrieval == "hybrid": sims = _rrf(self._B[f][pool], sims)
        return pool[np.argsort(-sims, kind="stable")][:min(self.k, len(pool))]

    def observe(self, task, agent, outcome):
        self._n += 1; self._strict += int(outcome) if self._picked else 0
        self.stats["success_strict"] = self._strict / self._n
        self.stats["fallback_rate"] = 1 - self.stats["picks"] / max(1, sum(self.stats[k] for k in ("picks", "fallbacks", "failures", "bad_name")))
        if self.retrieval in ("midian", "midian_va"):
            self.mid.observe(task, agent, outcome)

    def fetch(self, task) -> int:
        cand = self.retrieve(task)
        self.view.ledger.compare(len(cand)); self.view.ledger.hop(1)
        self.view.ledger.message(len(cand) + 2)             # k descriptions read + supervisor request/reply
        payload = [{"name": self.names[a], "description": self.desc[a]} for a in cand]
        resp = self.bridge.select(self._task_text(task), payload, self.supervisor, self._base_url or _endpoint(self.supervisor), params=self.params)   # re-pick per call: replicas that join mid-run get used
        choice = resp.get("choice")
        self._picked = choice in self._name2id and self._name2id[choice] in set(int(a) for a in cand)
        if self._picked:
            self.stats["picks"] += 1
            return self._name2id[choice]
        failed = str(resp.get("raw") or "").startswith("FAILURE")   # the framework answered instead of delegating
        self.stats["failures" if failed else "fallbacks" if choice is None else "bad_name"] += 1
        D = self.view.declared
        return int(cand[np.argmax(D[cand, task.family])])
