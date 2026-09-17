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
        """(K, k) shortlist for retrieval='sota': fuse BM25 with the dense scores, rerank the top `rerank_pool` with the
        cross-encoder, keep k. It depends only on the population, so it is computed once and cached beside it -- the
        reranker never runs inside a routing job on a population that scripts/embed_populations.py has warmed."""
        d = self._popdir(view)
        tag = f"{self._slug(self.embed_model)}_{self._slug(self.rerank_model)}_k{self.k}p{self.rerank_pool}{'_dd' if self.dedup else ''}"
        path = (d / f"shortlist_sota_{tag}.npy") if d is not None else None
        if path is not None and path.exists():
            T = np.load(path)
            if T.shape[0] == len(self.fdesc): return T
        pool0 = self._pool if self.dedup else np.arange(view.n)
        rows = []
        for f in range(len(self.fdesc)):
            fused = _rrf(self._B[f][pool0], (self._Xa @ self._Xf[f])[pool0])
            cand = pool0[np.argsort(-fused, kind="stable")][:min(self.rerank_pool, len(pool0))]
            cand = cand[np.argsort(-self._rerank(self.fdesc[f], [self.desc[a] for a in cand]), kind="stable")]
            rows.append(np.pad(cand[:self.k], (0, max(0, self.k - len(cand))), constant_values=-1))
        T = np.stack(rows).astype(np.int64)
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
            E = embed(texts, self.embed_model, **kw)
            if path is not None:                             # atomic: concurrent jobs may race to write the same file
                tmp = path.with_suffix(f".{os.getpid()}.tmp.npy"); np.save(tmp, E); os.replace(tmp, path)
            return E

        # both blocks are cached, so a routing job on a precomputed population never loads the embedder at all
        return cached(cache, self.desc, view.n), cached(fcache, self.fdesc, len(self.fdesc), prompt_name=qp)

    def _rerank(self, query: str, docs: list[str]) -> np.ndarray:
        """Cross-encoder relevance for one family against the fused pool. Qwen3-Reranker scores as a causal LM -- the
        yes/no logit gap at the final position -- so it is driven directly; any other name loads as a CrossEncoder."""
        if self._rr is None:
            import torch
            if "Qwen3-Reranker" in self.rerank_model:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                rp = _resolve(self.rerank_model)
                tok = AutoTokenizer.from_pretrained(rp, padding_side="left")
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                mdl = AutoModelForCausalLM.from_pretrained(rp, dtype=torch.bfloat16).to(dev).eval()
                self._rr = ("qwen", tok, mdl, dev, tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no"))
            else:
                from sentence_transformers import CrossEncoder
                self._rr = ("ce", CrossEncoder(_resolve(self.rerank_model), device="cuda" if torch.cuda.is_available() else "cpu"))
        if self._rr[0] == "ce":
            return np.asarray(self._rr[1].predict([(query, d) for d in docs]), dtype=np.float32)
        import torch
        _, tok, mdl, dev, yes, no = self._rr
        ins = "Given a task family, judge whether the agent described is competent at that family."
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'},
             {"role": "user", "content": f"<Instruct>: {ins}\n<Query>: {query}\n<Document>: {d}"}],
            tokenize=False, add_generation_prompt=True) + "<think>\n\n</think>\n\n" for d in docs]
        out = np.zeros(len(docs), dtype=np.float32)
        with torch.no_grad():
            for lo in range(0, len(prompts), 16):
                bt = tok(prompts[lo:lo + 16], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(dev)
                lg = mdl(**bt).logits[:, -1, :].float()
                out[lo:lo + 16] = torch.log_softmax(torch.stack([lg[:, no], lg[:, yes]], dim=1), dim=1)[:, 1].cpu().numpy()
        return out

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
