"""Shared base for framework rivals: needs={"declared"} only; retrieval adapter (top-k by description
similarity, SPEC §6A "common scaling adapter"); agent-name <-> id mapping; ledger accounting
(compare(k) for the k descriptions read, hop(1) for the one supervisor call); fallback to declared
argmax among the k when the framework fails to pick (counted in stats). Two accountings per task: lenient
(the fallback pick is routed and scored) and strict (`success_strict`: a task the framework did not delegate
-- worker answered "FAILURE: ..." -- or named nothing/not a candidate scores 0)."""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from ...stable_hash import stable_seed_32
from .._learned import MINILM, resolve as _resolve
from ..base import Method
from ._bridge import Bridge, RTE_DATA

SUPERVISOR = "Qwen/Qwen2.5-7B-Instruct"
STRONG_EMBED = "Qwen/Qwen3-Embedding-8B"      # top of MTEB; the SOTA stack's dense half
RERANKER = "Qwen/Qwen3-Reranker-4B"          # cross-encoder over the fused pool
DENSE = ("embed", "hybrid", "sota")          # modes that need the dense block
DECLARED = "declared"                        # top-k by the declared claim; no text retrieval at all
LEXICAL = ("bm25", "hybrid", "sota")        # modes that need the BM25 block
# Framework errors that are the supervisor LLM's own invalid action, not infrastructure (every class that has failed a
# unit, 2026-09-23): ADK / OpenAI Agents -- a tool named after the agent instead of the routing tool; MAF -- no next
# speaker. Anything else stays an infrastructure error (erratum 28).
INVALID_ACTION = re.compile(r"Tool '?[\w.-]+'? not found|ModelBehaviorError|next_speaker must be provided|KeyError: '?agent_\d+'?")   # MAF: orchestrator names a non-candidate
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


def sota_cache_name(embed_model: str, rerank_model: str, k: int, rerank_pool: int, dedup: bool, instruct: str = "") -> str:
    """The cache filename carries every input that changes the table, so a changed setting can never read a stale one."""
    import hashlib
    sl = lambda m: "minilm" if m == MINILM else re.sub(r"[^a-z0-9]+", "_", m.lower()).strip("_")
    it = "" if not instruct else "_i" + hashlib.blake2b(instruct.encode(), digest_size=4).hexdigest()
    return f"shortlist_sota_{sl(embed_model)}_{sl(rerank_model)}_k{k}p{rerank_pool}{'_dd' if dedup else ''}{it}.npy"


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
                 embed_model: str = MINILM, rerank_model: str = RERANKER, rerank_pool: int = 50,
                 embed_instruct: str = "", shuffle: bool = False, lie_text: bool = False,
                 claim_threshold: float = 0.0, **params):
        super().__init__(k=k, supervisor=supervisor, retrieval=retrieval, r=r, **params)
        self.k, self.supervisor, self._base_url = int(k), supervisor, base_url
        self.retrieval, self.r = retrieval, int(r)
        # The SOTA retrieval stack (2026-09-17, labeled variant): what a production system would actually put in front
        # of a framework, rather than the pre-registered hashed TF-IDF. retrieval="bm25" is the lexical half alone;
        # "hybrid" is reciprocal-rank fusion of BM25 with a strong dense embedder; "sota" reranks the fused top
        # `rerank_pool` with a cross-encoder. Shortlists are per FAMILY (K = 16 live), so the reranker costs
        # K * rerank_pool pairs per population -- the embedder is the only real cost, and it is cached on disk.
        self.embed_model, self.rerank_model, self.rerank_pool = embed_model, rerank_model, int(rerank_pool)
        # embed_instruct: Qwen3-Embedding is instruction-tuned and asymmetric -- the QUERY side carries a task
        # description, the document side carries none. Empty means the model's stock web-search instruction. Only the
        # family (query) block depends on it, so a probe re-embeds 16 texts per population, never the n descriptions.
        self.embed_instruct = str(embed_instruct or "")
        # shuffle (position control): the midian modes hand the framework [MIDIAN's pick] + its leaf cohort, so the
        # best agent sits in position 1 and a supervisor with position bias gets it for free. Shuffling permutes that
        # list deterministically per cohort, which separates "the cohort is better material" from "the pick was first".
        self.shuffle = bool(shuffle)
        # lie_text (erratum 27): the benchmark's lie inflates the declared MATRIX but leaves the self-description TEXT
        # stating the agent's TRUE specialty, so a liar's text and its numbers disagree and every text retriever is
        # shielded from the attack. With lie_text the "Declared areas:" clause -- the structured claim carried IN the
        # text -- is rederived from view.declared for EVERY agent, liar or not, so the method needs no knowledge of
        # who lies. Honest agents barely move (their declared is true skill plus 0.05 noise); liars now claim in text
        # what they claim in the matrix. The LLM prose is untouched and still describes the real specialty, so this is
        # a PARTIAL text lie and must be reported as one.
        self.lie_text = bool(lie_text)
        # claim_threshold > 0: the clause lists EVERY family the agent rates above the threshold instead of its top 3.
        # The top-3 form barely carries the benchmark's lie -- `inflate` adds +0.4 to every family, which preserves an
        # agent's ranking, so only 14% of cartel liars' top-3 changes. With a threshold of 0.7 a cartel liar claims
        # ~15 of 16 families against ~3 honestly: the "I can do everything" description a lying agent actually writes.
        self.claim_threshold = float(claim_threshold)
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
        self.stats = {"picks": 0, "fallbacks": 0, "failures": 0, "bad_name": 0, "invalid_action": 0, "success_strict": 0.0, "fallback_rate": 0.0}
        self._picked, self._n, self._strict, self._calls = False, 0, 0, 0

    # ---- world accessors (llm backend provides real text; bernoulli/replay get synthesized descriptions)
    def _relabel(self, desc, view, top=3):
        """Rewrite the trailing 'Declared areas: ...' clause from the DECLARED channel (see lie_text)."""
        fams = list(view.families); D = view.declared
        out = []
        for a, d in enumerate(desc):
            if self.claim_threshold > 0:
                fs = [f for f in np.argsort(-D[a], kind="stable") if D[a, f] > self.claim_threshold] or [int(np.argmax(D[a]))]
            else:
                fs = np.argsort(-D[a], kind="stable")[:top]
            claim = ", ".join(fams[f] for f in fs)
            out.append(re.sub(r"\s*Declared areas: .*$", "", d).rstrip() + f" Declared areas: {claim}.")
        return out

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
        """Where this population's embedding / shortlist caches live. Live backend: the population directory. Any other
        backend (RouterEval, whose descriptions are rendered at run time), only when RTE_EMBED_CACHE_DIR is set: a
        directory under it keyed by a hash of the exact agent
        and family TEXTS, so a GPU pre-warm (scripts/embed_routereval.py) and the CPU routing units meet on the same files,
        and any text change misses. Needs self.desc / self.fdesc, i.e. call after _texts."""
        try:
            from rte.backends import llm as L
            be = L.current_backend()
            if be is not None and be.n == view.n: return be.dir
        except Exception:
            pass
        root = os.environ.get("RTE_EMBED_CACHE_DIR")        # opt-in: unset (tests, every other grid) -> no cache, as before
        if not root or getattr(self, "desc", None) is None: return None
        import hashlib
        h = hashlib.blake2b(digest_size=12)
        for t in list(self.desc) + ["\x00"] + list(self.fdesc): h.update(t.encode()); h.update(b"\x00")
        d = Path(root) / h.hexdigest(); d.mkdir(parents=True, exist_ok=True)
        return d

    def _ltag(self):
        """Cache-name suffix when lie_text is on. Keyed on a hash of the DOCUMENTS, not on beta/liar_select: the view
        deliberately does not expose those (they are adversary knowledge), and content-hashing is what the cache
        actually needs -- two regimes that produce the same text should share vectors, and any text change must miss."""
        if not self.lie_text: return ""
        import hashlib
        return "_lt" + hashlib.blake2b("\x00".join(self.desc).encode(), digest_size=4).hexdigest()

    def _itag(self):
        import hashlib
        return "" if not self.embed_instruct else "_i" + hashlib.blake2b(self.embed_instruct.encode(), digest_size=4).hexdigest()

    def _slug(self, model): return "minilm" if model == MINILM else re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")

    def _sota_table(self, view):
        """The (K, k) shortlist for retrieval='sota'. It depends only on the population, so it is computed once and
        cached beside it; scripts/embed_populations.py pre-warms it, which is what keeps routing jobs off the GPU."""
        d = self._popdir(view)
        name = sota_cache_name(self.embed_model, self.rerank_model, self.k, self.rerank_pool, self.dedup,
                               self.embed_instruct + self._ltag())
        path = (d / name) if d is not None else None
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
        slug = self._slug(self.embed_model) + self._ltag()   # lie_text rewrites the documents: never reuse honest vectors
        d = self._popdir(view)
        cache = (d / f"descriptions_{slug}.npy") if d is not None else None
        fcache = cache.with_name(f"families_{slug}{self._itag()}.npy") if cache is not None else None
        qp = None if self.embed_model == MINILM else "query"
        qprompt = f"Instruct: {self.embed_instruct}\nQuery:" if self.embed_instruct else None

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
        return cached(cache, self.desc, view.n), cached(fcache, self.fdesc, len(self.fdesc), prompt_name=qp, prompt=qprompt)

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
        self._index(view)
        self.bridge = Bridge(self.env, self.worker); self._pre = {}
        self.base_url = self._base_url or _endpoint(self.supervisor)
        view.ledger.message(view.n)                         # every agent sends its description to the registry once
        if self.retrieval in ("midian", "midian_va"):
            from ..midian import Midian; from ..midian_va import MidianVA
            self.mid = (MidianVA(r=self.r) if self.retrieval == "midian_va" else Midian(verify=True, cached=True, r=self.r)); self.mid.build(view, budget)

    def _index(self, view):
        """Texts, retrieval vectors and shortlist tables: everything build() derives from the population alone, with no
        supervisor. scripts/embed_routereval.py calls exactly this on a GPU to pre-warm the RTE_EMBED_CACHE_DIR files."""
        self.desc, self.fdesc, self._task_text = self._texts(view)
        if self.lie_text: self.desc = self._relabel(self.desc, view)
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

    def retrieve(self, task) -> np.ndarray:
        if self.retrieval in ("midian", "midian_va"):         # MIDIAN's pick first, then the rest of its leaf cohort
            a = self.mid.fetch(task)
            coh = self.mid.leaves[self.mid.leaf_of[a]]
            out = np.concatenate([[a], coh[(coh >= 0) & (coh != a)]])
            if self.shuffle:                                 # position control: same members, pick no longer first
                out = out[np.random.default_rng(stable_seed_32(int(a), "fw_shuffle", int(task.family))).permutation(len(out))]
            return out
        f = int(task.family)
        if self.retrieval == "sota":
            row = self._sota[f]
            return row[row >= 0]
        pool = self._pool if self.dedup else np.arange(self.view.n)   # dedup: one representative per distinct text
        if self.retrieval == "declared":                     # top-k by the DECLARED claim: the cheap baseline a
            sims = self.view.declared[pool, f]               # practitioner reaches for before any retriever
        elif self.retrieval == "bm25": sims = self._B[f][pool]
        else:
            sims = (self._Xa @ self._Xf[f])[pool]            # cosine: rows are L2-normalised in every dense mode
            if self.retrieval == "hybrid": sims = _rrf(self._B[f][pool], sims)
        return pool[np.argsort(-sims, kind="stable")][:min(self.k, len(pool))]

    def observe(self, task, agent, outcome):
        self._n += 1; self._strict += int(outcome) if self._picked else 0
        self.stats["success_strict"] = self._strict / self._n
        self.stats["fallback_rate"] = 1 - self.stats["picks"] / max(1, sum(self.stats[k] for k in ("picks", "fallbacks", "failures", "bad_name", "invalid_action")))
        if self.retrieval in ("midian", "midian_va"):
            self.mid.observe(task, agent, outcome)

    def prefetch(self, stream):
        """Ask the framework about every task at once, PARALLEL requests in flight, before the run loop consumes them in
        order. Only for stateless shortlists: the workers build a fresh team per request at temperature 0 and keep no
        memory, so a pick cannot depend on which requests came before it -- fetch() sees exactly the response a
        sequential run would. The MIDIAN cohorts learn online (retrieve depends on earlier observes) and stay sequential."""
        n = int(os.environ.get("RTE_FW_PARALLEL", "1"))
        if n <= 1 or self.retrieval in ("midian", "midian_va"): return
        import queue
        from concurrent.futures import ThreadPoolExecutor
        free = queue.Queue()
        for _ in range(n): free.put(Bridge(self.env, self.worker))

        def one(task):
            cand = self.retrieve(task); b = free.get()
            try:
                payload = [{"name": self.names[a], "description": self.desc[a]} for a in cand]
                return b.select(self._task_text(task), payload, self.supervisor, self._base_url or _endpoint(self.supervisor), params=self.params)
            finally: free.put(b)
        with ThreadPoolExecutor(max_workers=n) as ex:
            self._pre = dict(zip(map(id, stream), ex.map(one, stream)))
        while not free.empty(): free.get().close()

    def fetch(self, task) -> int:
        cand = self.retrieve(task)
        self.view.ledger.compare(len(cand)); self.view.ledger.hop(1)
        self.view.ledger.message(len(cand) + 2)             # k descriptions read + supervisor request/reply
        payload = [{"name": self.names[a], "description": self.desc[a]} for a in cand]
        ask = lambda: self.bridge.select(self._task_text(task), payload, self.supervisor, self._base_url or _endpoint(self.supervisor), params=self.params)   # re-pick per call: replicas that join mid-run get used
        resp = self._pre.pop(id(task), None) or ask()      # prefetched response, if prefetch() ran
        if resp.get("error") and not INVALID_ACTION.search(resp["error"]): resp = ask()   # one retry: the bridge restarts a dead worker
        self._calls += 1
        if resp.get("error") and INVALID_ACTION.search(resp["error"]):
            # the SUPERVISOR's invalid action, raised by the framework itself (a tool named after the agent instead of the
            # routing tool, no next speaker): the framework did not delegate -- a non-pick like an unparseable reply, with
            # the same declared-argmax fallback inside the shortlist, counted in fallback_rate. Not retried (no other
            # non-pick gets a second sample), and never an infrastructure error.
            self.stats["invalid_action"] += 1; self._picked = False
            D = self.view.declared
            return int(cand[np.argmax(D[cand, task.family])])
        if resp.get("error"):
            # INFRASTRUCTURE, not framework behaviour: a crashed worker, a missing shared library, a dead endpoint. These
            # used to fall through to declared argmax and write a normal-looking row that measured declared argmax under
            # the framework's name -- ~4,500 rows on 2026-09-22 (CrewAI / ADK venvs with deleted .so files). A few are
            # tolerated and counted apart from real fallbacks; past 2 % of calls the unit FAILS and writes no row.
            self.stats["infra_errors"] = self.stats.get("infra_errors", 0) + 1
            if self.stats["infra_errors"] > max(3, 0.02 * self._calls):
                raise RuntimeError(f"{self.name}: {self.stats['infra_errors']} of {self._calls} supervisor calls failed "
                                   f"({resp['error'][:160]}) -- refusing to write a fallback-contaminated row")
            D = self.view.declared; self._picked = False
            return int(cand[np.argmax(D[cand, task.family])])
        choice = resp.get("choice")
        self._picked = choice in self._name2id and self._name2id[choice] in set(int(a) for a in cand)
        if self._picked:
            self.stats["picks"] += 1
            return self._name2id[choice]
        failed = str(resp.get("raw") or "").startswith("FAILURE")   # the framework answered instead of delegating
        self.stats["failures" if failed else "fallbacks" if choice is None else "bad_name"] += 1
        D = self.view.declared
        return int(cand[np.argmax(D[cand, task.family])])
