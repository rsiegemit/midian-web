"""RouterEval backend: a candidate pool of REAL LLMs from RouterEval (Huang et al. 2025) with their recorded per-prompt
scores, inside our World — so every method, the report channel and the liars run on real outcomes at n up to 1,000.

Agents = the m LLMs of one of RouterEval's hard-setting pools (`dataset`, `m`, `pool` in {all_strong, all_weak,
strong_to_weak}); n must equal m. Families = the K largest MMLU subjects named in the prompt text (or, on datasets
without subjects, K KMeans clusters of their RoBERTa prompt embeddings). Probes = index-seeded TRAIN prompts of the
family (a fresh one per (agent, family, k), like the llm backend's instances); tasks = TEST prompts of the family.
True skill S[a, f] = the agent's mean train score on the family (used for the oracle and liar selection only; never
shown to a method). Declarations = noisy_declared(S) (no self-descriptions exist here; both sources are the honest
control, as in replay). `text(f, inst)` returns the prompt, so knn_router / mlp_router run unchanged.
Data: $RTE_DATA/data/routereval/router_dataset/<dataset>_router_dataset.pkl (scripts: TARGETS_rte_v3.md part D)."""
from __future__ import annotations
import os, re, pickle, numpy as np
from . import noisy_declared

DATA = os.path.join(os.environ.get("RTE_DATA", os.path.expanduser("~/rte_data")), "data", "routereval", "router_dataset")
SUBJECT = re.compile(r"questions \(with answers\) about (.+?)\.")


class RouterEvalBackend:
    def __init__(self, n: int, K: int, dist: str, seed: int, rng, dataset: str = "mmlu", pool: str | None = None, **_):
        pool = pool or dist                                              # the grid's `dist` axis names the pool config
        d = pickle.load(open(f"{DATA}/{dataset}_router_dataset.pkl", "rb"))
        assert int(n) in d["hard"], f"n={n} is not a RouterEval pool size {list(d['hard'])}"
        c = d["hard"][int(n)][pool]; self.n, self.dist, self.seed = int(n), dist, int(seed)
        self.model_names = [str(x) for x in c["model"]]
        key = "train_label" if dataset == "harness_truthfulqa_mc_0" else "train_score"
        Ytr, Yte = np.asarray(c["data"][key], np.int8), np.asarray(c["data"]["test_score"], np.int8)   # (prompts, n)
        Ptr, Pte = list(d["prompt"]["train_prompt"]), list(d["prompt"]["test_prompt"])
        ftr, fte, names = self._families(dataset, Ptr, Pte, d["embedding"], int(K), seed)
        self.families = names; self.K = len(names)
        self._tr = [np.flatnonzero(ftr == k) for k in range(self.K)]     # train prompt rows per family
        self._te = [np.flatnonzero(fte == k) for k in range(self.K)]     # test prompt rows per family
        self._Ytr, self._Yte, self._Ptr, self._Pte = Ytr, Yte, Ptr, Pte
        self._Etr, self._Ete = np.asarray(d["embedding"]["train_embed"], np.float32), np.asarray(d["embedding"]["test_embed"], np.float32)
        self._S = np.stack([Ytr[r].mean(0) for r in self._tr], 1).astype(np.float32)   # (n, K)

    @staticmethod
    def _families(dataset, Ptr, Pte, emb, K, seed):
        """K largest MMLU subjects (from the prompt text) or K KMeans clusters of their embeddings; returns train/test family ids and names."""
        s_tr = [(SUBJECT.search(str(p)) or [None, None])[1] for p in Ptr]
        if dataset == "mmlu" and all(s_tr):
            s_te = [(SUBJECT.search(str(p)) or [None, "?"])[1] for p in Pte]
            top = [s for s, _ in sorted(((s, s_tr.count(s)) for s in set(s_tr)), key=lambda x: -x[1])[:K]]
            idx = {s: i for i, s in enumerate(top)}
            return (np.array([idx.get(s, -1) for s in s_tr]), np.array([idx.get(s, -1) for s in s_te]), top)
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(np.asarray(emb["train_embed"]))
        return km.labels_, km.predict(np.asarray(emb["test_embed"])), [f"cluster{k}" for k in range(K)]

    # ---- churn: a pool is fixed; replaced agents keep their real model (a swap would change the pool)
    def snapshot(self): return None
    def restore(self, snap): pass
    def redraw(self, ids, rng): pass

    def true_skill(self) -> np.ndarray: return self._S
    def declared(self, source: str = "programmatic") -> np.ndarray: return noisy_declared(self._S, self.seed)

    def text(self, f: int, inst: int, probe: bool = False) -> str:
        """Probe instances (index-seeded) address the family's TRAIN prompts, task instances its TEST prompts."""
        rows, P = (self._tr, self._Ptr) if probe else (self._te, self._Pte)
        return str(P[rows[f][inst % len(rows[f])]])

    def embedding(self, f: int, inst: int, probe: bool = False) -> np.ndarray:
        """Their RoBERTa embedding of the same prompt `text` returns (unit-normalised)."""
        rows, E = (self._tr, self._Etr) if probe else (self._te, self._Ete)
        e = E[rows[f][inst % len(rows[f])]]; return e / (np.linalg.norm(e) + 1e-9)

    def execute(self, a: int, task) -> int:
        r = self._te[task.family]; return int(self._Yte[r[task.instance % len(r)], a])

    def execute_many(self, agents, families, inst) -> np.ndarray:
        """Probes (from World._probe) carry index-seeded instances: mapped onto TRAIN rows of the family."""
        agents, families, inst = np.broadcast_arrays(np.asarray(agents), np.asarray(families), np.asarray(inst))
        out = np.empty(agents.shape, np.int8)
        for f in np.unique(families):
            m = families == f; r = self._tr[f]
            out[m] = self._Ytr[r[inst[m] % len(r)], agents[m]]
        return out

    def stats(self) -> dict:
        return {"routereval_K": self.K, "routereval_families": self.families[:8], "routereval_test_prompts": int(sum(len(r) for r in self._te))}
