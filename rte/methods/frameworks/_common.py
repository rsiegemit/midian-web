"""Shared base for framework rivals: needs={"declared"} only; retrieval adapter (top-k by description
similarity, SPEC §6A "common scaling adapter"); agent-name <-> id mapping; ledger accounting
(compare(k) for the k descriptions read, hop(1) for the one supervisor call); fallback to declared
argmax among the k when the framework fails to pick (counted in stats)."""
from __future__ import annotations

import os
import re

import numpy as np

from ..base import Method
from ._bridge import Bridge, RTE_DATA

SUPERVISOR = "Qwen/Qwen2.5-7B-Instruct"
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

    def __init__(self, k: int = 10, supervisor: str = SUPERVISOR, base_url: str | None = None, **params):
        super().__init__(k=k, supervisor=supervisor, **params)
        self.k, self.supervisor, self._base_url = int(k), supervisor, base_url
        self.stats = {"picks": 0, "fallbacks": 0, "bad_name": 0}

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

    def build(self, view, budget):
        super().build(view, budget)
        self.desc, self.fdesc, self._task_text = self._texts(view)
        self.names = [f"agent_{a:06d}" for a in range(view.n)]
        self._name2id = {nm: a for a, nm in enumerate(self.names)}
        X = _hash_tfidf(self.desc + self.fdesc)
        self._Xa, self._Xf = X[:view.n], X[view.n:]
        self.bridge = Bridge(self.env, self.worker)
        self.base_url = self._base_url or _endpoint(self.supervisor)
        view.ledger.message(view.n)                         # every agent sends its description to the registry once

    def retrieve(self, task) -> np.ndarray:
        sims = self._Xa @ self._Xf[task.family]
        k = min(self.k, self.view.n)
        return np.argsort(-sims, kind="stable")[:k]

    def fetch(self, task) -> int:
        cand = self.retrieve(task)
        self.view.ledger.compare(len(cand)); self.view.ledger.hop(1)
        self.view.ledger.message(len(cand) + 2)             # k descriptions read + supervisor request/reply
        payload = [{"name": self.names[a], "description": self.desc[a]} for a in cand]
        resp = self.bridge.select(self._task_text(task), payload, self.supervisor, self.base_url, params=self.params)
        choice = resp.get("choice")
        if choice in self._name2id and self._name2id[choice] in set(int(a) for a in cand):
            self.stats["picks"] += 1
            return self._name2id[choice]
        self.stats["fallbacks" if choice is None else "bad_name"] += 1
        D = self.view.declared
        return int(cand[np.argmax(D[cand, task.family])])
