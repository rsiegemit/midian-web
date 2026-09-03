"""RouterBench's KNN predictive router (Hu et al. 2024) on our terms: MIDIAN's probe budget, no report channel.
Predicted success of agent a on a task = mean outcome of a's k nearest probes (cosine over prompt embeddings; k = b by
default because an agent has only b probes per family). Pick = argmax over agents: n comparisons per task.
`online=True` adds every routed (prompt, agent, outcome) to the store, the learned-router analogue of flat_online."""
import numpy as np
from .base import Method
from ._learned import embed, probe_set, task_text


class KNNRouter(Method):
    name = "knn_router"
    needs = frozenset({"probe"})

    def __init__(self, online=False, k=None, **p):
        super().__init__(online=online, k=k, **p)
        self.online, self.k = online, k

    def build(self, view, budget):
        self.view, self.b = view, budget.b
        self.E, self.Y, _ = probe_set(view, budget.b)                  # (n, m, d), (n, m)
        self.cnt = np.full(view.n, self.E.shape[1]); self.k = self.k or budget.b

    def _pred(self, q):
        sims = self.E @ q                                               # (n, m)
        sims[np.arange(self.E.shape[1])[None, :] >= self.cnt[:, None]] = -np.inf     # unfilled slots (online growth)
        top = np.argpartition(-sims, self.k - 1, axis=1)[:, :self.k]
        return np.take_along_axis(self.Y, top, 1).mean(1)

    def fetch(self, task):
        self.view.ledger.compare(self.view.n)
        self._q = embed([task_text(self.view, task)])[0]
        return int(np.argmax(self._pred(self._q)))

    def observe(self, task, agent, outcome):
        if not self.online: return
        if self.cnt[agent] == self.E.shape[1]:                         # grow the store
            self.E = np.concatenate([self.E, np.zeros_like(self.E)], 1); self.Y = np.concatenate([self.Y, np.zeros_like(self.Y)], 1)
        self.E[agent, self.cnt[agent]], self.Y[agent, self.cnt[agent]] = self._q, outcome; self.cnt[agent] += 1

    def churn(self, departed, arrived):
        """Re-probe the replaced agents (len(arrived)*K*b probes) and replace their stores."""
        ids = np.asarray(arrived)
        for a in ids:
            Y, I = self.view.probe_text(a, np.arange(self.view.K), self.b)
            m = self.E.shape[1]; self.E[a] = 0; self.Y[a] = 0
            self.E[a, :self.view.K * self.b] = embed([self.view.text(f, i) for f in range(self.view.K) for i in I[f]])
            self.Y[a, :self.view.K * self.b] = Y.reshape(-1); self.cnt[a] = self.view.K * self.b
