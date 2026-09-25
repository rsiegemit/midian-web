"""RouterBench's KNN predictive router (Hu et al. 2024) on MIDIAN's probe budget, without the report channel.

Mechanism: every probe's prompt is embedded (all-MiniLM-L6-v2, or the backend's own vectors); the predicted success of
agent a on a task is the mean outcome of a's k nearest probes by cosine (k = b by default: an agent has b probes per
family); fetch takes the argmax over agents. online=True adds every routed (prompt, agent, outcome) to the store.

Ledger: build = n*K*b probes; fetch = n comparisons; observe = 0.
Churn: |arrived|*K*b probes (the replaced agents' stores are rebuilt).
Params: online=False, k=None (= b)."""
import numpy as np
from .base import Method
from ._est import scan_argmax
from ._learned import vec, probe_set, task_vec


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
        self._q = task_vec(self.view, task)
        return scan_argmax(self.view, self._pred(self._q))

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
            self.E[a] = 0; self.Y[a] = 0
            self.E[a, :self.view.K * self.b] = np.stack([vec(self.view, f, i, True) for f in range(self.view.K) for i in I[f]])
            self.Y[a, :self.view.K * self.b] = Y.reshape(-1); self.cnt[a] = self.view.K * self.b
