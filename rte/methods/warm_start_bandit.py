"""warm_start_bandit.py -- Beta prior seeded from the declared channel, with
pseudo-count n0=5 at mean D[a,f]: alpha0 = n0*D, beta0 = n0*(1-D). Build
spends the same n*K*b budget as the other probe-only methods, folding
outcomes into the posterior uniformly; `observe` keeps updating it online.
Fetch is Thompson sampling (argmax of a per-family Beta draw).
"""
from __future__ import annotations

import numpy as np

from .base import Method

CHUNK = 1_000_000


class WarmStartBandit(Method):
    name = "warm_start_bandit"
    needs = frozenset({"declared", "probe"})

    def __init__(self, **params):
        super().__init__(**params)
        self.n0 = float(params.get("n0", 5))

    def build(self, view, budget) -> None:
        self.view = view
        D = np.clip(np.asarray(view.declared, dtype=np.float64), 1e-3, 1 - 1e-3)
        self.alpha = self.n0 * D
        self.beta = self.n0 * (1.0 - D)
        n, K, b = view.n, view.K, budget.b
        for f in range(K):
            for lo in range(0, n, CHUNK):
                hi = min(n, lo + CHUNK)
                agents = np.arange(lo, hi)
                outs = view.probe_many(agents, np.full(hi - lo, f), b)
                succ = outs.sum(axis=1)
                self.alpha[lo:hi, f] += succ
                self.beta[lo:hi, f] += (b - succ)

    def fetch(self, task) -> int:
        f = task.family
        self.view.ledger.compare(self.view.n)
        samples = self.view.rng.beta(self.alpha[:, f], self.beta[:, f])
        return int(np.argmax(samples))

    def observe(self, task, agent: int, outcome: int) -> None:
        f = task.family
        if outcome:
            self.alpha[agent, f] += 1
        else:
            self.beta[agent, f] += 1
