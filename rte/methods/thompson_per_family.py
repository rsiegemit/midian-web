"""thompson_per_family.py -- Beta(1+s, 1+f) posteriors per (agent, family).
Same n*K*b warmup as ucb_per_family. Fetch samples the whole family column
via `view.rng.beta` (vectorized) and returns the argmax; `observe` updates
the posterior online.
"""
from __future__ import annotations

import numpy as np

from .base import Method

CHUNK = 1_000_000


class ThompsonPerFamily(Method):
    name = "thompson_per_family"
    needs = frozenset({"probe"})

    def build(self, view, budget) -> None:
        self.view = view
        n, K, b = view.n, view.K, budget.b
        self.s = np.ones((n, K), dtype=np.float64)               # Beta alpha = 1 + successes
        self.f_ = np.ones((n, K), dtype=np.float64)               # Beta beta  = 1 + failures
        for fam in range(K):
            for lo in range(0, n, CHUNK):
                hi = min(n, lo + CHUNK)
                agents = np.arange(lo, hi)
                outs = view.probe_many(agents, np.full(hi - lo, fam), b)
                succ = outs.sum(axis=1)
                self.s[lo:hi, fam] += succ
                self.f_[lo:hi, fam] += (b - succ)

    def fetch(self, task) -> int:
        f = task.family
        self.view.ledger.compare(self.view.n)
        samples = self.view.rng.beta(self.s[:, f], self.f_[:, f])
        return int(np.argmax(samples))

    def observe(self, task, agent: int, outcome: int) -> None:
        f = task.family
        if outcome:
            self.s[agent, f] += 1
        else:
            self.f_[agent, f] += 1
