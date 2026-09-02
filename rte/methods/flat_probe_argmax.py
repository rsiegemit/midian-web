"""flat_probe_argmax.py -- the key control for MIDIAN.

Probe every agent `b` times per family in build (same probe budget as MIDIAN's
level 0), estimate skill as the mean, then argmax per family at fetch. No
hierarchy, no report channel: O(n) comparisons per task by default, or O(1)
via a cached argmax (`cached=True`) recomputed only on `observe` when
`online=True`.
"""
from __future__ import annotations

import numpy as np

from .base import Method

CHUNK = 1_000_000


class FlatProbeArgmax(Method):
    name = "flat_probe_argmax"
    needs = frozenset({"probe"})

    def __init__(self, **params):
        super().__init__(**params)
        self.cached = bool(params.get("cached", False))
        self.online = bool(params.get("online", False))

    def build(self, view, budget) -> None:
        self.view = view
        n, K, b = view.n, view.K, budget.b
        self.counts = np.full((n, K), b, dtype=np.int64)
        self.est = np.zeros((n, K), dtype=np.float64)
        for f in range(K):
            for lo in range(0, n, CHUNK):
                hi = min(n, lo + CHUNK)
                agents = np.arange(lo, hi)
                outs = view.probe_many(agents, np.full(hi - lo, f), b)
                self.est[lo:hi, f] = outs.mean(axis=1)
        if self.cached:
            self._argmax = np.argmax(self.est, axis=0)          # [K]

    def fetch(self, task) -> int:
        f = task.family
        if self.cached:
            self.view.ledger.compare(1)
            return int(self._argmax[f])
        self.view.ledger.compare(self.view.n)
        return int(np.argmax(self.est[:, f]))

    def observe(self, task, agent: int, outcome: int) -> None:
        if not self.online:
            return
        f = task.family
        self.counts[agent, f] += 1
        self.est[agent, f] += (outcome - self.est[agent, f]) / self.counts[agent, f]
        if self.cached:
            self._argmax[f] = int(np.argmax(self.est[:, f]))
