"""ucb_per_family.py -- UCB1 arms (a, f). Warmup spends the same n*K*b budget
as flat_probe_argmax/MIDIAN, spread uniformly (b pulls per agent per family).
Online `observe` keeps counts/means and the per-family pull total updated.
"""
from __future__ import annotations

import math

import numpy as np

from .base import Method

CHUNK = 1_000_000


class UcbPerFamily(Method):
    name = "ucb_per_family"
    needs = frozenset({"probe"})

    def __init__(self, **params):
        super().__init__(**params)
        self.c = float(params.get("c", math.sqrt(2)))

    def build(self, view, budget) -> None:
        self.view = view
        n, K, b = view.n, view.K, budget.b
        self.counts = np.full((n, K), b, dtype=np.int64)
        self.means = np.zeros((n, K), dtype=np.float64)
        for f in range(K):
            for lo in range(0, n, CHUNK):
                hi = min(n, lo + CHUNK)
                agents = np.arange(lo, hi)
                outs = view.probe_many(agents, np.full(hi - lo, f), b)
                self.means[lo:hi, f] = outs.mean(axis=1)
        self.t = np.full(K, n * b, dtype=np.int64)              # total pulls per family so far

    def fetch(self, task) -> int:
        f = task.family
        self.view.ledger.compare(self.view.n)
        bonus = self.c * np.sqrt(np.log(max(int(self.t[f]), 2)) / self.counts[:, f])
        return int(np.argmax(self.means[:, f] + bonus))

    def observe(self, task, agent: int, outcome: int) -> None:
        f = task.family
        self.counts[agent, f] += 1
        self.means[agent, f] += (outcome - self.means[agent, f]) / self.counts[agent, f]
        self.t[f] += 1
