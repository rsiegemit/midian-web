"""sequential_halving.py -- per family, fixed-budget best-arm identification
over all n agents with total budget n*b probes for that family (so summed
over K families this is exactly budget.total_probes(n, K), same as
flat_probe_argmax/MIDIAN). Standard sequential halving: each round pulls the
per-round share of the remaining budget evenly across survivors, then keeps
the top half by empirical mean. Runs entirely at build; fetch is a cached
lookup (no online update).
"""
from __future__ import annotations

import math

import numpy as np

from .base import Method

CHUNK = 1_000_000


class SequentialHalving(Method):
    name = "sequential_halving"
    needs = frozenset({"probe"})

    def build(self, view, budget) -> None:
        self.view = view
        n, K, b = view.n, view.K, budget.b
        self.best = np.zeros(K, dtype=np.int64)
        used_total = 0
        for f in range(K):
            arm, used = self._sh_family(view, n, b, f)
            self.best[f] = arm
            used_total += used
        assert used_total <= n * K * b, "sequential_halving exceeded its build budget"

    def _sh_family(self, view, n: int, b: int, f: int):
        total_budget = n * b
        survivors = np.arange(n)
        est_sum = np.zeros(n, dtype=np.float64)
        est_cnt = np.zeros(n, dtype=np.int64)
        num_rounds = max(1, math.ceil(math.log2(max(n, 2))))
        used = 0
        for r in range(num_rounds):
            if survivors.size <= 1:
                break
            remaining_rounds = num_rounds - r
            remaining_budget = total_budget - used
            pulls = max(1, remaining_budget // (survivors.size * remaining_rounds))
            pulls = min(pulls, remaining_budget // survivors.size)
            if pulls <= 0:
                break
            sums = np.zeros(survivors.size, dtype=np.float64)
            for lo in range(0, survivors.size, CHUNK):
                hi = min(survivors.size, lo + CHUNK)
                outs = view.probe_many(survivors[lo:hi], np.full(hi - lo, f), pulls)
                sums[lo:hi] = outs.sum(axis=1)
            used += survivors.size * pulls
            est_sum[survivors] += sums
            est_cnt[survivors] += pulls
            means = est_sum[survivors] / est_cnt[survivors]
            keep = max(1, survivors.size // 2)
            order = np.argsort(-means, kind="stable")
            survivors = survivors[order[:keep]]
        cnt = est_cnt[survivors]
        means = np.divide(est_sum[survivors], cnt, out=np.zeros_like(est_sum[survivors]), where=cnt > 0)
        best = int(survivors[np.argmax(means)])
        return best, used

    def fetch(self, task) -> int:
        self.view.ledger.compare(1)
        return int(self.best[task.family])

    def observe(self, task, agent: int, outcome: int) -> None:
        return None
