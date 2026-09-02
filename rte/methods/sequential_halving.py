"""Per family, fixed-budget best-arm identification (sequential halving) with budget n*b probes per family.
Runs entirely at build; fetch is a cached lookup, no online update."""
import math
import numpy as np
from .base import Method
from ._est import CHUNK


def halving(view, f, budget):
    """Sequential halving over all agents for family f. Returns (best agent, probes used)."""
    n = view.n
    alive, tot, cnt, used = np.arange(n), np.zeros(n), np.zeros(n, np.int64), 0
    rounds = max(1, math.ceil(math.log2(n)))
    for r in range(rounds):
        pulls = max(1, (budget - used) // (alive.size * (rounds - r)))
        if alive.size == 1 or pulls * alive.size > budget - used:
            break
        for lo in range(0, alive.size, CHUNK):
            a = alive[lo:lo + CHUNK]
            tot[a] += view.probe_many(a, f, pulls).sum(1); cnt[a] += pulls
        used += alive.size * pulls
        alive = alive[np.argsort(-tot[alive] / cnt[alive], kind="stable")[:max(1, alive.size // 2)]]
    return int(alive[np.argmax(tot[alive] / np.maximum(cnt[alive], 1))]), used


class SequentialHalving(Method):
    name = "sequential_halving"
    needs = frozenset({"probe"})

    def build(self, view, budget):
        self.view = view
        res = [halving(view, f, view.n * budget.b) for f in range(view.K)]
        self.best = np.array([a for a, _ in res])
        assert sum(u for _, u in res) <= budget.total_probes(view.n, view.K)

    def fetch(self, task):
        self.view.ledger.compare(1)
        return int(self.best[task.family])
