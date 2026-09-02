"""Per family, fixed-budget best-arm identification (sequential halving) with budget n*b probes per family.
Runs entirely at build; fetch is a cached lookup, no online update."""
import math
import numpy as np
from .base import Method
from ._est import CHUNK, trimmed_by_reporter


def halving(view, f, budget, peers=0, delta=1 / 3):
    """Sequential halving over all agents for family f. Returns (best agent, probes used).
    peers>0: the router is NOT a trusted observer — each probe outcome reaches it only as the reports of `peers` random
    other agents (liars may corrupt them), aggregated by a per-reporter trimmed mean, exactly MIDIAN-V's channel."""
    n = view.n
    alive, tot, cnt, used = np.arange(n), np.zeros(n), np.zeros(n, np.int64), 0
    rounds = max(1, math.ceil(math.log2(n)))
    for r in range(rounds):
        pulls = max(1, (budget - used) // (alive.size * (rounds - r)))
        if alive.size == 1 or pulls * alive.size > budget - used:
            break
        for lo in range(0, alive.size, CHUNK):
            a = alive[lo:lo + CHUNK]
            out = view.probe_many(a, f, pulls)
            if peers:
                rep = (a[:, None] + view.rng.integers(1, n, (a.size, peers))) % n              # random peers, never self
                out = trimmed_by_reporter(view.report_many(rep[:, :, None], a[:, None, None], out[:, None, :]), delta, peers + 1)[:, None] * pulls
            tot[a] += out.sum(1); cnt[a] += pulls
        used += alive.size * pulls
        alive = alive[np.argsort(-tot[alive] / cnt[alive], kind="stable")[:max(1, alive.size // 2)]]
    return int(alive[np.argmax(tot[alive] / np.maximum(cnt[alive], 1))]), used


class SequentialHalving(Method):
    name = "sequential_halving"
    needs = frozenset({"probe"})

    def __init__(self, peer_reported=False, r=10, delta=1 / 3, **p):
        super().__init__(peer_reported=peer_reported, r=r, delta=delta, **p)
        self.peers, self.delta = (r - 1) if peer_reported else 0, delta
        if peer_reported:
            self.needs = self.needs | {"reports"}                       # no trusted observer: MIDIAN's channel

    def build(self, view, budget):
        self.view = view
        res = [halving(view, f, view.n * budget.b, self.peers, self.delta) for f in range(view.K)]
        self.best = np.array([a for a, _ in res])
        assert sum(u for _, u in res) <= budget.total_probes(view.n, view.K)

    def fetch(self, task):
        self.view.ledger.compare(1)
        return int(self.best[task.family])
