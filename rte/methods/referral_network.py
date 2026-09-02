"""Referral network (SPEC §6): random d-regular graph; node j believes only about its d neighbours, from the outcomes it
observed through the report channel (a lying j corrupts its own map). Fetch = greedy referral walk of `depth` hops.
Each agent is probed b times per family and each outcome observed by ONE random neighbour, so per-edge coverage is b/d
(DEVIATIONS). Messages: build n*d (2 per undirected edge); fetch 2*d per hop."""
import numpy as np
from .base import Method
from ._est import observed_reports, greedy_walk


def regular_graph(n, d, rng):
    """Exactly d-regular (d even): union of d/2 random permutations; slot s of i pairs with slot s^1 of nbr[i, s]."""
    nbr = np.empty((n, d), np.int32)
    for k in range(d // 2):
        sigma = rng.permutation(n); nbr[:, 2 * k] = sigma; nbr[sigma, 2 * k + 1] = np.arange(n)
    return nbr


class ReferralNetwork(Method):
    name = "referral_network"
    needs = frozenset({"probe", "reports", "bus"})

    def __init__(self, d=10, depth=4, **p):
        super().__init__(d=d, depth=depth, **p)
        self.d, self.depth = d + d % 2, depth

    def build(self, view, budget):
        self.view, n, d = view, view.n, self.d
        self.nbr = regular_graph(n, d, view.rng); view.bus.send_many(n * d)
        self.belief = np.zeros((n, d, view.K), np.float16)          # belief[j, slot, f]; 0 = no evidence

        def observers(ag, b):                                         # b distinct neighbour slots per agent
            self.slots = (view.rng.integers(0, d, len(ag))[:, None] + np.arange(b)) % d
            return self.nbr[ag[:, None], self.slots]
        for f in range(view.K):
            s, c = np.zeros(n * d), np.zeros(n * d)
            for ag, obs, got in observed_reports(view, f, budget.b, observers):
                flat = (obs * d + (self.slots ^ 1)).ravel()          # (observer, its slot holding ag)
                s += np.bincount(flat, got.ravel(), n * d); c += np.bincount(flat, minlength=n * d)
            self.belief[:, :, f] = (s / np.maximum(c, 1)).reshape(n, d)

    def fetch(self, task):
        f = task.family
        return greedy_walk(self.view, int(self.view.rng.integers(self.view.n)), self.depth,
                           lambda cur: self.nbr[cur], lambda cur, nb: self.belief[cur, :, f].astype(np.float32))
