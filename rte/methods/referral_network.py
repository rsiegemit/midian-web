"""referral_network — decentralized, non-hierarchical rival (SPEC §6).

Structure. A random d-regular graph over the n agents. Node j keeps beliefs only
about its own d neighbours: `belief[j, slot, f]`, i.e. an O(n*d*K) sparse store,
never an n*K table. Nothing is centralized: no node sees more than d agents.

Budget. Every method gets the same n*K*b probes, so a referral network cannot
afford d independent probes per edge. Each agent a is probed exactly b times per
family (n*K*b probes total, the full budget and no more), and each single outcome
is observed by exactly ONE of a's d neighbours, chosen at random, who records it
through the report channel. Coverage of an edge is therefore b/d per family: a
node knows a fraction of its neighbours per family, which is the honest price of
decentralization at a fixed probe budget, not an implementation shortcut.
Because j records what the report channel returns, a lying j corrupts its OWN
beliefs about others (vouching for liars, zeroing the honest agents it rates
highest) — the report-channel lie of SPEC §4 acts on the searcher's map.

Graph. Exact d-regularity via a union of d/2 random permutations: for each k,
slot 2k of i points at sigma_k(i) and slot 2k+1 of sigma_k(i) points back at i.
The relation is symmetric with partner slot `s ^ 1`, so nbr[nbr[i,s], s^1] == i.
Odd d is rounded up to the next even number; self-loops / repeated edges are
possible but rare (O(d^2/n)) and are left in place.

Fetch. Start at a uniformly random node, read its d neighbours' beliefs about
family f (d messages, d comparisons, 1 hop), walk to the best-believed
neighbour, repeat up to `depth` times, return the best-believed agent seen.
"""
from __future__ import annotations

import numpy as np

from .base import Method

CHUNK = 1_000_000


class ReferralNetwork(Method):
    name = "referral_network"
    needs = frozenset({"probe", "reports", "bus"})

    def __init__(self, d: int = 10, depth: int = 4, **params):
        super().__init__(d=d, depth=depth, **params)
        self.d = int(d) + (int(d) % 2)          # paired-slot construction needs an even degree
        self.depth = int(depth)
        self.nbr = None
        self.belief = None

    # ------------------------------------------------------------------ build
    @staticmethod
    def _regular_graph(n: int, d: int, rng) -> np.ndarray:
        """Union of d/2 random permutations -> exactly d-regular, symmetric under slot ^ 1."""
        nbr = np.empty((n, d), dtype=np.int32)
        ident = np.arange(n, dtype=np.int32)
        for k in range(d // 2):
            sigma = rng.permutation(n).astype(np.int32)
            nbr[:, 2 * k] = sigma
            nbr[sigma, 2 * k + 1] = ident
        return nbr

    def build(self, view, budget) -> None:
        self.view = view
        self.rng = view.rng
        n, K, d, b = int(view.n), int(view.K), self.d, int(budget.b)
        self.nbr = self._regular_graph(n, d, self.rng)
        # float16: n*d*K*2 bytes (320 MB at n=1e6, d=10, K=16). Values are means of
        # <= b binary outcomes, so half precision is exact enough.
        self.belief = np.zeros((n, d, K), dtype=np.float16)
        sums = np.zeros(n * d, dtype=np.float64)
        cnts = np.zeros(n * d, dtype=np.float64)
        rep_slot = np.arange(b, dtype=np.int64)
        for f in range(K):
            sums[:] = 0.0
            cnts[:] = 0.0
            for lo in range(0, n, CHUNK):
                hi = min(n, lo + CHUNK)
                ag = np.arange(lo, hi, dtype=np.int64)
                out = view.probe_many(ag, np.full(hi - lo, f, dtype=np.int64), b)   # (m, b)
                # b distinct observer slots per (agent, family): a random rotation of 0..d-1
                slots = (self.rng.integers(0, d, size=hi - lo)[:, None] + rep_slot) % d
                obs = self.nbr[ag[:, None], slots].astype(np.int64)                 # (m, b) reporters
                got = view.report_many(obs, ag[:, None], out).astype(np.float64)
                flat = (obs * d + (slots ^ 1)).ravel()                              # a's slot inside obs
                sums += np.bincount(flat, weights=got.ravel(), minlength=n * d)
                cnts += np.bincount(flat, minlength=n * d)
            np.divide(sums, np.maximum(cnts, 1.0), out=sums)
            self.belief[:, :, f] = sums.reshape(n, d).astype(np.float16)
        # unobserved (agent, neighbour, family) triples read as 0.0, i.e. "no evidence"
        # ranks with "observed and failed"; documented in DEVIATIONS.md.

    # ------------------------------------------------------------------ fetch
    def fetch(self, task) -> int:
        v, d = self.view, self.d
        f = int(task.family)
        cur = int(self.rng.integers(0, v.n))
        best_a, best_v = cur, -1.0
        for _ in range(self.depth):
            v.bus.send_many(d)                 # ask the d neighbours what they believe
            v.ledger.compare(d)
            v.ledger.hop(1)
            vals = self.belief[cur, :, f]
            s = int(np.argmax(vals))
            cand = int(self.nbr[cur, s])
            if float(vals[s]) > best_v:
                best_v, best_a = float(vals[s]), cand
            cur = cand
        return best_a
