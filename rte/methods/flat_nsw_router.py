"""flat_nsw_router — hnswlib navigable-small-world index over probed estimates (SPEC §6).

The E7 flat rival: no hierarchy of cohorts, no report channel, a single global
ANN index. Trusted-observer mode, exactly like `flat_probe_argmax`: every agent
is probed b times per family (n*K*b probes, the full budget) and est[a, f] is the
plain mean. The n x K est matrix is indexed by hnswlib under the inner-product
space; a query for family f is the one-hot vector e_f, whose nearest neighbour by
inner product is the agent with the largest est[a, f]. Greedy graph search finds
that agent in O(log n) steps instead of an O(n) scan.

Cost accounting. hnswlib does not report how many nodes its search visited, so
per query we charge hop(ceil(log2 n)) -- the textbook expected greedy-search
depth in an NSW graph -- and compare(ef), the size of the candidate list the
search maintains. Both are approximations of the true search cost; see
DEVIATIONS.md.

Threads are pinned to 1 (`index.set_num_threads(1)`) so a build on a shared
login node does not oversubscribe it.
"""
from __future__ import annotations

import math

import numpy as np

from .base import Method

CHUNK = 1_000_000


class FlatNSWRouter(Method):
    name = "flat_nsw_router"
    needs = frozenset({"probe"})

    def __init__(self, M: int = 16, ef: int = 50, ef_construction: int = 200, **params):
        super().__init__(M=M, ef=ef, ef_construction=ef_construction, **params)
        self.M, self.ef, self.ef_construction = int(M), int(ef), int(ef_construction)
        self.index = None

    def build(self, view, budget) -> None:
        try:
            import hnswlib
        except ImportError as e:                                          # pragma: no cover
            raise ImportError("flat_nsw_router needs hnswlib; install it with "
                              "`bash scripts/03_install_deps.sh`") from e
        self.view = view
        rng = view.rng
        n, K, b = int(view.n), int(view.K), int(budget.b)
        self.est = np.empty((n, K), dtype=np.float32)
        for f in range(K):
            for lo in range(0, n, CHUNK):
                hi = min(n, lo + CHUNK)
                ag = np.arange(lo, hi, dtype=np.int64)
                out = view.probe_many(ag, np.full(hi - lo, f, dtype=np.int64), b)
                self.est[lo:hi, f] = out.mean(axis=1, dtype=np.float32)
        self.index = hnswlib.Index(space="ip", dim=K)
        self.index.init_index(max_elements=n, ef_construction=self.ef_construction,
                              M=self.M, random_seed=int(rng.integers(1, 2 ** 31 - 1)))
        self.index.set_num_threads(1)
        for lo in range(0, n, CHUNK):
            hi = min(n, lo + CHUNK)
            self.index.add_items(self.est[lo:hi], np.arange(lo, hi), num_threads=1)
        self.index.set_ef(max(self.ef, 1))
        self._queries = np.eye(K, dtype=np.float32)                       # one-hot per family
        self._hops = max(1, int(math.ceil(math.log2(max(n, 2)))))

    def fetch(self, task) -> int:
        f = int(task.family)
        labels, _ = self.index.knn_query(self._queries[f:f + 1], k=1, num_threads=1)
        self.view.ledger.hop(self._hops)          # approximation: expected NSW greedy depth
        self.view.ledger.compare(self.ef)         # approximation: candidate-list size
        return int(labels[0, 0])
