"""hnswlib navigable-small-world index over probed estimates (the E7 flat rival). Trusted observer: est = mean of b probes
(same budget as flat_probe_argmax); query one-hot(f) under inner product -> nearest = argmax est[:, f] in ~log n steps.
hnswlib exposes no visit count, so per query we charge hop(ceil log2 n) and compare(ef) (DEVIATIONS)."""
import math
import numpy as np
from .base import Method
from ._est import probe_successes, CHUNK


class FlatNSWRouter(Method):
    name = "flat_nsw_router"
    needs = frozenset({"probe"})

    def __init__(self, M=16, ef=50, ef_construction=200, **p):
        super().__init__(M=M, ef=ef, ef_construction=ef_construction, **p)
        self.M, self.ef, self.efc = M, ef, ef_construction

    def build(self, view, budget):
        import hnswlib
        self.view = view
        self.est = (probe_successes(view, budget.b) / budget.b).astype(np.float32)
        self.index = hnswlib.Index("ip", view.K)
        self.index.init_index(view.n, M=self.M, ef_construction=self.efc,   # keyword: positional order is M first
                              random_seed=int(view.rng.integers(2 ** 31 - 1)))
        self.index.set_num_threads(1)
        for lo in range(0, view.n, CHUNK):
            self.index.add_items(self.est[lo:lo + CHUNK], np.arange(lo, min(view.n, lo + CHUNK)))
        self.index.set_ef(self.ef); self.hops = math.ceil(math.log2(max(view.n, 2)))

    def fetch(self, task):
        q = np.zeros((1, self.view.K), np.float32); q[0, task.family] = 1
        self.view.ledger.hop(self.hops); self.view.ledger.compare(self.ef)
        return int(self.index.knn_query(q, k=1)[0][0, 0])
