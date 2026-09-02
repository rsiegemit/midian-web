"""AgentNet++-style two-level router: k-means on the declared-skill matrix,
one "head" agent per cluster carries the cluster's reputation.

build: partition agents into k = ceil(n/r) clusters via a vectorized,
chunked k-means over D (fixed iterations, centroid init and empty-cluster
reseeding both seeded through `view.rng`). Each cluster's head = the member
with the highest mean declared skill across families.

fetch: pick the cluster whose head has the highest declared[f] among the k
heads (`compare(k)`, `hop(1)`), then argmax declared[f] within that cluster's
members (`compare(cluster_size)`, `hop(1)`).

Scales to n=1e6: k-means never materializes an (n, k) matrix -- distance
scoring is chunked over agents, and "distance" is scored via the
<=,-2*a.c+|c|^2> expansion, so each chunk is a single BLAS matmul.
`needs = {"declared"}` (hops/comparisons are charged directly; no bus).
"""
from __future__ import annotations

import math

import numpy as np

from .base import Method

R = 10
KMEANS_ITERS = 5
CHUNK = 4096


class ClusterHeadRouter(Method):
    name = "cluster_head_router"
    needs = frozenset({"declared"})

    def __init__(self, **params):
        super().__init__(**params)
        self.r = int(params.get("r", R))
        self.kmeans_iters = int(params.get("kmeans_iters", KMEANS_ITERS))
        self.chunk = int(params.get("chunk", CHUNK))

    # ---- k-means internals -------------------------------------------------
    @staticmethod
    def _assign(D: np.ndarray, centroids: np.ndarray, chunk: int) -> np.ndarray:
        n = D.shape[0]
        c_norm = (centroids.astype(np.float64) ** 2).sum(axis=1)   # (k,)
        labels = np.empty(n, dtype=np.int64)
        ct = centroids.T
        for lo in range(0, n, chunk):
            hi = min(n, lo + chunk)
            block = D[lo:hi]                                        # (b, K)
            score = 2.0 * (block @ ct) - c_norm[None, :]             # argmax <=> argmin dist^2
            labels[lo:hi] = np.argmax(score, axis=1)
        return labels

    @staticmethod
    def _update(D: np.ndarray, labels: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        K = D.shape[1]
        sums = np.zeros((k, K), dtype=np.float64)
        for d in range(K):
            sums[:, d] = np.bincount(labels, weights=D[:, d], minlength=k)
        counts = np.bincount(labels, minlength=k).astype(np.float64)
        empty = counts == 0
        counts_safe = np.where(empty, 1.0, counts)
        centroids = sums / counts_safe[:, None]
        if empty.any():
            reinit = rng.choice(D.shape[0], size=int(empty.sum()), replace=False)
            centroids[empty] = D[reinit]
        return centroids.astype(D.dtype)

    def build(self, view, budget) -> None:
        self.view = view
        n = view.n
        D = np.ascontiguousarray(view.declared, dtype=np.float32)
        k = max(1, math.ceil(n / self.r))
        rng = view.rng
        init_idx = rng.choice(n, size=k, replace=(k > n))
        centroids = D[init_idx].copy()
        for _ in range(self.kmeans_iters):
            labels = self._assign(D, centroids, self.chunk)
            centroids = self._update(D, labels, k, rng)
        labels = self._assign(D, centroids, self.chunk)

        order = np.argsort(labels, kind="stable")
        sorted_labels = labels[order]
        uniq, start_idx, counts = np.unique(sorted_labels, return_index=True, return_counts=True)
        offsets = np.append(start_idx, sorted_labels.size)

        rowmean = D.mean(axis=1)
        rowmean_sorted = rowmean[order]
        heads = np.empty(uniq.size, dtype=np.int64)
        for i in range(uniq.size):
            lo, hi = offsets[i], offsets[i + 1]
            heads[i] = order[lo + int(np.argmax(rowmean_sorted[lo:hi]))]

        self.order = order
        self.offsets = offsets
        self.heads = heads
        self.k_eff = int(uniq.size)

    def fetch(self, task) -> int:
        f = task.family
        v = self.view
        D = v.declared
        head_d = D[self.heads, f]
        v.ledger.compare(self.heads.size)
        best_c = int(np.argmax(head_d))
        v.ledger.hop(1)
        lo, hi = self.offsets[best_c], self.offsets[best_c + 1]
        members = self.order[lo:hi]
        member_d = D[members, f]
        v.ledger.compare(member_d.size)
        v.ledger.hop(1)
        return int(members[int(np.argmax(member_d))])
