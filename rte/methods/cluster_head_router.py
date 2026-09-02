"""Two-level router: k-means clusters of ~r agents on D; head = per-cluster argmax mean skill.
fetch: best cluster by its head's D[f] (compare(k), hop 1, message 2 -- ask the head), then argmax
within it (compare(cluster size), hop 1, message 2 -- ask the member).

Naive k-means with k=ceil(n/r) centroids is O(n*k) per assignment sweep: infeasible at n=1e6
(measured ~9s at n=1e5,k=1e4 -> ~90 min/sweep at n=1e6). So build clusters *within* random buckets
of `bucket` agents instead of the whole population (see DEVIATIONS.md); fetch is unaffected -- it
still searches the full global set of ~n/r clusters."""
import math
import numpy as np
from .base import Method
from ._decl import declared

R, ITERS, BUCKET, CHUNK = 10, 5, 20_000, 4096


def _assign(D, C):
    """Nearest centroid per row, chunked over D so no (n, k) matrix is materialized."""
    labels = np.empty(len(D), np.int64)
    c_norm = (C.astype(np.float64) ** 2).sum(1)
    for lo in range(0, len(D), CHUNK):
        hi = min(len(D), lo + CHUNK)
        labels[lo:hi] = np.argmax(2.0 * (D[lo:hi] @ C.T) - c_norm, axis=1)   # argmax <=> argmin dist^2
    return labels


def _kmeans(D, k, rng, iters):
    C = D[rng.choice(len(D), k, replace=k > len(D))].copy()
    for _ in range(iters + 1):
        labels = _assign(D, C)
        sums = np.stack([np.bincount(labels, weights=D[:, j], minlength=k) for j in range(D.shape[1])], axis=1)
        cnt = np.bincount(labels, minlength=k).astype(np.float64)
        empty = cnt == 0
        C = (sums / np.where(empty, 1.0, cnt)[:, None]).astype(D.dtype)
        if empty.any():
            C[empty] = D[rng.choice(len(D), int(empty.sum()), replace=False)]
    return labels


class ClusterHeadRouter(Method):
    name = "cluster_head_router"
    needs = frozenset({"declared"})

    def __init__(self, r=R, iters=ITERS, bucket=BUCKET, **p):
        super().__init__(r=r, iters=iters, bucket=bucket, **p)
        self.r, self.iters, self.bucket = r, iters, bucket

    def build(self, view, budget):
        self.view = view
        D = declared(view).astype(np.float32)
        rng, bucket = view.rng, min(view.n, self.bucket)
        perm = rng.permutation(view.n)
        orders, heads, starts, base = [], [], [], 0
        for lo in range(0, view.n, bucket):
            idx = perm[lo:lo + bucket]
            labels = _kmeans(D[idx], max(1, math.ceil(idx.size / self.r)), rng, self.iters)
            order = idx[np.argsort(labels, kind="stable")]
            _, s = np.unique(np.sort(labels), return_index=True)
            rowmean = D[order].mean(1)
            for i, lo2 in enumerate(s):
                hi2 = s[i + 1] if i + 1 < len(s) else order.size
                heads.append(order[lo2 + int(np.argmax(rowmean[lo2:hi2]))])
            orders.append(order); starts.append(s + base); base += order.size
        self.order = np.concatenate(orders)
        self.heads = np.array(heads, dtype=np.int64)
        self.offsets = np.concatenate(starts + [[self.order.size]])

    def fetch(self, task):
        v, D, f = self.view, self.view.declared, task.family
        v.ledger.compare(self.heads.size); v.ledger.hop(1); v.ledger.message(2)
        c = int(np.argmax(D[self.heads, f]))
        lo, hi = self.offsets[c], self.offsets[c + 1]
        members = self.order[lo:hi]
        v.ledger.compare(members.size); v.ledger.hop(1); v.ledger.message(2)
        return int(members[int(np.argmax(D[members, f]))])
