"""EigenTrust reputation + greedy forwarding (SPEC §6). Each agent probed b times per family, each outcome reported by ONE
random peer. R[j,a] = mean report of j about a; EigenTrust = power iteration on row-normalised R, no pre-trusted seed;
est[a,f] = trust-weighted mean of reports about a. T-Man overlay: c most-similar neighbours in est space, found by gossip
rounds. Fetch = greedy walk on trust*est. Messages: build nnz(R) per power iteration + 2 per node per T-Man round;
fetch 2*c per hop."""
import numpy as np
from .base import Method
from ._est import observed_reports, greedy_walk


def eigentrust(R, iters):
    """Global trust t = C^T t, C = row-normalised R (dangling rows spread uniformly). Returns (t / max t, iterations run)."""
    n = R.shape[0]; rs = np.asarray(R.sum(1)).ravel(); dangling = rs <= 0
    Ct = R.multiply(np.where(dangling, 1, 1 / np.maximum(rs, 1e-12))[:, None]).tocsr().T
    t = np.full(n, 1 / n)
    for k in range(1, iters + 1):
        nt = Ct @ t + t[dangling].sum() / n; nt /= nt.sum()
        done = np.abs(nt - t).sum() < 1e-12; t = nt
        if done: break
    return (t / t.max()).astype(np.float32), k


def tman(E, c, rounds, rng, chunk=100_000):
    """c nearest neighbours in cosine space, approximated by T-Man gossip: candidates = own view + a neighbour's view + 2 random."""
    n = len(E); nb = rng.integers(0, n, (n, c)).astype(np.int32)
    for _ in range(rounds):
        cand = np.concatenate([nb, nb[nb[np.arange(n), rng.integers(0, c, n)]], rng.integers(0, n, (n, 2)).astype(np.int32)], 1)
        for lo in range(0, n, chunk):
            cc = cand[lo:lo + chunk]; sc = np.einsum("ijk,ik->ij", E[cc], E[lo:lo + chunk])
            sc[cc == np.arange(lo, lo + len(cc))[:, None]] = -np.inf
            nb[lo:lo + chunk] = np.take_along_axis(cc, np.argpartition(-sc, c - 1, 1)[:, :c], 1)
    return nb


class GossipReputationGreedy(Method):
    name = "gossip_reputation_greedy"
    needs = frozenset({"probe", "reports", "bus"})

    def __init__(self, c=10, depth=6, iters=50, rounds=3, **p):
        super().__init__(c=c, depth=depth, iters=iters, rounds=rounds, **p)
        self.c, self.depth, self.iters, self.rounds = c, depth, iters, rounds

    def build(self, view, budget):
        from scipy.sparse import coo_matrix
        self.view, n, K, b = view, view.n, view.K, budget.b
        peer = lambda ag, b: (ag[:, None] + view.rng.integers(1, n, (len(ag), b))) % n      # random peer, never self
        rj, rv = zip(*[(obs.ravel(), got.ravel()) for f in range(K) for _, obs, got in observed_reports(view, f, b, peer)])
        rj, rv = np.concatenate(rj).astype(np.int32), np.concatenate(rv).astype(np.float32)
        a = np.tile(np.repeat(np.arange(n, dtype=np.int32), b), K)                           # subject of each report
        R = coo_matrix((rv, (rj, a)), (n, n)).tocsr(); R.data /= coo_matrix((np.ones_like(rv), (rj, a)), (n, n)).tocsr().data
        self.trust, k = eigentrust(R, self.iters); view.bus.send_many(R.nnz * k)
        w, cell = self.trust[rj], a + n * np.repeat(np.arange(K), n * b)
        self.est = (np.bincount(cell, w * rv, n * K) / np.maximum(np.bincount(cell, w, n * K), 1e-12)).reshape(K, n).T.astype(np.float32)
        view.bus.send_many(2 * n * self.rounds)
        self.nb = tman(self.est / np.maximum(np.linalg.norm(self.est, axis=1, keepdims=True), 1e-12), self.c, self.rounds, view.rng)

    def fetch(self, task):
        f = task.family
        return greedy_walk(self.view, int(self.view.rng.integers(self.view.n)), self.depth,
                           lambda cur: self.nb[cur], lambda cur, nb: self.trust[nb] * self.est[nb, f])
