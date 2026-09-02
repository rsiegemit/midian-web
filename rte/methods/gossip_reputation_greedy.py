"""gossip_reputation_greedy — EigenTrust reputation + greedy forwarding (SPEC §6).

Budget. Identical to every other probing method: each agent is probed exactly b
times per family (n*K*b probes) and each single outcome is observed and reported
by ONE uniformly random peer j. A lying j corrupts the reports it emits, so the
reputation matrix, the trust vector and the skill estimates are all built on a
partly poisoned channel.

Reputation. R[j, a] = mean value j reported about a (all families pooled), a
scipy CSR matrix with ~K*b nnz per row. EigenTrust = row-normalize R, then power
iterate t <- C^T t with no pre-trusted seed, uniform start, dangling rows (peers
who reported nothing) redistributing their mass uniformly; <= `iters` iterations
or L1 change < 1e-12. t is rescaled by its max (monotone; argmax unaffected).

Estimates. est[a, f] = trust-weighted mean of the reports about a on family f.

T-Man overlay. Each agent keeps c neighbours that are most similar in est space
(cosine). Exact similarity is O(n^2), so this is the standard T-Man gossip
approximation: start from c random peers, then in each of `rounds` rounds build a
candidate set from own neighbours + one random neighbour's neighbours + 2 fresh
random peers, and keep the c most similar. Duplicate entries are possible and
left in place.

Fetch. Greedy forwarding from a uniformly random start on score
trust[a] * est[a, f] over the c-neighbour set: c messages, c comparisons and 1
hop per step, at most `depth` steps, return the best-scoring agent seen. No node
ever ranks more than c candidates, so fetch is O(c * depth) regardless of n.
"""
from __future__ import annotations

import numpy as np

from .base import Method

CHUNK = 1_000_000
SIM_CHUNK = 100_000


class GossipReputationGreedy(Method):
    name = "gossip_reputation_greedy"
    needs = frozenset({"probe", "reports", "bus"})

    def __init__(self, c: int = 10, depth: int = 6, iters: int = 50, rounds: int = 3, **params):
        super().__init__(c=c, depth=depth, iters=iters, rounds=rounds, **params)
        self.c, self.depth, self.iters, self.rounds = int(c), int(depth), int(iters), int(rounds)

    # ------------------------------------------------------------------ build
    def build(self, view, budget) -> None:
        try:
            from scipy.sparse import coo_matrix
        except ImportError as e:                                      # pragma: no cover
            raise ImportError("gossip_reputation_greedy needs scipy: "
                              "bash scripts/03_install_deps.sh") from e
        self.view, self.rng = view, view.rng
        n, K, b = int(view.n), int(view.K), int(budget.b)
        rows, data = [], []
        per_family = []                                                # (reporters, values) per family
        for f in range(K):
            rj, rv = [], []
            for lo in range(0, n, CHUNK):
                hi = min(n, lo + CHUNK)
                ag = np.arange(lo, hi, dtype=np.int64)
                out = view.probe_many(ag, np.full(hi - lo, f, dtype=np.int64), b)      # (m, b)
                obs = self.rng.integers(0, n, size=(hi - lo, b))                       # random peer
                obs = np.where(obs == ag[:, None], (obs + 1) % n, obs)                 # never self
                got = view.report_many(obs, ag[:, None], out)
                rj.append(obs.astype(np.int32).ravel()); rv.append(got.astype(np.int8).ravel())
            rj, rv = np.concatenate(rj), np.concatenate(rv)
            per_family.append((rj, rv)); rows.append(rj); data.append(rv.astype(np.float32))
        agents = np.repeat(np.arange(n, dtype=np.int32), b)
        cols = np.tile(agents, K)
        rows, data = np.concatenate(rows), np.concatenate(data)
        R = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()                     # sums duplicates
        C = coo_matrix((np.ones_like(data), (rows, cols)), shape=(n, n)).tocsr()
        R.data /= C.data                                                               # same pattern
        del rows, cols, data, C
        self.trust = self._eigentrust(R, n)
        self.est = np.zeros((n, K), dtype=np.float32)
        for f, (rj, rv) in enumerate(per_family):
            w = self.trust[rj]
            num = np.bincount(agents, weights=w * rv, minlength=n)
            den = np.bincount(agents, weights=w, minlength=n)
            self.est[:, f] = num / np.maximum(den, 1e-12)
        del per_family
        E = self.est / np.maximum(np.linalg.norm(self.est, axis=1, keepdims=True), 1e-12)
        self.nb = self._tman(E.astype(np.float32), self.c, self.rounds, self.rng)

    def _eigentrust(self, R, n: int) -> np.ndarray:
        rs = np.asarray(R.sum(axis=1)).ravel()
        dangling = rs <= 0
        scale = np.where(dangling, 1.0, 1.0 / np.maximum(rs, 1e-12))
        C = R.multiply(scale[:, None]).tocsr()                # row-stochastic on non-dangling rows
        Ct = C.T
        t = np.full(n, 1.0 / n)
        for _ in range(self.iters):
            nt = Ct @ t + t[dangling].sum() / n
            s = nt.sum()
            nt = nt / s if s > 0 else np.full(n, 1.0 / n)
            if np.abs(nt - t).sum() < 1e-12:
                t = nt
                break
            t = nt
        return (t / max(t.max(), 1e-30)).astype(np.float32)

    @staticmethod
    def _tman(E: np.ndarray, c: int, rounds: int, rng) -> np.ndarray:
        n = E.shape[0]
        nb = rng.integers(0, n, size=(n, c)).astype(np.int32)
        for _ in range(rounds):
            pick = nb[np.arange(n), rng.integers(0, c, size=n)]
            cand = np.concatenate([nb, nb[pick],
                                   rng.integers(0, n, size=(n, 2)).astype(np.int32)], axis=1)
            new = np.empty((n, c), dtype=np.int32)
            for lo in range(0, n, SIM_CHUNK):
                hi = min(n, lo + SIM_CHUNK)
                cc = cand[lo:hi]
                sc = np.einsum("ijk,ik->ij", E[cc], E[lo:hi])
                sc[cc == np.arange(lo, hi, dtype=np.int32)[:, None]] = -np.inf
                top = np.argpartition(-sc, c - 1, axis=1)[:, :c]
                new[lo:hi] = np.take_along_axis(cc, top, axis=1)
            nb = new
        return nb

    # ------------------------------------------------------------------ fetch
    def fetch(self, task) -> int:
        v, c, f = self.view, self.c, int(task.family)
        cur = int(self.rng.integers(0, v.n))
        best_a, best_s = cur, float(self.trust[cur] * self.est[cur, f])
        for _ in range(self.depth):
            cand = self.nb[cur]
            v.bus.send_many(c)
            v.ledger.compare(c)
            v.ledger.hop(1)
            sc = self.trust[cand] * self.est[cand, f]
            i = int(np.argmax(sc))
            if float(sc[i]) > best_s:
                best_s, best_a = float(sc[i]), int(cand[i])
            if int(cand[i]) == cur:
                break
            cur = int(cand[i])
        return best_a
