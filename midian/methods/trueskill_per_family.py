"""Pairwise per-family TrueSkill ratings from head-to-head probes.

Mechanism: per family, n*b/2 random agent pairs (self-pairs dropped) are probed once each; the two outcomes give a
win / loss / draw update to the pair's TrueSkill ratings; fetch takes argmax mu. The update loop is pure Python
(`trueskill.rate_1vs1` has no vectorised form), so n >= 10^5 raises NotImplementedError (see
docs/archive/DEVIATIONS.md).

Ledger: build <= n*K*b probes; fetch = n comparisons; observe = 0.
Params: none."""
from __future__ import annotations

import numpy as np

from ._est import scan_argmax
from .base import Method

_MAX_N = 100_000


class TrueSkillPerFamily(Method):
    name = "trueskill_per_family"
    needs = frozenset({"probe"})

    def build(self, view, budget) -> None:
        try:
            import trueskill
        except ImportError as e:
            raise ImportError(
                "trueskill_per_family needs the `trueskill` package: pip install trueskill"
            ) from e
        n, K, b = view.n, view.K, budget.b
        if n >= _MAX_N:
            raise NotImplementedError(
                f"trueskill_per_family is an O(pairs) pure-Python rating loop "
                f"({n}*{K}*{b // 2} pair updates); not implemented at n>={_MAX_N}. "
                "See docs/archive/DEVIATIONS.md.")
        self.view = view
        env = trueskill.TrueSkill()
        self.env = env
        ratings = [[env.create_rating() for _ in range(K)] for _ in range(n)]
        n_pairs = (n * b) // 2
        for f in range(K):
            a1 = view.rng.integers(0, n, size=n_pairs)
            a2 = view.rng.integers(0, n, size=n_pairs)
            keep = a1 != a2
            a1, a2 = a1[keep], a2[keep]
            o1 = view.probe_many(a1, np.full(a1.size, f), 1)[:, 0]
            o2 = view.probe_many(a2, np.full(a2.size, f), 1)[:, 0]
            for i in range(a1.size):
                x, y = int(a1[i]), int(a2[i])
                r1, r2 = ratings[x][f], ratings[y][f]
                if o1[i] == o2[i]:
                    r1, r2 = env.rate_1vs1(r1, r2, drawn=True)
                elif o1[i] > o2[i]:
                    r1, r2 = env.rate_1vs1(r1, r2)
                else:
                    r2, r1 = env.rate_1vs1(r2, r1)
                ratings[x][f], ratings[y][f] = r1, r2
        self.mu = np.array([[ratings[a][f].mu for f in range(K)] for a in range(n)])

    def fetch(self, task) -> int:
        return scan_argmax(self.view, self.mu[:, task.family])

    def observe(self, task, agent: int, outcome: int) -> None:
        return None
