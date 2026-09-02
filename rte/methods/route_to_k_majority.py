"""Route-to-many: top-k by declared skill, runner executes all k and takes
the majority outcome. `needs = {"declared"}`. `fetch` returns a LIST of k
agent ids -- per CONTRACT convention the runner executes every agent
(tasks += k) and scores majority-of-outcomes (ties -> 0). This is an
optimistic proxy for majority-of-answers (see DEVIATIONS.md). O(n) flat scan
to find the top-k: charge `compare(n)`.
"""
from __future__ import annotations

import numpy as np

from .base import Method

K_ROUTE = 3


class RouteToKMajority(Method):
    name = "route_to_k_majority"
    needs = frozenset({"declared"})

    def __init__(self, **params):
        super().__init__(**params)
        self.k = int(params.get("k", K_ROUTE))

    def build(self, view, budget) -> None:
        self.view = view
        self.k = min(self.k, view.n)

    def fetch(self, task) -> list[int]:
        f = task.family
        d = self.view.declared[:, f]
        self.view.ledger.compare(self.view.n)
        k = self.k
        top = np.argpartition(-d, k - 1)[:k]
        top = top[np.argsort(-d[top], kind="stable")]
        return [int(a) for a in top]
