"""Route-to-many: the top-k agents by declared skill all execute the task; the majority of their outcomes counts.

Mechanism: collect the declarations once at build; fetch returns the k highest declarers for the family as a list, which
the runner executes in full (k tasks charged) and scores by majority vote (ties fail).

Ledger: build = n messages; fetch = n comparisons + k executions; observe = 0.
Params: k=3."""
import numpy as np

from ._decl import declared, scan
from .base import Method

K_ROUTE = 3


class RouteToKMajority(Method):
    name = "route_to_k_majority"
    needs = frozenset({"declared"})

    def __init__(self, k=K_ROUTE, **p):
        super().__init__(k=k, **p)
        self.k = k

    def build(self, view, budget):
        self.view = view
        self.D = declared(view)
        self.k = min(self.k, view.n)

    def fetch(self, task):
        d = scan(self.view, self.D, task.family)
        top = np.argpartition(-d, self.k - 1)[:self.k]
        return [int(a) for a in top]
