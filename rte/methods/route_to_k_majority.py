"""Route-to-many: top-k by declared skill; the runner executes all k and majority-votes
(fetch returns a list of ids, per CONTRACT's route-to-many convention)."""
import numpy as np
from .base import Method
from ._decl import declared, scan

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
