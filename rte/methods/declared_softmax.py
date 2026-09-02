"""Sample ~ softmax(D[:,f] / tau); numerically stable (subtract the row max)."""
import numpy as np
from .base import Method
from ._decl import declared, scan

TAU = 0.1


class DeclaredSoftmax(Method):
    name = "declared_softmax"
    needs = frozenset({"declared"})

    def __init__(self, tau=TAU, **p):
        super().__init__(tau=tau, **p)
        self.tau = tau

    def build(self, view, budget):
        self.view = view
        self.D = declared(view)

    def fetch(self, task):
        d = scan(self.view, self.D, task.family)
        w = np.exp((d - d.max()) / self.tau)
        return int(self.view.rng.choice(self.view.n, p=w / w.sum()))
