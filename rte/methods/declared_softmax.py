"""Declared-channel soft pick: sample an agent with probability proportional to exp(D[a, f] / tau).

Mechanism: collect the declarations once at build; fetch scans the family's column and samples from its softmax
(computed after subtracting the column max, for numerical stability) with view.rng.

Ledger: build = n messages; fetch = n comparisons; observe = 0.
Params: tau=0.1."""
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
