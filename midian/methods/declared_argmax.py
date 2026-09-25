"""Declared-channel floor: route to argmax_a D[a, f].

Mechanism: collect the declarations once at build; fetch scans the family's column (or, cached, reads a per-family
argmax computed at build).

Ledger: build = n messages; fetch = n comparisons (cached: 1); observe = 0.
Params: cached=False."""
import numpy as np

from ._decl import declared, scan
from ._est import lookup
from .base import Method


class DeclaredArgmax(Method):
    name = "declared_argmax"
    needs = frozenset({"declared"})

    def __init__(self, cached=False, **p):
        super().__init__(cached=cached, **p)
        self.cached = cached

    def build(self, view, budget):
        self.view = view
        self.D = declared(view)
        if self.cached:
            self.best = np.argmax(self.D, axis=0)

    def fetch(self, task):
        if self.cached:
            return lookup(self.view, self.best, task.family)
        return int(np.argmax(scan(self.view, self.D, task.family)))
