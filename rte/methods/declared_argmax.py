"""Declared-channel floor: argmax_a D[a,f]. cached=True precomputes the per-family argmax
at build (O(1) fetch); default is a flat O(n) scan per fetch."""
import numpy as np
from .base import Method
from ._decl import declared, scan


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
            self.view.ledger.compare(1)
            return int(self.best[task.family])
        return int(np.argmax(scan(self.view, self.D, task.family)))
