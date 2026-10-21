"""Declared-channel floor: route to argmax_a D[a, f].

Mechanism: collect the declarations once at build; fetch scans the family's column.

Ledger: build = n messages; fetch = n comparisons; observe = 0.
Params: none."""
import numpy as np

from ._decl import declared, scan
from .base import Method


class DeclaredArgmax(Method):
    name = "declared_argmax"
    needs = frozenset({"declared"})

    def build(self, view, budget):
        self.view = view
        self.D = declared(view)

    def fetch(self, task):
        return int(np.argmax(scan(self.view, self.D, task.family)))
