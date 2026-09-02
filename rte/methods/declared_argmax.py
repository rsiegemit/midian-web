"""Declared-channel floor: argmax_a D[a,f]. `needs = {"declared"}`.

Default is a flat O(n) scan per fetch, charging `compare(n)`. With
`cached=True`, the per-family argmax is precomputed once at `build` (a build
step, not charged -- the runner resets the ledger after build) and each
fetch is an O(1) lookup charged as `compare(1)`.
"""
from __future__ import annotations

import numpy as np

from .base import Method


class DeclaredArgmax(Method):
    name = "declared_argmax"
    needs = frozenset({"declared"})

    def __init__(self, **params):
        super().__init__(**params)
        self.cached = bool(params.get("cached", False))

    def build(self, view, budget) -> None:
        self.view = view
        if self.cached:
            D = view.declared           # (n, K)
            self._best = np.argmax(D, axis=0).astype(np.int64)   # (K,) per-family argmax

    def fetch(self, task) -> int:
        f = task.family
        if self.cached:
            self.view.ledger.compare(1)
            return int(self._best[f])
        D = self.view.declared
        self.view.ledger.compare(self.view.n)
        return int(np.argmax(D[:, f]))
