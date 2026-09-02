"""Declared-channel: sample proportional to exp(D[a,f]/tau).

Models an LLM-supervisor's soft pick over the declared channel instead of a
hard argmax. `needs = {"declared"}`. Numerically stable (subtract the row
max before exponentiating). O(n) flat scan per fetch: charge `compare(n)`.
"""
from __future__ import annotations

import numpy as np

from .base import Method

TAU = 0.1


class DeclaredSoftmax(Method):
    name = "declared_softmax"
    needs = frozenset({"declared"})

    def __init__(self, **params):
        super().__init__(**params)
        self.tau = float(params.get("tau", TAU))

    def build(self, view, budget) -> None:
        self.view = view

    def fetch(self, task) -> int:
        f = task.family
        d = self.view.declared[:, f]
        self.view.ledger.compare(self.view.n)
        z = (d - d.max()) / self.tau
        w = np.exp(z)
        p = w / w.sum()
        return int(self.view.rng.choice(self.view.n, p=p))
