"""Shared estimation helpers. Every probing method spends the same build budget through here."""
from __future__ import annotations
import numpy as np
from .base import Method

CHUNK = 1_000_000


def probe_successes(view, b: int) -> np.ndarray:
    """Probe every agent b times per family (n*K*b probes, chunked). Returns successes float64[n, K]."""
    S = np.zeros((view.n, view.K))                 # float: exact-estimate mocks return fractions
    for f in range(view.K):
        for lo in range(0, view.n, CHUNK):
            hi = min(view.n, lo + CHUNK)
            S[lo:hi, f] = view.probe_many(np.arange(lo, hi), f, b).sum(1)
    return S


class BetaBandit(Method):
    """Thompson sampling over Beta(alpha, beta) per (agent, family); subclasses set the prior in `prior(view)`."""
    needs = frozenset({"probe"})

    def prior(self, view):                       # -> (alpha0, beta0) arrays or scalars
        return 1.0, 1.0

    def build(self, view, budget):
        self.view = view
        a0, b0 = self.prior(view)
        s = probe_successes(view, budget.b)
        self.alpha, self.beta = a0 + s, b0 + (budget.b - s)

    def fetch(self, task):
        self.view.ledger.compare(self.view.n)
        return int(np.argmax(self.view.rng.beta(self.alpha[:, task.family], self.beta[:, task.family])))

    def observe(self, task, agent, outcome):
        (self.alpha if outcome else self.beta)[agent, task.family] += 1
