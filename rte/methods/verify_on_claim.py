"""verify_on_claim.py -- rank agents by declared skill D[:, f]; probe the top
unverdicted candidate k=3 times; accept if the probed mean >= D[a,f] - 0.15,
else reject and try the next (max 5 fresh verifications per fetch). Verdicts
are cached per (agent, family) so a later fetch of the same family spends no
probes on already-verdicted agents. "The most dangerous baseline": liars who
inflate D get probed and rejected, but only after burning verification budget.

Cost model: the declared-order scan is O(n) comparisons the first time a
family is ranked (D is static so the order never changes); every later
fetch of that family reuses the cached order for O(1) comparisons. This
documents the "compare(1) after the first ranking per family is cached"
option named in SPEC.md.
"""
from __future__ import annotations

import numpy as np

from .base import Method


class VerifyOnClaim(Method):
    name = "verify_on_claim"
    needs = frozenset({"declared", "probe"})

    def __init__(self, **params):
        super().__init__(**params)
        self.k = int(params.get("k", 3))
        self.max_tries = int(params.get("max_tries", 5))
        self.margin = float(params.get("margin", 0.15))

    def build(self, view, budget) -> None:
        self.view = view
        self.D = view.declared
        self._order_cache: dict[int, np.ndarray] = {}
        self._verdict: dict[tuple[int, int], tuple[bool, float]] = {}

    def _order(self, f: int) -> np.ndarray:
        order = self._order_cache.get(f)
        if order is None:
            order = np.argsort(-self.D[:, f], kind="stable")
            self._order_cache[f] = order
            self.view.ledger.compare(self.view.n)
        else:
            self.view.ledger.compare(1)
        return order

    def fetch(self, task) -> int:
        f = task.family
        order = self._order(f)
        tries = 0
        best_agent, best_mean = None, -1.0
        last = None
        for a in order:
            a = int(a)
            key = (a, f)
            verdict = self._verdict.get(key)
            if verdict is None:
                if tries >= self.max_tries:
                    break
                outs = self.view.probe_many(np.array([a]), np.array([f]), self.k)
                mean = float(outs.mean())
                accepted = mean >= self.D[a, f] - self.margin
                verdict = (accepted, mean)
                self._verdict[key] = verdict
                tries += 1
            accepted, mean = verdict
            last = a
            if accepted:
                return a
            if mean > best_mean:
                best_mean, best_agent = mean, a
        return best_agent if best_agent is not None else last

    def observe(self, task, agent: int, outcome: int) -> None:
        return None
