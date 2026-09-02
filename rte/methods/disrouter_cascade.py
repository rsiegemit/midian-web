"""Cascade of self-declared agents, cheapest first.

At build, agents are ordered ascending by their mean declared skill across
families (a cost proxy: agents that declare low competence overall are
treated as "cheap" and tried first). At fetch, the cascade walks that order
for the task's family; an agent takes the task if `D[a,f] >= tau`, else it
forwards to the next agent in the order (one message, one hop each).
`hops` accumulated before a take equals the position (0-indexed) of the
first taker, matching the "hops = position of first taker" convention.

If nobody in the whole order takes the task (rare, only near tau=1), we fall
back to the agent with the highest declared[f] rather than a literal "last
agent in cost order" -- documented as a deviation in DEVIATIONS.md, since
"return the last (cheapest-cost) agent" would silently route to a low-skill
agent whenever the whole population under-declares for a family.

`needs = {"declared", "bus"}`.
"""
from __future__ import annotations

import numpy as np

from .base import Method

TAU = 0.7


class DisrouterCascade(Method):
    name = "disrouter_cascade"
    needs = frozenset({"declared", "bus"})

    def __init__(self, **params):
        super().__init__(**params)
        self.tau = float(params.get("tau", TAU))

    def build(self, view, budget) -> None:
        self.view = view
        D = view.declared
        mean_cost = D.mean(axis=1)                          # ascending -> "cheap" (low declared) first
        self.order = np.argsort(mean_cost, kind="stable").astype(np.int64)

    def fetch(self, task) -> int:
        f = task.family
        v = self.view
        order = self.order
        Df = v.declared[order, f]
        takers = np.flatnonzero(Df >= self.tau)
        if takers.size:
            pos = int(takers[0])
            if pos > 0:
                v.bus.send_many(pos)                        # pos agents forwarded before the taker
                v.ledger.hop(pos)
            return int(order[pos])
        # nobody takes: everyone forwards once, then fall back to the highest declarer
        n = len(order)
        if n > 1:
            v.bus.send_many(n - 1)
            v.ledger.hop(n - 1)
        return int(order[int(np.argmax(Df))])
