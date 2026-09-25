"""Declared-skill cascade: agents in ascending mean-declared order; the first whose claim clears tau takes the task.

Mechanism: at build, order agents by their mean declared skill (cheap first). Each fetch walks that order for the task's
family; an agent takes the task once D[a, f] >= tau, else forwards it (one message, one hop). If nobody takes it, the
highest declarer does rather than the last agent (see docs/archive/DEVIATIONS.md).

Ledger: build = n messages; fetch = p messages + p hops (p = position of the taker, n - 1 if none); observe = 0.
Params: tau=0.7."""
import numpy as np

from ._decl import declared
from .base import Method

TAU = 0.7


class DisrouterCascade(Method):
    name = "disrouter_cascade"
    needs = frozenset({"declared", "bus"})

    def __init__(self, tau=TAU, **p):
        super().__init__(tau=tau, **p)
        self.tau = tau

    def build(self, view, budget):
        self.view = view
        self.D = declared(view)
        self.order = np.argsort(self.D.mean(1), kind="stable")

    def fetch(self, task):
        v = self.view
        d = self.D[self.order, task.family]
        takers = np.flatnonzero(d >= self.tau)
        pos = int(takers[0]) if takers.size else d.size - 1
        # one forward message per hop
        v.bus.send_many(pos)
        v.ledger.hop(pos)
        a = self.order[pos] if takers.size else self.order[int(np.argmax(d))]
        return int(a)
