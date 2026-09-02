"""Cascade of agents ordered by ascending mean declared skill (cheap first). Each fetch walks the
order for the task's family; an agent takes the task once D[a,f] >= tau, else forwards (1 message,
1 hop, per the "state it" alternative for a single-primitive forward). If nobody takes -- everyone
forwards once -- fall back to the highest declarer rather than the last (cheapest) agent: see
DEVIATIONS.md."""
import numpy as np
from .base import Method
from ._decl import declared

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
        v.bus.send_many(pos); v.ledger.hop(pos)                  # one forward message per hop
        a = self.order[pos] if takers.size else self.order[int(np.argmax(d))]
        return int(a)
