"""Contract Net Protocol: broadcast the task, every agent bids its declared skill plus noise, the highest bid wins.

Mechanism: collect the declarations once at build; per task a broadcast and n replies with bids D[a, f] + N(0, noise)
(noise from view.rng), then an argmax over the bids.

Ledger: build = n messages; fetch = 2n messages + n comparisons; observe = 0.
Params: noise=0.02."""
import numpy as np
from .base import Method
from ._decl import declared

NOISE = 0.02


class CnpSelfBid(Method):
    name = "cnp_self_bid"
    needs = frozenset({"declared", "bus"})

    def __init__(self, noise=NOISE, **p):
        super().__init__(noise=noise, **p)
        self.noise = noise

    def build(self, view, budget):
        self.view = view
        self.D = declared(view)

    def fetch(self, task):
        v = self.view
        v.bus.broadcast(-1, task)
        bids = self.D[:, task.family] + v.rng.normal(0, self.noise, v.n)
        v.bus.send_many(v.n)
        v.ledger.compare(v.n)
        return int(np.argmax(bids))
