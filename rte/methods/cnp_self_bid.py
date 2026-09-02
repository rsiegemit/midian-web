"""Contract Net Protocol: broadcast the task, every agent self-bids D[a,f]+noise, argmax bid wins.
2n messages per fetch (broadcast + every agent's reply), matching the CNP primitive."""
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
