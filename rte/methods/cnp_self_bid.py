"""Contract Net Protocol, self-bid variant.

A virtual coordinator broadcasts the task to every agent (`view.bus.broadcast`,
n messages); each agent bids its declared skill plus noise; all n agents
reply with their bid (`view.bus.send_many(n)`); the coordinator picks the
argmax bid. Total messages per fetch = 2n (broadcast + replies), matching the
CONTRACT convention. `needs = {"declared", "bus"}`. O(n) flat scan over bids:
charge `compare(n)`.
"""
from __future__ import annotations

import numpy as np

from .base import Method

BID_NOISE_STD = 0.02
COORDINATOR = -1   # not a real agent id; sentinel for the broadcast source


class CnpSelfBid(Method):
    name = "cnp_self_bid"
    needs = frozenset({"declared", "bus"})

    def __init__(self, **params):
        super().__init__(**params)
        self.bid_noise_std = float(params.get("bid_noise_std", BID_NOISE_STD))

    def build(self, view, budget) -> None:
        self.view = view

    def fetch(self, task) -> int:
        f = task.family
        v = self.view
        v.bus.broadcast(COORDINATOR, task)                 # announce the task: n messages
        noise = v.rng.normal(0.0, self.bid_noise_std, size=v.n)
        bids = v.declared[:, f] + noise
        v.bus.send_many(v.n)                                # every agent replies with a bid: n messages
        v.ledger.compare(v.n)
        return int(np.argmax(bids))
