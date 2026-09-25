"""Floor: a uniform random pick with no channel access.

Mechanism: fetch draws an agent id uniformly from view.rng. The file is named random.py inside midian.methods and
imports nothing but the base class, so it cannot shadow the stdlib `random` module.

Ledger: build = 0; fetch = 0; observe = 0.
Params: none."""
from .base import Method


class RandomMethod(Method):
    name = "random"
    needs = frozenset()

    def fetch(self, task):
        return int(self.view.rng.integers(0, self.view.n))
