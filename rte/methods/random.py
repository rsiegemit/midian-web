"""Floor: uniform random pick, no channel access. File named random.py inside rte.methods;
imports nothing but the base class, so it cannot shadow the stdlib `random` module."""
from .base import Method


class RandomMethod(Method):
    name = "random"
    needs = frozenset()

    def fetch(self, task):
        return int(self.view.rng.integers(0, self.view.n))
