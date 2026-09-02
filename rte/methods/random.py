"""Floor baseline: uniform random agent, no channel access at all.

`needs = frozenset()` -- touches nothing on the View except `.rng`/`.n`, which
are always available. This file is named `random.py` inside the `rte.methods`
package; it imports only numpy (absolute import), so it cannot shadow the
stdlib `random` module.
"""
from __future__ import annotations

from .base import Method


class RandomMethod(Method):
    name = "random"
    needs = frozenset()

    def build(self, view, budget) -> None:
        self.view = view
        self.n = view.n

    def fetch(self, task) -> int:
        # Uniform pick; no channel read, no comparison against candidates.
        return int(self.view.rng.integers(0, self.n))
