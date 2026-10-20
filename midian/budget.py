"""The build budget: b probes per (agent, family), n*K*b probes in total, the same for every probing method.

The runner warns when a method's build spends more than `total_probes(n, K)`; a method may spend less.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Budget:
    """Build budget. Every probing method gets the same n*K*b probes."""
    b: int = 3                       # probes per (agent, family)

    def total_probes(self, n: int, K: int) -> int:
        return int(n) * int(K) * int(self.b)
