from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Budget:
    """Build budget. Every probing method gets the same n*K*b probes."""
    probes_per_agent_family: int = 3

    @property
    def b(self) -> int:
        return self.probes_per_agent_family

    def total_probes(self, n: int, K: int) -> int:
        return int(n) * int(K) * int(self.probes_per_agent_family)
