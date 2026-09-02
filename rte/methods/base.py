"""The method interface. Every rival and MIDIAN subclass this.

    needs   subset of {"declared", "probe", "reports", "bus"}; the View raises
            on any access outside it (see rte.world.View).
    build   pre-emptive step; may spend budget.probes_per_agent_family * n * K
            probes through view.probe / view.probe_many.
    fetch   returns an agent id in [0, n); charges hops/comparisons/messages
            through view.ledger.
    observe online update after the runner executed the task; default no-op.
"""
from __future__ import annotations
from typing import Any

from ..budget import Budget


class Method:
    name: str = "base"
    needs: frozenset = frozenset()

    def __init__(self, **params: Any):
        self.params = params
        self.view = None

    def build(self, view, budget: Budget) -> None:
        self.view = view

    def fetch(self, task) -> int:
        raise NotImplementedError

    def observe(self, task, agent: int, outcome: int) -> None:
        return None

    def __repr__(self):
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params.items())})"
