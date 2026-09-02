"""Per-run cost counters. Exactly one increment site per counter: the named
method below. Nothing else in the package touches the counts."""
from __future__ import annotations

COUNTERS = ("probes", "reports", "messages", "hops", "comparisons", "tasks")


class Ledger:
    __slots__ = tuple(COUNTERS)

    def __init__(self):
        self.reset()

    def reset(self):
        for c in COUNTERS:
            setattr(self, c, 0)

    # one increment site each ------------------------------------------------
    def probe(self, k: int = 1):      self.probes += int(k)
    def report(self, k: int = 1):     self.reports += int(k)
    def message(self, k: int = 1):    self.messages += int(k)
    def hop(self, k: int = 1):        self.hops += int(k)
    def compare(self, k: int = 1):    self.comparisons += int(k)
    def task(self, k: int = 1):       self.tasks += int(k)

    def snapshot(self) -> dict:
        return {c: getattr(self, c) for c in COUNTERS}

    def diff(self, before: dict) -> dict:
        return {c: getattr(self, c) - before[c] for c in COUNTERS}

    def __repr__(self):
        return "Ledger(" + ", ".join(f"{c}={getattr(self, c)}" for c in COUNTERS) + ")"
