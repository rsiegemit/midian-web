"""The key control: MIDIAN's probes without the tree or the report channel.

Mechanism: est = mean of b probes per (agent, family); fetch takes argmax_a est[a, f]; online=True folds every routed
outcome into a running mean.

Ledger: build = n*K*b probes; fetch = n comparisons; observe = 0.
Churn: |arrived|*K*b probes (the replaced agents are re-probed).
Params: online=False."""
import numpy as np

from ._est import probe_means, reprobe, running_mean, scan_argmax
from .base import Method


class FlatProbeArgmax(Method):
    name = "flat_probe_argmax"
    needs = frozenset({"probe"})

    def __init__(self, online=False, **p):
        super().__init__(online=online, **p)
        self.online = online

    def build(self, view, budget):
        self.view, self.b = view, budget.b
        self.cnt = np.full((view.n, view.K), budget.b, np.int64)
        self.est = probe_means(view, budget.b)

    def fetch(self, task):
        return scan_argmax(self.view, self.est[:, task.family])

    def observe(self, task, agent, outcome):
        if self.online:
            running_mean(self.est, self.cnt, agent, task.family, outcome)

    def churn(self, departed, arrived):
        """Re-probe the replaced agents b times per family (len(arrived)*K*b probes)."""
        ids = np.asarray(arrived)
        self.est[ids] = reprobe(self.view, ids, self.b).mean(-1)
        self.cnt[ids] = self.b
