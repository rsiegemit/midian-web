"""UCB1 over arms (agent, family), warmed up on the shared probe budget.

Mechanism: b probes per arm give the initial means; fetch takes argmax_a mean[a, f] + c * sqrt(log t_f / count[a, f])
(t_f = pulls on family f so far); observe updates the running mean and the counts.

Ledger: build = n*K*b probes; fetch = n comparisons; observe = 0.
Params: c=sqrt(2)."""
import numpy as np

from ._est import probe_means, running_mean, scan_argmax
from .base import Method


class UcbPerFamily(Method):
    name = "ucb_per_family"
    needs = frozenset({"probe"})

    def __init__(self, c=2 ** 0.5, **p):
        super().__init__(c=c, **p)
        self.c = c

    def build(self, view, budget):
        self.view = view
        self.cnt = np.full((view.n, view.K), budget.b, np.int64)
        self.mean = probe_means(view, budget.b)
        self.t = np.full(view.K, view.n * budget.b)          # pulls per family

    def fetch(self, task):
        f = task.family
        return scan_argmax(self.view, self.mean[:, f] + self.c * np.sqrt(np.log(self.t[f]) / self.cnt[:, f]))

    def observe(self, task, agent, outcome):
        f = task.family
        self.t[f] += 1
        running_mean(self.mean, self.cnt, agent, f, outcome)
