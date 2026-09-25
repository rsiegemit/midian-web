"""UCB1 over arms (agent, family); warmup = the shared n*K*b budget, b pulls per arm; online updates."""
import numpy as np
from .base import Method
from ._est import probe_means, running_mean, scan_argmax


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
