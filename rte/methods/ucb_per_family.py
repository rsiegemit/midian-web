"""UCB1 over arms (agent, family); warmup = the shared n*K*b budget, b pulls per arm; online updates."""
import numpy as np
from .base import Method
from ._est import probe_successes


class UcbPerFamily(Method):
    name = "ucb_per_family"
    needs = frozenset({"probe"})

    def __init__(self, c=2 ** 0.5, **p):
        super().__init__(c=c, **p)
        self.c = c

    def build(self, view, budget):
        self.view = view
        self.cnt = np.full((view.n, view.K), budget.b, np.int64)
        self.mean = probe_successes(view, budget.b) / budget.b
        self.t = np.full(view.K, view.n * budget.b)          # pulls per family

    def fetch(self, task):
        f = task.family
        self.view.ledger.compare(self.view.n)
        return int(np.argmax(self.mean[:, f] + self.c * np.sqrt(np.log(self.t[f]) / self.cnt[:, f])))

    def observe(self, task, agent, outcome):
        f = task.family
        self.cnt[agent, f] += 1; self.t[f] += 1
        self.mean[agent, f] += (outcome - self.mean[agent, f]) / self.cnt[agent, f]
