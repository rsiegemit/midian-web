"""The key control: MIDIAN's probes, no hierarchy, no report channel. est = mean of b probes; argmax per family.
`cached=True` precomputes the argmax (O(1) per fetch, recomputed on observe if online)."""
import numpy as np
from .base import Method
from ._est import lookup, probe_means, reprobe, running_mean, scan_argmax


class FlatProbeArgmax(Method):
    name = "flat_probe_argmax"
    needs = frozenset({"probe"})

    def __init__(self, cached=False, online=False, **p):
        super().__init__(cached=cached, online=online, **p)
        self.cached, self.online = cached, online

    def build(self, view, budget):
        self.view, self.b = view, budget.b
        self.cnt = np.full((view.n, view.K), budget.b, np.int64)
        self.est = probe_means(view, budget.b)
        self.best = np.argmax(self.est, axis=0)

    def fetch(self, task):
        if self.cached:
            return lookup(self.view, self.best, task.family)
        return scan_argmax(self.view, self.est[:, task.family])

    def observe(self, task, agent, outcome):
        if self.online:
            f = task.family
            running_mean(self.est, self.cnt, agent, f, outcome)
            self.best[f] = np.argmax(self.est[:, f])

    def churn(self, departed, arrived):
        """Re-probe the replaced agents b times per family (len(arrived)*K*b probes), then re-argmax."""
        ids = np.asarray(arrived)
        self.est[ids] = reprobe(self.view, ids, self.b).mean(-1)
        self.cnt[ids] = self.b; self.best = np.argmax(self.est, axis=0)
