"""Thompson sampling with a Beta prior of pseudo-count n0 centred on the declared skill D[a,f]."""
import numpy as np
from ._est import BetaBandit


class WarmStartBandit(BetaBandit):
    name = "warm_start_bandit"
    needs = frozenset({"declared", "probe"})

    def __init__(self, n0=5.0, **p):
        super().__init__(n0=n0, **p)
        self.n0 = n0

    def prior(self, view):
        view.ledger.message(view.n)                          # collect declarations
        D = np.clip(view.declared, 1e-3, 1 - 1e-3)          # Beta shape params must be > 0
        return self.n0 * D, self.n0 * (1 - D)

    def build(self, view, budget):
        self.b = budget.b; super().build(view, budget)

    def churn(self, departed, arrived):
        """Fresh prior from the arrivals' declarations (len(arrived) messages) plus b probes per family each."""
        v, ids = self.view, np.asarray(arrived); v.ledger.message(ids.size)
        D = np.clip(v.declared[ids], 1e-3, 1 - 1e-3); s = v.probe_many(ids[:, None], np.arange(v.K)[None, :], self.b).sum(-1)
        self.alpha[ids], self.beta[ids] = self.n0 * D + s, self.n0 * (1 - D) + self.b - s
