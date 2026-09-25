"""Thompson sampling with a Beta prior of pseudo-count n0 centred on the declared skill D[a, f].

Mechanism: prior Beta(n0*D, n0*(1-D)) (D clipped to [1e-3, 1-1e-3]), then the shared warm-up of b probes per
(agent, family); fetch samples every agent's Beta and takes the argmax; observe adds the outcome to the posterior.

Ledger: build = n*K*b probes + n messages; fetch = n comparisons; observe = 0.
Churn: |arrived| messages + |arrived|*K*b probes (fresh prior and warm-up for the arrivals).
Params: n0=5.0."""
import numpy as np
from ._est import BetaBandit, reprobe


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
        self.b = budget.b
        super().build(view, budget)

    def churn(self, departed, arrived):
        """Fresh prior from the arrivals' declarations (len(arrived) messages) plus b probes per family each."""
        v, ids = self.view, np.asarray(arrived)
        v.ledger.message(ids.size)
        D = np.clip(v.declared[ids], 1e-3, 1 - 1e-3)
        s = reprobe(v, ids, self.b).sum(-1)
        self.alpha[ids], self.beta[ids] = self.n0 * D + s, self.n0 * (1 - D) + self.b - s
