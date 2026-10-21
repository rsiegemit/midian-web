"""Thompson sampling with a Beta prior of pseudo-count n0 centred on the declared skill D[a, f].

Mechanism: prior Beta(n0*D, n0*(1-D)) (D clipped to [1e-3, 1-1e-3]), then the shared warm-up of b probes per
(agent, family); fetch samples every agent's Beta and takes the argmax; observe adds the outcome to the posterior.

Ledger: build = n*K*b probes + n messages; fetch = n comparisons; observe = 0.
Churn: |arrived| messages + |arrived|*K*b probes (fresh prior and warm-up for the arrivals).
Params: n0=5.0."""
import numpy as np

from ._decl import declared
from ._est import BetaBandit, reprobe


class WarmStartBandit(BetaBandit):
    name = "warm_start_bandit"
    needs = frozenset({"declared", "probe"})

    def __init__(self, n0=5.0, **p):
        super().__init__(n0=n0, **p)
        self.n0 = n0

    def prior(self, view):
        return self._prior(declared(view))

    def _prior(self, D):
        """(alpha0, beta0) = n0 * (D, 1 - D), D clipped: Beta shape params must be > 0."""
        D = np.clip(D, 1e-3, 1 - 1e-3)
        return self.n0 * D, self.n0 * (1 - D)

    def build(self, view, budget):
        self.b = budget.b
        super().build(view, budget)

    def churn(self, departed, arrived):
        """Fresh prior from the arrivals' declarations (len(arrived) messages) plus b probes per family each."""
        v, ids = self.view, np.asarray(arrived)
        v.ledger.message(ids.size)
        a0, b0 = self._prior(v.declared[ids])
        s = reprobe(v, ids, self.b).sum(-1)
        self.alpha[ids], self.beta[ids] = a0 + s, b0 + self.b - s
