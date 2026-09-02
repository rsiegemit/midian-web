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
        D = np.clip(view.declared, 1e-3, 1 - 1e-3)          # Beta shape params must be > 0
        return self.n0 * D, self.n0 * (1 - D)
