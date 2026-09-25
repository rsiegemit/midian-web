"""Thompson sampling per (agent, family) with a flat Beta(1, 1) prior.

Mechanism: the shared warm-up of b probes per (agent, family) seeds the posteriors; fetch samples every agent's Beta and
takes the argmax; observe adds the outcome to the posterior.

Ledger: build = n*K*b probes; fetch = n comparisons; observe = 0.
Params: none."""
from ._est import BetaBandit


class ThompsonPerFamily(BetaBandit):
    name = "thompson_per_family"
