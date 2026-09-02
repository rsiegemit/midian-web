"""Thompson sampling with a flat Beta(1,1) prior; same n*K*b warmup as the other probe methods."""
from ._est import BetaBandit


class ThompsonPerFamily(BetaBandit):
    name = "thompson_per_family"
