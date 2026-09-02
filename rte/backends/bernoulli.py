"""Synthetic backend: outcome ~ Bernoulli(S[a,f]). Unit tests, tree invariants,
and the n=1e5..1e7 cost-scaling curves only. Never a headline number.

`calibrate_from` : path to a measured S (.npy) from the llm world; skills are
then resampled from its empirical per-family distribution instead of `dist`."""
from __future__ import annotations

import numpy as np

from ..stable_hash import stable_seed_32
from ..world import sample_skill, Task
from . import noisy_declared

CHUNK = 1_000_000


class BernoulliBackend:
    def __init__(self, n: int, K: int, dist: str, seed: int, rng: np.random.Generator,
                 calibrate_from: str | None = None, declared_noise: float = 0.05, **_):
        self.n, self.K = int(n), int(K)
        self.families = [f"fam{f:02d}" for f in range(self.K)]
        self.seed = int(seed)
        if calibrate_from:
            Sm = np.load(calibrate_from).astype(np.float32)
            self._S = np.empty((self.n, self.K), dtype=np.float32)
            for lo in range(0, self.n, CHUNK):
                hi = min(self.n, lo + CHUNK)
                self._S[lo:hi] = Sm[rng.integers(0, Sm.shape[0], size=hi - lo)]
        elif self.n <= CHUNK:
            self._S = sample_skill(dist, self.n, self.K, rng).astype(np.float32)
        else:
            self._S = np.empty((self.n, self.K), dtype=np.float32)
            for lo in range(0, self.n, CHUNK):
                hi = min(self.n, lo + CHUNK)
                self._S[lo:hi] = sample_skill(dist, hi - lo, self.K, rng)
        self.declared_noise = float(declared_noise)

    def true_skill(self) -> np.ndarray:
        return self._S

    def declared(self, source: str = "programmatic") -> np.ndarray:
        return noisy_declared(self._S, self.seed, self.declared_noise)   # no LLM here: both sources are the honest control

    def execute(self, a: int, task: Task) -> int:
        u = np.random.default_rng(stable_seed_32(self.seed, "exec", int(a), task.family, task.instance)).random()
        return int(u < self._S[a, task.family])

    def execute_many(self, agents, families, reps, rng) -> np.ndarray:
        p = self._S[agents, families][..., None]
        return (rng.random(p.shape[:-1] + (int(reps),)) < p).astype(np.int8)

    def stats(self) -> dict:
        return {}
