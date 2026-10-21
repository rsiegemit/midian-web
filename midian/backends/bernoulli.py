"""Synthetic backend: outcome ~ Bernoulli(S[a,f]). Unit tests, tree invariants,
and the n=1e5..1e7 cost-scaling curves only. Never a headline number.

`calibrate_from` : path to a measured S (.npy) from the llm world; skills are
then resampled from its empirical per-family distribution instead of `dist`."""
from __future__ import annotations

import numpy as np

from ..stable_hash import stable_seed_32
from ..world import Task, sample_skill
from . import declared_for

CHUNK = 1_000_000


class BernoulliBackend:
    def __init__(self, n: int, K: int, dist: str, seed: int, rng: np.random.Generator,
                 calibrate_from: str | None = None, **_):
        self.n, self.K, self.dist = int(n), int(K), dist
        self.families = [f"fam{f:02d}" for f in range(self.K)]
        self.seed = int(seed)
        self._Sm = np.load(calibrate_from).astype(np.float32) if calibrate_from else None
        self._S = np.empty((self.n, self.K), dtype=np.float32)
        for lo in range(0, self.n, CHUNK):                     # chunked: bounded peak memory at n = 10^7
            m = min(self.n, lo + CHUNK) - lo
            self._S[lo:lo + m] = self._draw(m, rng)

    def _draw(self, m: int, rng) -> np.ndarray:
        """m skill rows: resampled from the calibration S, else drawn from `dist`."""
        return self._Sm[rng.integers(0, self._Sm.shape[0], size=m)] if self._Sm is not None \
            else sample_skill(self.dist, m, self.K, rng)

    def true_skill(self) -> np.ndarray:
        return self._S

    def declared(self, source: str = "programmatic") -> np.ndarray:
        # no LLM here: programmatic / self_described are the honest control
        return declared_for(self._S, self.seed, source)

    def execute(self, a: int, task: Task) -> int:
        u = np.random.default_rng(stable_seed_32(self.seed, "exec", int(a), task.family, task.instance)).random()
        return int(u < self._S[a, task.family])

    def execute_many(self, agents, families, inst) -> np.ndarray:
        # 53-bit uniform
        u = (np.asarray(inst, np.uint64) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(self.seed)) >> np.uint64(11)
        return ((u.astype(np.float64) / 2 ** 53) < self._S[agents, families]).astype(np.int8)

    def stats(self) -> dict:
        return {}

    # ---- churn
    def snapshot(self):
        return self._S.copy()
    def restore(self, snap):
        self._S = snap.copy()
    def redraw(self, ids, rng):
        self._S[ids] = self._draw(len(ids), rng)
