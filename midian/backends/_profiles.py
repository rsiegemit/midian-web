"""Shared per-distribution shape primitives for backend profile draws.

`world.sample_skill` (frozen core) bakes bernoulli's shapes into itself; this
module factors out the two conventions it and `replay.py` both need — k good
categories per agent, and f % G category groups — as plain functions so any
backend can import them instead of re-deriving the shape."""
from __future__ import annotations

import numpy as np


def pick_k_per_agent(n: int, K: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """bool[n,K], exactly min(k,K) True entries per row, uniform at random."""
    idx = np.argsort(rng.random((n, K)), axis=1)[:, :min(k, K)]
    picked = np.zeros((n, K), dtype=bool)
    picked[np.arange(n)[:, None], idx] = True
    return picked


def group_of(K: int, G: int = 4) -> np.ndarray:
    return np.arange(K) % G


def group_mask(n: int, K: int, G: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """bool[n,K]: one Bernoulli(p) draw per (agent, group), broadcast within the group."""
    return (rng.random((n, G)) < p)[:, group_of(K, G)]
