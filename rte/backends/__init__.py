"""Backends: bernoulli (synthetic), replay (RouterBench cells), llm (vLLM agents).

Backend protocol (duck-typed; see bernoulli.py for the reference):
    n, K, families                              ints / list[str]
    true_skill() -> S[n,K]                      float in [0,1]; runner-only
    declared(source) -> D[n,K]                  honest declared skill; source in {programmatic, self_described}
    execute(a, task) -> int                     0/1; Task.instance is the instance seed
    execute_many(agents, families, inst) -> int8 of inst.shape   fresh instances from the given seeds (World index-seeds them)
    stats() -> dict                             e.g. cache hit rate
    snapshot() / restore(snap) / redraw(ids, rng)   churn support: copy the population state, put it back,
                                                and replace agents `ids` in place with fresh profiles
"""
from __future__ import annotations

import numpy as np

from ..stable_hash import stable_seed_32


def noisy_declared(S, seed, sigma=0.05):
    """Honest programmatic declaration D = clip(S + N(0, sigma)). Backends without an LLM use it for both sources."""
    rng = np.random.default_rng(stable_seed_32(seed, "declared"))
    return np.clip(S + rng.normal(0, sigma, S.shape), 0, 1).astype(np.float32)


def make(name: str, **kw):
    if name == "bernoulli":
        from .bernoulli import BernoulliBackend
        return BernoulliBackend(**kw)
    if name == "replay":
        from .replay import ReplayBackend
        return ReplayBackend(**kw)
    if name == "llm":
        from .llm import LLMBackend
        return LLMBackend(**kw)
    raise ValueError(f"unknown backend {name!r}")
