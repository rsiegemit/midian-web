"""Backends: bernoulli (synthetic), replay (RouterBench cells), routereval (RouterEval pools), llm (vLLM agents).

Backend protocol (duck-typed; see bernoulli.py for the reference):
    n, K, families                              ints / list[str]
    true_skill() -> S[n,K]                      float in [0,1]; runner-only
    declared(source) -> D[n,K]                  honest declared skill; source in {programmatic, self_described,
                                                calibrated (non-live)}
    execute(a, task) -> int                     0/1; Task.instance is the instance seed
    execute_many(agents, families, inst)        int8 of inst.shape: fresh instances from the given seeds (World
                                                index-seeds them)
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


# `calibrated` declarations (docs/errata.md): the live self-rating's error pattern. D | S is drawn from the empirical
# distribution of the live D_self_described given S's decile, pooled over the live specialist n = 100 and 1,000
# populations (scripts/data/fit_declared_calibration.py fits this table and validates it). i.i.d. per (agent, family).
CAL_VALUES = np.array([0, .2, .5, .6, .7, .75, .8, .85, .9, .95, 1], np.float32)   # the ratings live models emit
CAL_P = np.array([                         # P(D = CAL_VALUES[j] | S in decile k), rows k = 0..9
    [0.0535, 0.0439, 0.5577, 0.0000, 0.0445, 0.0000, 0.0865, 0.0528, 0.0445, 0.0000, 0.1167],
    [0.0000, 0.0000, 0.6515, 0.0467, 0.0112, 0.0000, 0.1999, 0.0000, 0.0690, 0.0105, 0.0113],
    [0.1635, 0.0000, 0.4389, 0.0000, 0.0805, 0.0000, 0.0187, 0.0800, 0.1007, 0.0821, 0.0357],
    [0.0000, 0.0000, 0.4372, 0.0000, 0.0000, 0.0000, 0.1702, 0.0235, 0.0000, 0.1005, 0.2686],
    [0.0799, 0.0000, 0.3899, 0.0000, 0.0795, 0.0000, 0.0649, 0.0794, 0.1320, 0.0941, 0.0804],
    [0.0919, 0.0000, 0.4261, 0.0000, 0.0000, 0.0000, 0.0887, 0.0000, 0.1330, 0.1088, 0.1515],
    [0.2955, 0.0000, 0.1222, 0.0000, 0.0000, 0.0000, 0.0000, 0.1955, 0.0443, 0.1980, 0.1445],
    [0.0323, 0.0000, 0.1031, 0.0000, 0.0000, 0.0144, 0.2246, 0.0859, 0.1385, 0.0000, 0.4012],
    [0.0000, 0.0000, 0.0242, 0.0000, 0.0000, 0.0000, 0.0256, 0.0000, 0.1637, 0.0755, 0.7110],
    [0.0837, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0838, 0.1682, 0.0841, 0.0000, 0.5802],
])
CAL_CHUNK = 1 << 18


def cal_bin(S):
    return np.minimum((np.asarray(S) * 10).astype(np.int64), 9)


def calibrated_declared(S, seed):
    """Realistic declaration: over-confident and weakly informative like the live self-rating (mean ~0.69,
    corr ~0.35)."""
    rng = np.random.default_rng(stable_seed_32(seed, "declared_calibrated"))
    cum = np.cumsum(CAL_P, 1)[:, :-1] / CAL_P.sum(1, keepdims=True)
    D = np.empty(S.shape, np.float32)
    for lo in range(0, S.shape[0], CAL_CHUNK):
        b, u = cal_bin(S[lo:lo + CAL_CHUNK]), rng.random(S[lo:lo + CAL_CHUNK].shape)
        D[lo:lo + CAL_CHUNK] = CAL_VALUES[sum((u > cum[b, j]).astype(np.int64) for j in range(cum.shape[1]))]
    return D


def declared_for(S, seed, source, sigma=0.05):
    """The non-live backends' declared(source): `calibrated`, else the honest programmatic copy (for both other
    sources)."""
    return calibrated_declared(S, seed) if source == "calibrated" else noisy_declared(S, seed, sigma)


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
    if name == "routereval":
        from .routereval import RouterEvalBackend
        return RouterEvalBackend(**kw)
    raise ValueError(f"unknown backend {name!r}")
