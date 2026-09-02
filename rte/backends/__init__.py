"""Backends: bernoulli (synthetic), replay (RouterBench cells), llm (vLLM agents).

Backend protocol (duck-typed; see bernoulli.py for the reference):
    n, K, families                              ints / list[str]
    true_skill() -> S[n,K]                      float in [0,1]; runner-only
    declared(source) -> D[n,K]                  honest declared skill; source in {programmatic, self_described}
    execute(a, task) -> int                     0/1; Task.instance is the instance seed
    execute_many(agents, families, reps, rng) -> (len(agents), reps) int8   fresh instances
    stats() -> dict                             e.g. cache hit rate
"""
from __future__ import annotations


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
