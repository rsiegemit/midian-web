#!/usr/bin/env python3
"""End-to-end smoke test of LLMBackend against a live vLLM endpoint.

Run AFTER `sbatch scripts/serve_smoke.sbatch` has written $RTE_DATA/endpoints.json:

    $RTE_DATA/env/rte/bin/python scripts/smoke_backend.py

Only models that are actually served are used, so the single-model smoke fleet works: every
agent is pinned to the one served model and the population is n=5, K=4, 20 probes total.
Exercises the whole path -- prompt build, generation, tool round, verifier, memo, ledger.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                                          # noqa: E402

from rte import llm_client                                                  # noqa: E402
from rte.backends import llm as B                                           # noqa: E402
from rte.world import Task                                                  # noqa: E402

N, K, REPS = 5, 4, 4                    # 5 agents x 4 families x ... = 20 probes below
FAMILIES = ["basic_arithmetic", "gcd", "leg_counting", "spell_backward"]


def main() -> int:
    eps = llm_client.endpoints()
    print(f"endpoints: {eps}")
    served = sorted(eps)
    if not served:
        print("FAIL: endpoints.json is empty")
        return 1

    # Pin the ladder to whatever is actually served (the smoke fleet serves one model).
    B.LADDER = served
    B.SMALL = served
    B.BIG = served
    B.LARGE_MODELS = frozenset()

    b = B.LLMBackend(n=N, K=K, dist="specialist", seed=1, rng=np.random.default_rng(1),
                     families=FAMILIES, concurrency=16,
                     population_dir=str(Path(os.environ.get(
                         "RTE_DATA", "/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte"))
                         / "populations" / "smoke"))
    print("profiles:")
    for p in b.profiles:
        print(f"  agent {p['id']}: {p['model']} tool={p['tool']} "
              f"specialty={[FAMILIES[f] for f in p['specialty']]}")

    # ---- 1. one execution
    t0 = time.time()
    o = b.execute(0, Task(0, 0, 12345))
    print(f"\nexecute(agent 0, basic_arithmetic#12345) -> {o}   ({time.time() - t0:.1f}s)")
    assert o in (0, 1)

    # ---- 2. 20 probes: 5 agents x 4 families x 1 rep
    t0 = time.time()
    agents = np.repeat(np.arange(N), K)
    fams = np.tile(np.arange(K), N)
    out = b.execute_many(agents, fams, reps=1, rng=np.random.default_rng(0))
    dt = time.time() - t0
    print(f"\n20-probe grid ({out.size} probes, {dt:.1f}s, {out.size / max(dt, 1e-9):.2f} probes/s):")
    grid = out.reshape(N, K)
    print("       " + "".join(f"{f[:12]:>14s}" for f in FAMILIES))
    for a in range(N):
        print(f"agent{a}  " + "".join(f"{int(grid[a, f]):>14d}" for f in range(K)))
    assert out.shape == (N * K, 1)
    assert set(np.unique(out)) <= {0, 1}

    # ---- 3. memo: the identical call must be a cache hit and cost no generation
    s0 = llm_client.stats()
    b.execute(0, Task(0, 0, 12345))
    s1 = llm_client.stats()
    print(f"\nrepeat execution: generations {s0['generations']} -> {s1['generations']} "
          f"(expect no change), hits {s0['hits']} -> {s1['hits']}")
    assert s1["generations"] == s0["generations"], "memo did not serve the repeat"

    print(f"\nllm_client.stats(): {llm_client.stats()}")
    print(f"backend.stats():    {b.stats()}")
    print(f"mean outcome over the 20 probes: {out.mean():.3f}")
    print("\nSMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
