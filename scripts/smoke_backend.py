#!/usr/bin/env python3
"""End-to-end smoke test of LLMBackend against whatever the fleet is serving.

    $RTE_DATA/env/rte/bin/python scripts/smoke_backend.py

n=5, K=4, 20 probes. Exercises the whole path: prompt, generation, tool round, verifier, memo,
ledger. Works against the one-model smoke server as well as the full fleet.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                                          # noqa: E402

from rte import llm_client                                                  # noqa: E402
from rte.backends import llm, population                                    # noqa: E402
from rte.world import Task                                                  # noqa: E402

N, K = 5, 4
FAMILIES = ["basic_arithmetic", "gcd", "leg_counting", "spell_backward"]


def main() -> int:
    served = sorted(llm_client.endpoints())
    print(f"endpoints: {served}")
    if not served:
        print("FAIL: nothing is being served")
        return 1
    cfg = population.pinned_cfg(served)
    llm.ladder = population.ladder = lambda: cfg           # every agent uses a served model

    b = llm.LLMBackend(n=N, K=K, dist="specialist", seed=1, families=FAMILIES, concurrency=16,
                       population_dir=str(llm.POP_DIR / "smoke"))
    for p in b.profiles:
        print(f"  agent {p['id']}: {p['model']} tool={p['tool']} "
              f"specialty={[FAMILIES[f] for f in p['specialty']]}")

    t0 = time.time()
    o = b.execute(0, Task(0, 0, 12345))
    print(f"\nexecute(agent 0, {FAMILIES[0]}#12345) -> {o}   ({time.time() - t0:.1f}s)")

    t0 = time.time()
    out = b.execute_many(np.repeat(np.arange(N), K), np.tile(np.arange(K), N), 1,
                         np.random.default_rng(0))
    dt = time.time() - t0
    print(f"\n{out.size}-probe grid ({dt:.1f}s, {out.size / max(dt, 1e-9):.1f} probes/s):")
    grid = out.reshape(N, K)
    print("       " + "".join(f"{f[:12]:>14s}" for f in FAMILIES))
    for a in range(N):
        print(f"agent{a}  " + "".join(f"{int(grid[a, f]):>14d}" for f in range(K)))

    before = llm_client.stats()["generations"]
    b.execute(0, Task(0, 0, 12345))
    after = llm_client.stats()["generations"]
    print(f"\nrepeat execution: generations {before} -> {after} (expect no change)")
    assert after == before, "the memo did not serve the repeat"

    print(f"\nllm_client.stats(): {llm_client.stats()}")
    print(f"backend.stats():    {b.stats()}")
    print(f"mean outcome over the {out.size} probes: {out.mean():.3f}")
    print("\nSMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
