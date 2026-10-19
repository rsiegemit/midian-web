"""Build one de-cloned live population (LLMBackend declone) before any grid runs on it.

    PYTHONPATH=. python scripts/data/build_declone.py --n 1000 --dist specialist --seed 1 [--declone 36] [--reps 5]

Measures S (per signature, shared beside the population: declone<V>_signature_S.json), the self-described claims (per
agent) and every probe outcome a b <= reps method reads: the first `reps` index-seeded probes of every (agent, family),
the same instances every method sees (World._probe). Outcomes go to the population's outcome store
(midian.backends.outcome_store), chunk by chunk, so a killed build resumes where it stopped; a finished build writes
<population>/BUILT. The World is built exactly
as the grid builds it (beta = 0: liars change reports and claims, never answers), so the probe instances match."""
from __future__ import annotations

import argparse
import time

import numpy as np

from midian.world import World


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--dist", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--declone", type=int, default=36)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--chunk", type=int, default=2000, help="agents per batch")
    a = p.parse_args(argv)
    t0 = time.time()
    w = World(a.n, 16, a.dist, 0.0, seed=a.seed, backend="llm", declared_source="self_described",
              backend_kwargs={"declone": a.declone})          # measures S and the self-described claims
    print(f"[build] {w.backend.dir.name}: S and claims in {time.time() - t0:.0f}s", flush=True)
    fams = np.arange(w.K)
    for lo in range(0, w.n, a.chunk):
        ag = np.arange(lo, min(lo + a.chunk, w.n))
        o = w.probe_many(ag[:, None], fams[None, :], a.reps)   # k = 0..reps-1 for every cell: a fresh World
        print(f"[build] agents {lo}-{ag[-1]}: mean outcome {o.mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)
    (w.backend.dir / "BUILT").write_text(f"reps={a.reps}\n")       # the grid driver launches units once this exists
    print(f"[build] done {w.backend.dir.name} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
