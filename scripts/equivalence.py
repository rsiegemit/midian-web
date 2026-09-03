"""Refactor guard: fingerprint every pick, ledger count and estimate of the MIDIAN family on the bernoulli backend.
    python scripts/equivalence.py before.json      # on the old code
    python scripts/equivalence.py after.json       # on the new code, then diff the two files (must be identical)"""
import hashlib, json, os, sys
import numpy as np
from rte.budget import Budget
from rte.methods import load_method
from rte.world import World

SPECS = [("midian", {}), ("midian", {"r": 5}), ("midian", {"verify": True, "cached": True}), ("midian_v", {}),
         ("midian", {"stratify": True}), ("midian_sh", {}), ("midian_a", {}), ("midian_sha", {})]
CELLS = [(n, d, s) for n in (100, 1000) for d in ("specialist", "heavy_tail") for s in (1, 2, 3)]
h = lambda *xs: hashlib.blake2b(b"|".join(np.ascontiguousarray(x).tobytes() if isinstance(x, np.ndarray) else json.dumps(x, sort_keys=True).encode() for x in xs), digest_size=12).hexdigest()

out = {}
for name, params in SPECS:
    for n, dist, seed in CELLS:
        w = World(n, 16, dist, 0.25, seed=seed, liar_select="low_skill_first"); m = load_method(name)(**params)
        m.build(w.view(m.needs), Budget(3)); build = w.ledger.snapshot(); picks = []
        for i, t in enumerate(w.tasks(300)):
            if i == 150:
                ids = w.churn(0.1); m.churn(ids, ids)
            a = m.fetch(t); m.observe(t, a, w.execute(a, t)); picks.append(int(a))
        out[f"{name}{json.dumps(params, sort_keys=True)}|n={n}|{dist}|s{seed}"] = dict(
            picks=h(picks), build=build, run=w.ledger.snapshot(), est=h(m.est), summary=h(*m.summary), best=h(*m.best))
json.dump(out, open(sys.argv[1], "w"), indent=1, sort_keys=True); print(len(out), "fingerprints ->", sys.argv[1])
