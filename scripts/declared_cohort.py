"""Diagnostic ONLY: is the DECLARED top-10 a better cohort source than MIDIAN's leaf cohort?
Declared top-10 is built from the live backend's own self-described D with the regime's liars applied
(world.select_liars + world.apply_lying, the exact benchmark mechanism). The MIDIAN cohort is
reconstructed on a bernoulli world carrying the population's exact measured S. S is read for MEASUREMENT
only; no method ever sees it."""
import os, sys
import numpy as np
sys.path.insert(0, '/n/home02/rsiegelmann/rte')
from rte.budget import Budget
from rte.world import World, select_liars, apply_lying, DELTA_INFLATE
from rte.methods.midian import Midian
from rte.stable_hash import stable_seed_32

RD = os.environ["RTE_DATA"]; K, B, TOPK = 16, 3, 10
REG = [(0.0, "random", "beta=0 (honest)"), (0.25, "random", "beta=0.25 random"),
       (0.25, "low_skill_first", "beta=0.25 CARTEL"), (0.5, "random", "beta=0.5 random"),
       (0.5, "low_skill_first", "beta=0.5 CARTEL")]
acc = {}
for seed in (1, 2, 3):
    P = f"{RD}/populations/specialist_n100000_K16_seed{seed}"
    S = np.load(f"{P}/S.npy").astype(np.float32); Dh = np.load(f"{P}/D_self_described.npy").astype(np.float32)
    for beta, ls, lbl in REG:
        liars = select_liars(S, beta, ls, np.random.default_rng(stable_seed_32(seed, "liars")))
        D = apply_lying(Dh, liars, "inflate", None, DELTA_INFLATE)
        mu = np.mean([S[np.argsort(-D[:, f], kind="stable")[:TOPK], f].mean() for f in range(K)])
        bs = np.mean([S[np.argsort(-D[:, f], kind="stable")[:TOPK], f].max() for f in range(K)])
        lr = np.mean([liars[np.argsort(-D[:, f], kind="stable")[:TOPK]].mean() for f in range(K)])
        acc.setdefault((lbl, "declared top-10"), []).append((mu, bs, lr))
        w = World(S.shape[0], K, "specialist", beta, seed=seed, liar_select=ls, backend="bernoulli",
                  backend_kwargs={"calibrate_from": f"{P}/S.npy"})
        w.backend._S = S
        tasks = {}
        for t in w.tasks(400):
            tasks.setdefault(int(t.family), t)
            if len(tasks) == K: break
        m = Midian(r=10); m.build(w.view(m.needs), Budget(B))
        mus, bss, lrs = [], [], []
        for f, t in sorted(tasks.items()):
            a = m.fetch(t); coh = m.leaves[m.leaf_of[a]]
            sel = np.concatenate([[a], coh[(coh >= 0) & (coh != a)]])[:TOPK]
            mus.append(S[sel, f].mean()); bss.append(S[sel, f].max()); lrs.append(w.liars[sel].mean())
        acc.setdefault((lbl, "MIDIAN cohort"), []).append((np.mean(mus), np.mean(bss), np.mean(lrs)))
    print(f"seed {seed} done", flush=True)
print(f"\nCohort sources at n = 10^5 specialist, top-{TOPK}, mean over 3 seeds")
print(f"{'regime':20s} {'source':18s} {'mean skill':>11s} {'best in list':>13s} {'liar frac':>10s}")
print("-" * 76)
for beta, ls, lbl in REG:
    for src in ("declared top-10", "MIDIAN cohort"):
        v = np.array(acc[(lbl, src)]).mean(0)
        print(f"{lbl:20s} {src:18s} {v[0]:11.3f} {v[1]:13.3f} {v[2]:10.2f}")
