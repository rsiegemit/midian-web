"""Diagnostic ONLY: true skill of the leaf cohorts of MIDIAN w/o audits and of MIDIAN, the rows missing from the
text-retriever table. Both observe probe OUTCOMES, which on the llm backend are Bernoulli(S) draws, so the
cohort is reconstructed on a bernoulli world carrying the population's exact measured S -- statistically the same
selection the live run makes. S is read for MEASUREMENT only; no method ever sees it.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from rte.budget import Budget
from rte.world import World
from rte.methods.midian import Midian

RD = os.environ["RTE_DATA"]; K, B = 16, 3
out = {}
for seed in (1, 2, 3):
    Spath = f"{RD}/populations/specialist_n100000_K16_seed{seed}/S.npy"
    S = np.load(Spath).astype(np.float32)
    w = World(S.shape[0], K, "specialist", 0.0, seed=seed, backend="bernoulli",
              backend_kwargs={"calibrate_from": Spath})
    w.backend._S = S                                   # exact matrix, so the cohort is comparable to the text shortlists
    tasks = {}
    for t in w.tasks(400):
        tasks.setdefault(int(t.family), t)
        if len(tasks) == K: break
    for name, mk in [("MIDIAN w/o audits leaf cohort", lambda: Midian(audit=False, r=10)),
                     ("MIDIAN leaf cohort", lambda: Midian(r=10))]:
        m = mk(); m.build(w.view(m.needs), Budget(B))
        sk, sizes, bst, pick = [], [], [], []
        for f, t in sorted(tasks.items()):
            a = m.fetch(t)
            coh = m.leaves[m.leaf_of[a]]
            sel = np.concatenate([[a], coh[(coh >= 0) & (coh != a)]])[:10]
            sk.append(S[sel, f].mean()); sizes.append(len(sel))
            bst.append(S[sel, f].max()); pick.append(S[a, f])
        out.setdefault(name, []).append(float(np.mean(sk)))
        out.setdefault(name + "  [best in list]", []).append(float(np.mean(bst)))
        out.setdefault(name + "  [MIDIAN's own pick]", []).append(float(np.mean(pick)))
        print(f"seed {seed} {name}: mean {np.mean(sk):.3f}  best-in-list {np.mean(bst):.3f}  own pick {np.mean(pick):.3f}", flush=True)
    out.setdefault("-- population mean --", []).append(float(S.mean()))
    out.setdefault("-- best single agent per family --", []).append(float(np.mean([S[:, f].max() for f in range(K)])))
print("\nTRUE skill of the MIDIAN cohorts, n = 10^5 specialist, mean over 3 seeds")
for k in sorted(out, key=lambda k: np.mean(out[k])):
    print(f"  {k:36s} {np.mean(out[k]):.3f}")
