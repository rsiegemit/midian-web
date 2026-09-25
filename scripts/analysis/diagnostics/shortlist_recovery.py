"""Diagnostic ONLY: what each framework shortlist is worth, and how much of it the frameworks recover.
    RTE_DATA=... python scripts/analysis/diagnostics/shortlist_recovery.py   ->
    paper/diagnostics/shortlist_recovery.{txt,csv}
Per (n, regime, shortlist source), live specialist populations, seeds 1-3, top-10 per family, averaged over families
and seeds, from the MEASURED S (read for measurement only; no method ever sees it):
  list mean   S of a uniform pick from the list        best in list   the ceiling a competent consumer reaches
  position 1  S of the list's first entry               liar frac      fraction of the list that is a liar
Lists are built exactly as the framework adapter builds them (rte/methods/frameworks/_common.py retrieve): TF-IDF
without dedup (pre-registered), every other source over the dedup pool; declared = top-k by the regime's lied-to D;
dense / rerank from the cached Qwen3 files. The MIDIAN cohort is MIDIAN's pick + its leaf cohort, rebuilt on a bernoulli
world carrying the population's exact S (as scripts/analysis/diagnostics/declared_cohort.py), so it is a reconstruction.
Recovery per framework = (framework success - list mean) / (best in list - list mean), framework success from
figures/shortlist/live.csv (seed mean, same cell); 0 = a uniform pick from the list, 1 = the list's best agent."""

import json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from rte.backends import families
from rte.budget import Budget
from rte.methods._learned import embed
from rte.methods.frameworks._common import _hash_tfidf
from rte.methods.midian import Midian
from rte.stable_hash import stable_seed_32
from rte.world import World, select_liars, apply_lying, DELTA_INFLATE

RD = os.environ["RTE_DATA"]
K, TOPK = 16, 10
REG = {"beta0": (0.0, "random"), "cartel": (0.5, "low_skill_first")}
SOTA = "shortlist_sota_qwen_qwen3_embedding_8b_qwen_qwen3_reranker_4b_k10p50_dd"
TAGS = {"": "", "_i0258be9d": "_icomp", "_i06c7d226": "_idemo"}
fdesc = [families.describe(f) for f in families.FAMILIES_16]
Fm = embed(fdesc)
recs = []
for n in (10000, 100000):
    for seed in (1, 2, 3):
        P = f"{RD}/populations/specialist_n{n}_K16_seed{seed}"
        desc = json.load(open(f"{P}/descriptions.json"))
        S = np.load(f"{P}/S.npy").astype(np.float32)
        Dh = np.load(f"{P}/D_self_described.npy").astype(np.float32)
        first = {}
        for i, t in enumerate(desc):
            first.setdefault(t, i)
        pool = np.array(sorted(first.values()), dtype=np.int64)
        rank = lambda sc: {f: pool[np.argsort(-sc(f)[pool], kind="stable")][:TOPK] for f in range(K)}
        X = _hash_tfidf(desc + fdesc)
        Xa, Xf = X[:n], X[n:]
        lists = {
            "tfidf": {f: np.argsort(-(Xa @ Xf[f]), kind="stable")[:TOPK] for f in range(K)},
            "embed": rank(lambda f, Em=np.load(f"{P}/descriptions_minilm.npy"): Em @ Fm[f]),
        }
        Eq = np.load(f"{P}/descriptions_qwen_qwen3_embedding_8b.npy")
        for tag, sfx in TAGS.items():
            if sfx:
                lists["dense" + sfx] = rank(
                    lambda f, Fq=np.load(f"{P}/families_qwen_qwen3_embedding_8b{tag}.npy"): Eq @ Fq[f]
                )
            T = np.load(f"{P}/{SOTA}{tag}.npy")
            lists["sota" + sfx] = {f: T[f][T[f] >= 0][:TOPK] for f in range(K)}
        for reg, (beta, ls) in REG.items():
            liars = select_liars(S, beta, ls, np.random.default_rng(stable_seed_32(seed, "liars")))
            D = apply_lying(Dh, liars, "inflate", None, DELTA_INFLATE)
            L = dict(lists, declared=rank(lambda f: D[:, f]))
            w = World(
                n,
                K,
                "specialist",
                beta,
                seed=seed,
                liar_select=ls,
                backend="bernoulli",
                backend_kwargs={"calibrate_from": f"{P}/S.npy"},
            )
            w.backend._S = S
            m = Midian(r=10)
            m.build(w.view(m.needs), Budget(3))
            va = {}
            for t in w.tasks(400):
                f = int(t.family)
                if f in va:
                    continue
                a = m.fetch(t)
                coh = m.leaves[m.leaf_of[a]]
                va[f] = np.concatenate([[a], coh[(coh >= 0) & (coh != a)]])[:TOPK]
                if len(va) == K:
                    break
            L["va_cohort"] = va
            for src, sl in L.items():
                if sl is None:
                    continue
                recs.append(
                    dict(
                        n=n,
                        seed=seed,
                        regime=reg,
                        shortlist=src,
                        list_mean=np.mean([S[sl[f], f].mean() for f in range(K)]),
                        best=np.mean([S[sl[f], f].max() for f in range(K)]),
                        pos1=np.mean([S[sl[f][0], f] for f in range(K)]),
                        liar_frac=np.mean([liars[sl[f]].mean() for f in range(K)]),
                    )
                )
            print(f"n={n} seed={seed} {reg} done", flush=True)

d = pd.DataFrame(recs).groupby(["n", "regime", "shortlist"]).mean(numeric_only=True).drop(columns="seed").reset_index()
fw = pd.read_csv(f"{ROOT}/figures/shortlist/live.csv")
fw = fw[(fw.dist == "specialist") & (fw.shortlist != "-")][["n", "regime", "shortlist", "arm", "mean", "seeds"]]
j = fw.merge(d, on=["n", "regime", "shortlist"])
j["recovery"] = (j["mean"] - j.list_mean) / (j.best - j.list_mean)
r = (
    j.groupby(["n", "regime", "shortlist"])
    .agg(
        frameworks=("arm", "nunique"),
        fw_mean=("mean", "mean"),
        rec_mean=("recovery", "mean"),
        rec_min=("recovery", "min"),
        rec_max=("recovery", "max"),
    )
    .reset_index()
)
out = d.merge(r, on=["n", "regime", "shortlist"], how="left")
os.makedirs(f"{ROOT}/paper/diagnostics", exist_ok=True)
out.to_csv(f"{ROOT}/paper/diagnostics/shortlist_recovery.csv", index=False)
j.to_csv(f"{ROOT}/paper/diagnostics/shortlist_recovery_per_framework.csv", index=False)
with open(f"{ROOT}/paper/diagnostics/shortlist_recovery.txt", "w") as fh:
    fh.write(__doc__ + "\n")
    fh.write(out.round(3).to_string(index=False) + "\n")
print(out.round(3).to_string(index=False))
