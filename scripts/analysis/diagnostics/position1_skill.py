"""Diagnostic ONLY: true skill of the agent each source puts in POSITION 1. Every retriever returns a ranked list
and every framework sees its top-ranked item first, so position-1 quality is the statistic the shuffle control says
actually drives outcomes. S is read for MEASUREMENT only; no method ever sees it."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from midian.backends import families
from midian.methods._learned import embed
from midian.methods.frameworks._common import _bm25, _hash_tfidf, _rrf

RD = os.environ["RTE_DATA"]
K = 16
fam = [families.describe(f) for f in families.FAMILIES_16]
base = "shortlist_sota_qwen_qwen3_embedding_8b_qwen_qwen3_reranker_4b_k10p50_dd"
top1, mean10 = {}, {}
for seed in (1, 2, 3):
    p = f"{RD}/populations/specialist_n100000_K16_seed{seed}"
    desc = json.load(open(f"{p}/descriptions.json"))
    S = np.load(f"{p}/S.npy")
    D = np.load(f"{p}/D_self_described.npy")
    first = {}
    for i, t in enumerate(desc):
        first.setdefault(t, i)
    pool = np.array(sorted(first.values()), dtype=np.int64)

    def rec(name, sel):
        top1.setdefault(name, []).append(float(np.mean([S[sel[f][0], f] for f in range(K)])))
        mean10.setdefault(name, []).append(float(np.mean([S[sel[f][:10], f].mean() for f in range(K)])))

    def rank(score):
        return {f: pool[np.argsort(-score[f][pool], kind="stable")] for f in range(K)}

    X = _hash_tfidf(desc + fam)
    Xa, Xf = X[: len(desc)], X[len(desc) :]
    rec("TF-IDF (pre-registered)", {f: np.argsort(-(Xa @ Xf[f]), kind="stable") for f in range(K)})
    rec("TF-IDF + dedup", rank({f: Xa @ Xf[f] for f in range(K)}))
    B = _bm25(desc, fam)
    rec("BM25", rank({f: B[f] for f in range(K)}))
    Em = np.load(f"{p}/descriptions_minilm.npy")
    Fm = embed(fam)
    rec("MiniLM", rank({f: Em @ Fm[f] for f in range(K)}))
    Eq = np.load(f"{p}/descriptions_qwen_qwen3_embedding_8b.npy")
    for tag, nm in [("", "stock"), ("_i0258be9d", "I-competent"), ("_i06c7d226", "I-demonstrated")]:
        Fq = np.load(f"{p}/families_qwen_qwen3_embedding_8b{tag}.npy")
        rec(f"dense Q3-8B ({nm})", rank({f: Eq @ Fq[f] for f in range(K)}))
        rec(
            f"hybrid ({nm})",
            {f: pool[np.argsort(-_rrf(B[f][pool], (Eq @ Fq[f])[pool]), kind="stable")] for f in range(K)},
        )
        T = np.load(f"{p}/{base}{tag}.npy")
        rec(f"SOTA+rerank ({nm})", {f: T[f][T[f] >= 0] for f in range(K)})
    rec("declared argmax", {f: np.argsort(-D[:, f], kind="stable") for f in range(K)})
    rng = np.random.default_rng(seed)
    rec("random", {f: rng.choice(len(desc), 10, replace=False) for f in range(K)})
    top1.setdefault("-- best agent in population --", []).append(float(np.mean([S[:, f].max() for f in range(K)])))
print("\nn = 10^5 specialist, 3 seeds. Every retriever ranks best-first; position 1 is what the framework reads.")
print(f"{'source':30s} {'POSITION 1':>11s} {'top-10 mean':>12s}")
print("-" * 56)
for k in sorted(top1, key=lambda k: np.mean(top1[k])):
    m = f"{np.mean(mean10[k]):12.3f}" if k in mean10 else " " * 12
    print(f"{k:30s} {np.mean(top1[k]):11.3f} {m}")
print(f"{'MIDIAN (its verified pick)':30s} {0.832:11.3f} {0.447:12.3f}   [measured in jobs/cohort_skill.py]")
