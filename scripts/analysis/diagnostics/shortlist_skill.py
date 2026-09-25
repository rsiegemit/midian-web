"""Diagnostic ONLY: the true skill of each shortlist source. S.npy is read here for MEASUREMENT and is never
visible to any method -- methods see the declared channel alone."""
import json, os, sys
import numpy as np
sys.path.insert(0, '/n/home02/rsiegelmann/rte')
from rte.backends import families
from rte.methods.frameworks._common import _bm25, _hash_tfidf, _rrf
from rte.methods._learned import embed

RD = os.environ["RTE_DATA"]; K = 16
fam = [families.describe(f) for f in families.FAMILIES_16]
base = "shortlist_sota_qwen_qwen3_embedding_8b_qwen_qwen3_reranker_4b_k10p50_dd"
rows, best = {}, {}
for seed in (1, 2, 3):
    p = f"{RD}/populations/specialist_n100000_K16_seed{seed}"
    desc = json.load(open(f"{p}/descriptions.json")); S = np.load(f"{p}/S.npy"); D = np.load(f"{p}/D_self_described.npy")
    first = {}
    for i, t in enumerate(desc): first.setdefault(t, i)
    pool = np.array(sorted(first.values()), dtype=np.int64)
    def sk(sel): return (float(np.mean([S[sel[f][sel[f] >= 0], f].mean() for f in range(K)])),
                         float(np.mean([S[sel[f][sel[f] >= 0], f].max() for f in range(K)])))
    def topk(score, k=10): return {f: pool[np.argsort(-score[f][pool], kind="stable")][:k] for f in range(K)}
    got = {}
    X = _hash_tfidf(desc + fam); Xa, Xf = X[:len(desc)], X[len(desc):]
    got["TF-IDF (pre-registered, no dedup)"] = {f: np.argsort(-(Xa @ Xf[f]), kind="stable")[:10] for f in range(K)}
    got["TF-IDF + dedup"] = topk({f: Xa @ Xf[f] for f in range(K)})
    Em = np.load(f"{p}/descriptions_minilm.npy"); Fm = embed(fam)
    got["MiniLM + dedup"] = topk({f: Em @ Fm[f] for f in range(K)})
    B = _bm25(desc, fam); got["BM25 + dedup"] = topk({f: B[f] for f in range(K)})
    Eq = np.load(f"{p}/descriptions_qwen_qwen3_embedding_8b.npy")
    for tag, lbl in [("", "Qwen3-8B dense (stock instr)"), ("_i0258be9d", "Qwen3-8B dense (instr: competent)"),
                     ("_i06c7d226", "Qwen3-8B dense (instr: demonstrated)")]:
        Fq = np.load(f"{p}/families_qwen_qwen3_embedding_8b{tag}.npy")
        got[lbl] = topk({f: Eq @ Fq[f] for f in range(K)})
        got[lbl.replace("dense", "hybrid")] = {f: pool[np.argsort(-_rrf(B[f][pool], (Eq @ Fq[f])[pool]), kind="stable")][:10] for f in range(K)}
        got[lbl.replace("dense", "SOTA+rerank")] = {f: np.load(f"{p}/{base}{tag}.npy")[f] for f in range(K)}
    got["declared argmax (top-10 by claim)"] = {f: np.argsort(-D[:, f], kind="stable")[:10] for f in range(K)}
    rng = np.random.default_rng(seed)
    got["random 10 from the population"] = {f: rng.choice(len(desc), 10, replace=False) for f in range(K)}
    for k, v in got.items():
        mu, mx = sk(v); rows.setdefault(k, []).append(mu); best.setdefault(k, []).append(mx)
    rows.setdefault("-- population mean --", []).append(float(S.mean())); best.setdefault("-- population mean --", []).append(float(np.mean([S[:, f].max() for f in range(K)])))
print("\nTRUE skill of the top-10 shortlist, n = 10^5 specialist, mean over 3 seeds")
print(f"{'shortlist source':42s} {'mean':>8s} {'best-in-list':>13s}")
print("-" * 66)
for k in sorted(rows, key=lambda k: np.mean(rows[k])):
    b = f"{np.mean(best[k]):13.3f}" if k in best else " " * 13
    print(f"{k:42s} {np.mean(rows[k]):8.3f} {b}")
