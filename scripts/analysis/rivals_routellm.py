"""RouteLLM's released routers vs the probe-family router, on RouterBench's own outcomes for RouteLLM's model pair,
scored with RouteLLM's metrics (TARGETS_rte_v3.md, part B).  python scripts/rivals_routellm.py

Pair: strong = gpt-4-1106-preview, weak = mistralai/mixtral-8x7b-chat (both RouterBench models: no new generations).
Routers: `bert` = routellm/bert_gpt4_augmented with RouteLLM's BERTRouter.calculate_strong_win_rate reproduced verbatim
(softmax over 3 labels, strong win-rate = 1 - P(tie or weak wins)); `random`; `optimal` (their oracle: strong where it
helps most); the probe-family router (score = est_strong - est_weak of the PREDICTED family, b probes per family, same
splits and embeddings as routerbench_terms.py); `knn_full` = k-NN on every train label of the pair (fully supervised
reference); `knn_b20` = k-NN on the probe prompts only (equal labels).  causal_llm needs gated meta-llama/Meta-Llama-3-8B
(no token here) and mf / sw_ranking need OpenAI embeddings: NOT RUN.
Metrics, RouteLLM's evaluate.py verbatim: thresholds at the router-score quantiles so strong calls span 0..100% in 10%
steps; accuracy = mean performance of the routed model; CPT(p) = strong % interpolated at p of the weak->strong gap;
AUC = trapz(accuracy, strong%/100); APGR = (AUC - weak) / (strong - weak)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from routerbench_terms import data, EMB, SEEDS, TEST, KNN_K, FAM_K, R
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

OUT = f"{R}/results/rivals_routellm"; os.makedirs(OUT, exist_ok=True)
STRONG, WEAK, B = "gpt-4-1106-preview", "mistralai/mixtral-8x7b-chat", 20
BERT_CACHE, BERT_CKPT = f"{R}/data/routerbench_bert_winrate.npy", "routellm/bert_gpt4_augmented"
PCTS = np.linspace(0, 1, 11)


def bert_win_rate(prompts):
    """RouteLLM BERTRouter, verbatim: 1 - P(label in {tie, weak wins})."""
    if os.path.exists(BERT_CACHE): return np.load(BERT_CACHE)
    import torch; from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(BERT_CKPT, num_labels=3).eval(); tok = AutoTokenizer.from_pretrained(BERT_CKPT)
    out = np.empty(len(prompts))
    with torch.no_grad():
        for i in range(0, len(prompts), 64):
            logits = model(**tok(list(prompts[i:i + 64]), return_tensors="pt", padding=True, truncation=True)).logits.numpy()
            p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            out[i:i + 64] = 1 - p[:, -2:].sum(1)
    np.save(BERT_CACHE, out); return out


def curve(score, ps, pw):
    """Route strong iff score >= threshold, thresholds at score quantiles -> (strong %, accuracy) at 0..100% strong."""
    pts = []
    for q in PCTS:
        thr = np.quantile(score, 1 - q) if q < 1 else -np.inf
        strong = score >= thr if q > 0 else np.zeros(len(score), bool)
        pts.append((100 * strong.mean(), np.where(strong, ps, pw).mean()))
    return np.array(pts)


def metrics(pts, acc_w, acc_s):
    x, y = pts[:, 0], pts[:, 1]
    cpt = {f"CPT{int(p*100)}": float(np.interp(p * (acc_s - acc_w) + acc_w, y, x)) for p in (0.2, 0.5, 0.8)}
    auc = float(np.trapezoid(y, x / 100)); span = x.max() / 100 - x.min() / 100
    return {**cpt, "AUC": auc, "APGR": (auc - acc_w * span) / ((acc_s - acc_w) * span)}


def main():
    prompts, fam, perf, cost, M = data(); e = np.load(EMB)
    fams = np.unique(fam); F = {f: i for i, f in enumerate(fams)}; fi = np.array([F[f] for f in fam])
    ps, pw = perf[:, M.index(STRONG)], perf[:, M.index(WEAK)]
    bert = bert_win_rate(prompts)
    rows, curves = [], []
    for seed, (tr, te) in enumerate(StratifiedShuffleSplit(n_splits=SEEDS, test_size=TEST, random_state=0).split(e, fam)):
        rng = np.random.default_rng(seed); jit = rng.normal(0, 1e-9, len(te))     # tie-break for few-valued scores
        probe = np.concatenate([rng.choice(tr[fi[tr] == k], size=min(B, (fi[tr] == k).sum()), replace=False) for k in range(len(fams))])
        est = np.stack([(ps - pw)[probe][fi[probe] == k].mean() for k in range(len(fams))])
        fam_pred = KNeighborsClassifier(n_neighbors=FAM_K).fit(e[probe], fi[probe]).predict(e[te])
        scores = {"bert (RouteLLM)": bert[te], "random": rng.uniform(size=len(te)), "optimal": (ps - pw)[te] + jit,
                  f"probe_family_b{B}": est[fam_pred] + jit, f"probe_family_b{B}_oraclefamily": est[fi[te]] + jit,
                  f"knn_b{B}": KNeighborsRegressor(n_neighbors=min(KNN_K, len(probe))).fit(e[probe], (ps - pw)[probe]).predict(e[te]) + jit,
                  "knn_full": KNeighborsRegressor(n_neighbors=KNN_K).fit(e[tr], (ps - pw)[tr]).predict(e[te]) + jit}
        acc_w, acc_s = pw[te].mean(), ps[te].mean()
        for name, sc in scores.items():
            pts = curve(sc, ps[te], pw[te]); rows.append({"seed": seed, "router": name, **metrics(pts, acc_w, acc_s)})
            curves += [{"seed": seed, "router": name, "strong_pct": x, "accuracy": y} for x, y in pts]
        rows.append({"seed": seed, "router": "weak only", "AUC": acc_w, "APGR": 0.0}); rows.append({"seed": seed, "router": "strong only", "AUC": acc_s, "APGR": 1.0})
        print(f"[seed {seed}] done", flush=True)
    df, cv = pd.DataFrame(rows), pd.DataFrame(curves); df.to_csv(f"{OUT}/metrics_rows.csv", index=False); cv.to_csv(f"{OUT}/curves.csv", index=False)
    agg = df.groupby("router").agg(APGR=("APGR", "mean"), APGR_sd=("APGR", "std"), CPT20=("CPT20", "mean"), CPT50=("CPT50", "mean"), CPT80=("CPT80", "mean")).sort_values("APGR", ascending=False)
    md = [f"# RouteLLM's routers vs the probe-family router on RouterBench outcomes, pair {STRONG} / {WEAK} (5 stratified splits, RouteLLM's metrics)", "",
          "NOT RUN: causal_llm (gated meta-llama/Meta-Llama-3-8B, no token), mf and sw_ranking (OpenAI embeddings).", "",
          "| router | APGR | sd (splits) | CPT 20% | CPT 50% | CPT 80% |", "|---|---|---|---|---|---|"]
    md += [f"| {r} | {v.APGR:.3f} | {v.APGR_sd:.3f} | {v.CPT20:.1f}% | {v.CPT50:.1f}% | {v.CPT80:.1f}% |" for r, v in agg.iterrows()]
    open(f"{OUT}/summary.md", "w").write("\n".join(md) + "\n"); print("\n".join(md)); fig(cv, agg, df)


def fig(cv, agg, df):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    f, ax = plt.subplots(figsize=(7, 5))
    for r in [c for c in agg.index if c not in ("weak only", "strong only")]:
        g = cv[cv.router == r].groupby("strong_pct").accuracy.mean(); ax.plot(g.index, g.values, marker=".", lw=1.6 if "probe" in r else 1.0, label=f"{r} (APGR {agg.loc[r, 'APGR']:.2f})")
    w, s = df[df.router == "weak only"].AUC.mean(), df[df.router == "strong only"].AUC.mean()
    ax.axhline(w, color="grey", ls="--", label=f"{WEAK} {w:.3f}"); ax.axhline(s, color="red", ls="--", label=f"{STRONG} {s:.3f}")
    ax.set_xlabel("Strong model calls (%)"); ax.set_ylabel("Performance"); ax.set_title("RouteLLM protocol on RouterBench outcomes (mean of 5 splits)"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    f.savefig(f"{OUT}/X2_routellm_pair.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
