"""RouterBench on its own terms (TARGETS_rte_v3.md, part A).  python scripts/routerbench_terms.py [--embed]

Their data (36,497 prompts x 11 models, per-prompt score and $ cost), their protocol (fit on a stratified 70/30 train
split, route each held-out prompt, sweep a willingness-to-pay lambda, score = mean test performance vs mean test cost,
summary = AIQ), their baselines (oracle, zero router = hull of the single models, KNN and MLP predictive routers).
One deviation for every router alike: prompt embeddings are a local all-MiniLM-L6-v2, not OpenAI's.
Our arm: the probe-family router (b probe prompts per eval_name, every model looked up on them, family PREDICTED at test
time by k-NN over the probe prompts).  Plain MIDIAN at n = 11 is a max-tree over the same scores (asserted == argmax).
AIQ here = mean over the cost range [cheapest model, dearest model] of the best performance reachable at that cost
(step envelope of the router's lambda sweep), i.e. area under the non-decreasing quality-vs-cost curve, normalised."""
import os, sys, json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor

R = os.environ.get("RTE_DATA", "/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte")
OUT = f"{R}/results/routerbench_terms"; os.makedirs(OUT, exist_ok=True)
EMB = f"{R}/data/routerbench_emb_minilm.npy"
MIN_FAMILY, SEEDS, TEST, KNN_K, FAM_K, R_MIDIAN = 60, 5, 0.3, 20, 5, 10
LAMBDAS = np.concatenate([[0.0], np.logspace(-1, 3, 25)])          # $ per unit of performance; 0 = quality only
B_PROBES = [5, 10, 20, 50]


def data():
    d = pd.read_pickle(f"{R}/data/routerbench_0shot.pkl")
    keep = d.eval_name.map(d.eval_name.value_counts()) >= MIN_FAMILY; d = d[keep].reset_index(drop=True)
    M = [c for c in d.columns if c + "|total_cost" in d.columns]
    return d.prompt.astype(str).to_numpy(), d.eval_name.to_numpy(), d[M].to_numpy(float), d[[m + "|total_cost" for m in M]].to_numpy(float), M


def embed(prompts):
    from sentence_transformers import SentenceTransformer
    e = SentenceTransformer("all-MiniLM-L6-v2", device="cpu").encode(list(prompts), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    np.save(EMB, e.astype(np.float32)); return e


def midian_tree_pick(score, r=R_MIDIAN):
    """Plain MIDIAN's pick over n agents with scores (n,): cohorts of r by index, best per cohort promoted, until one."""
    idx = np.arange(len(score))
    while len(idx) > 1:
        idx = np.array([idx[c][np.argmax(score[idx[c]])] for c in np.array_split(idx, max(1, int(np.ceil(len(idx) / r))))])
    return int(idx[0])


def sweep(pred, perf, cost, price):
    """lambda sweep -> (mean cost, mean performance) per lambda; pick = argmax pred - lambda*price."""
    pts = []
    for lam in LAMBDAS:
        pick = np.argmax(pred - lam * price[None, :], axis=1)
        pts.append((cost[np.arange(len(pick)), pick].mean(), perf[np.arange(len(pick)), pick].mean()))
    return np.array(pts)


def aiq(pts, c_lo, c_hi, hull=False):
    """Area under the non-decreasing quality-vs-cost envelope over [c_lo, c_hi], normalised to [0, 1] of the cost range."""
    grid = np.linspace(c_lo, c_hi, 2001)
    if hull:                                                       # zero router: linear interpolation between single models
        o = np.argsort(pts[:, 0]); c, q = pts[o, 0], np.maximum.accumulate(pts[o, 1])
        return float(np.interp(grid, c, q, left=0.0).mean())
    q = np.array([pts[pts[:, 0] <= g, 1].max() if (pts[:, 0] <= g).any() else 0.0 for g in grid])
    return float(q.mean())


def main():
    prompts, fam, perf, cost, M = data()
    e = embed(prompts) if "--embed" in sys.argv or not os.path.exists(EMB) else np.load(EMB)
    fams = np.unique(fam); F = {f: i for i, f in enumerate(fams)}; fi = np.array([F[f] for f in fam])
    rows, curves = [], []
    for seed, (tr, te) in enumerate(StratifiedShuffleSplit(SEEDS, test_size=TEST, random_state=0).split(e, fam)):
        rng = np.random.default_rng(seed)
        price = cost[tr].mean(0)                                   # each model's price = its mean train cost (known ahead)
        c_lo, c_hi = price.min(), price.max()
        P, C = perf[te], cost[te]
        def add(name, pred, extra=None, hull=False):
            pts = sweep(pred, P, C, price) if not hull else pred
            rows.append({"seed": seed, "router": name, "aiq": aiq(pts, c_lo, c_hi, hull), "quality_at_lambda0": pts[0, 1] if not hull else pts[:, 1].max(),
                         "labelled_outcomes": extra if extra is not None else len(tr) * len(M)})
            for lam, (c, q) in zip(LAMBDAS if not hull else [np.nan] * len(pts), pts): curves.append({"seed": seed, "router": name, "lam": lam, "cost": c, "perf": q})
        # oracle: cheapest correct model per prompt (their definition); one point
        best = P.max(1, keepdims=True); ocost = np.where(P >= best, C, np.inf); opick = ocost.argmin(1)
        rows.append({"seed": seed, "router": "oracle", "aiq": np.nan, "quality_at_lambda0": P[np.arange(len(te)), opick].mean(), "labelled_outcomes": np.nan})
        curves.append({"seed": seed, "router": "oracle", "lam": np.nan, "cost": C[np.arange(len(te)), opick].mean(), "perf": P[np.arange(len(te)), opick].mean()})
        # zero router: single models and their hull
        add("zero_router", np.c_[C.mean(0), P.mean(0)], extra=0, hull=True)
        # RouterBench's predictive routers, full training split
        add("knn", KNeighborsRegressor(KNN_K).fit(e[tr], perf[tr]).predict(e[te]))
        add("mlp", MLPRegressor((128,), max_iter=200, random_state=seed).fit(e[tr], perf[tr]).predict(e[te]))
        # probe-family router: b probes per family from the train split
        for b in B_PROBES:
            probe = np.concatenate([rng.choice(tr[fi[tr] == k], size=min(b, (fi[tr] == k).sum()), replace=False) for k in range(len(fams))])
            est = np.stack([perf[probe][fi[probe] == k].mean(0) for k in range(len(fams))])     # (F, M) per-family accuracy
            fam_pred = KNeighborsClassifier(FAM_K).fit(e[probe], fi[probe]).predict(e[te])
            add(f"probe_family_b{b}", est[fam_pred], extra=len(probe) * len(M))
            add(f"probe_family_b{b}_oraclefamily", est[fi[te]], extra=len(probe) * len(M))
            add(f"knn_b{b}", KNeighborsRegressor(min(KNN_K, len(probe))).fit(e[probe], perf[probe]).predict(e[te]), extra=len(probe) * len(M))   # same labels, their router
            if b == 20:                                            # T3-3: MIDIAN's max-tree == argmax on the same scores
                sc = est[fam_pred] - LAMBDAS[10] * price[None, :]
                assert all(midian_tree_pick(s) == int(np.argmax(s)) for s in sc[:2000]), "MIDIAN tree pick != argmax"
        print(f"[seed {seed}] done", flush=True)
    df, cv = pd.DataFrame(rows), pd.DataFrame(curves)
    df.to_csv(f"{OUT}/aiq_rows.csv", index=False); cv.to_csv(f"{OUT}/curves.csv", index=False)
    agg = df.groupby("router").agg(aiq=("aiq", "mean"), aiq_sd=("aiq", "std"), q0=("quality_at_lambda0", "mean"), labelled=("labelled_outcomes", "mean")).sort_values("aiq", ascending=False)
    md = ["# RouterBench on its own terms (5 stratified 70/30 splits; local MiniLM embeddings for every router)", "",
          f"families {len(fams)}, prompts {len(prompts)}, models {len(M)}; AIQ = normalised area under the non-decreasing quality-vs-cost envelope over the single-model cost range", "",
          "| router | AIQ | sd (splits) | quality at λ=0 | labelled outcomes used |", "|---|---|---|---|---|"]
    md += [f"| {r} | {v.aiq:.4f} | {v.aiq_sd:.4f} | {v.q0:.4f} | {v.labelled:,.0f} |" for r, v in agg.iterrows()]
    open(f"{OUT}/summary.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
    fig(cv, agg)


def fig(cv, agg):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    show = ["zero_router", "knn", "mlp", "probe_family_b5", "probe_family_b20", "probe_family_b50", "probe_family_b20_oraclefamily", "knn_b20"]
    f, ax = plt.subplots(figsize=(7.5, 5))
    for r in show:
        g = cv[cv.router == r].groupby("lam" if r != "zero_router" else "cost")[["cost", "perf"]].mean().sort_values("cost")
        ax.plot(g.cost, g.perf, marker="o", ms=3, lw=1.4 if "probe" in r else 1.0, label=f"{r} (AIQ {agg.loc[r, 'aiq']:.3f})")
    o = cv[cv.router == "oracle"][["cost", "perf"]].mean(); ax.plot(o.cost, o.perf, "k*", ms=12, label="oracle (cheapest correct)")
    ax.set_xscale("log"); ax.set_xlabel("mean $ per prompt (test)"); ax.set_ylabel("mean performance (test)"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    ax.set_title("RouterBench protocol: quality vs cost over the willingness-to-pay sweep (mean of 5 splits)")
    f.savefig(f"{OUT}/X1_routerbench_terms.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
