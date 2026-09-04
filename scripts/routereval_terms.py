"""RouterEval (Huang et al., EMNLP 2025 Findings) on its own terms — TARGETS_rte_v3.md part D1.
    python scripts/routereval_terms.py <dataset> [<dataset> ...]      (all 12 when none given)

Their data ($RTE_DATA/data/routereval/router_dataset/<ds>_router_dataset.pkl): candidate pools of m in {10, 100, 1000}
REAL LLMs (three pool configs: all_strong / all_weak / strong_to_weak), binary score of every candidate on every prompt,
8:1:1 train/val/test split, RoBERTa prompt embeddings (768-d), prompt text. Their metrics: mu = mean test score of the
routed model, V_B = mu / best single model (on test), V_R = mu / a reference model's accuracy (their acc_ref_dict);
averaged over the three configs as their test_router.py does. Their baselines, re-implemented line for line from
router/*: PRKnn (k = 5 cosine kNN over train embeddings, mean of neighbours' scores, argmax), C-RoBERTa-cluster
(KMeans K = 3 on train embeddings, best model per cluster, nearest centroid at test), R_o (oracle p = 1, random
p = 0), MLPR / LinearR (their torch FNN hidden 256 / linear; here sklearn MLPRegressor(256) / Ridge, batch 32 instead of
their batch 1 — the one deviation, stated). RoBERTa-MLC (fine-tuned RoBERTa) and A3M are not run (GPU / not needed).

Our arm, the **probe-family router**: b probe prompts per family drawn from the train split, every candidate looked up
on them, estimate = per-(candidate, family) accuracy, pick = argmax for the test prompt's PREDICTED family. Families:
`cluster{K}` = KMeans(K) over train embeddings (unsupervised; nearest centroid at test; K = 3 is exactly their cluster
baseline's partition, so cluster3 vs C-RoBERTa-cluster isolates the label budget) and, on mmlu only, `subject` = the
MMLU subject named in the prompt (57; predicted at test by 5-NN over the probe prompts' embeddings). Plain MIDIAN over
m truthful candidates is a max-tree over the same estimates and picks identically (asserted, as in part A); the
report-channel / liar dimension is part D2 (the routereval backend inside our benchmark), not here."""
import os, re, sys, pickle, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
import torch


def embedllm(Etr, Ytr, Ete, dim=64, epochs=5, batch=2048, seed=0):
    """EmbedLLM (Zhuang et al., ICLR 2025) router: model embedding table v_m, query projection v_q = W e_q, score
    s = sigmoid(w . (v_m * v_q)), BCE on every (prompt, model) train entry; pick argmax_m s. Their defaults are dim 232,
    50 epochs, batch 2048 (mf.py); here dim 64 and 5 epochs on CPU (11M entries at m = 1000) — stated deviation."""
    torch.manual_seed(seed); n_q, m = Ytr.shape
    Vm = torch.nn.Parameter(torch.randn(m, dim) * 0.1); W = torch.nn.Linear(Etr.shape[1], dim); w = torch.nn.Linear(dim, 1)
    opt = torch.optim.Adam([Vm, *W.parameters(), *w.parameters()], lr=1e-3)
    Xq, Y = torch.tensor(Etr, dtype=torch.float32), torch.tensor(Ytr, dtype=torch.float32)
    qi, mi = np.divmod(np.arange(n_q * m), m); rng = np.random.default_rng(seed)
    for _ in range(epochs):
        perm = rng.permutation(n_q * m)
        for lo in range(0, len(perm), batch):
            b = perm[lo:lo + batch]; q, mm = torch.tensor(qi[b]), torch.tensor(mi[b])
            logit = w(W(Xq[q]) * Vm[mm]).squeeze(1); loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, Y[q, mm])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Vq = W(torch.tensor(Ete, dtype=torch.float32))                                  # (n_te, dim)
        return (w.weight[0] * Vq[:, None, :] * Vm[None, :, :]).sum(-1).argmax(1).numpy()   # (n_te, m) scores -> argmax
sys.path.insert(0, os.path.dirname(__file__)); from routerbench_terms import midian_tree_pick

R = os.environ.get("RTE_DATA", "/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte")
DATA, OUT = f"{R}/data/routereval/router_dataset", f"{R}/results/routereval_terms"; os.makedirs(OUT, exist_ok=True)
DATASETS = ["arc", "bbh", "gpqa", "gsm8k", "harness_truthfulqa_mc_0", "hellaswag", "ifeval", "math", "mmlu_pro", "mmlu", "musr", "winogrande"]
ACC_REF = {"arc": 0.852, "hellaswag": 0.953, "mmlu": 0.864, "harness_truthfulqa_mc_0": 0.669, "winogrande": 0.875, "gsm8k": 0.92,
           "ifeval": 0.7689, "bbh": 0.8303, "gpqa": 0.397, "math": 0.4, "musr": 0.699, "mmlu_pro": 0.637}   # their acc_ref_dict
CONFIGS, B_PROBES, SEED = ["all_strong", "all_weak", "strong_to_weak"], [3, 10, 30], 0
SUBJECT = re.compile(r"questions \(with answers\) about (.+?)\.")


def score(pick, Yte):
    return float(Yte[np.arange(len(pick)), pick].mean())


def probe_family(fam_tr, fam_te, Ytr, b, rng):
    """b probes per family per candidate from train; estimate (F, m); pick argmax of the test prompt's family. Returns picks, labels used."""
    F = fam_tr.max() + 1; est = np.zeros((F, Ytr.shape[1])); used = 0
    for k in range(F):
        idx = np.flatnonzero(fam_tr == k); idx = rng.choice(idx, size=min(b, len(idx)), replace=False)
        est[k] = Ytr[idx].mean(0) if len(idx) else Ytr.mean(0); used += len(idx) * Ytr.shape[1]
    return np.argmax(est[fam_te], axis=1), used, est


def tuned(name, cands, Yva, Yte):
    """Pick the candidate with the best VALIDATION μ (their val split), report its TEST μ. cands = [(label, pick_va, pick_te, labels_used)]."""
    best = max(cands, key=lambda c: score(c[1], Yva)); return f"{name} [val-tuned: {best[0]}]", best[2], best[3]


def run(ds):
    d = pickle.load(open(f"{DATA}/{ds}_router_dataset.pkl", "rb"))
    Etr, Ete = np.asarray(d["embedding"]["train_embed"]), np.asarray(d["embedding"]["test_embed"]); Eva = np.asarray(d["embedding"]["val_embed"])
    Ptr, Pte = d["prompt"]["train_prompt"], d["prompt"]["test_prompt"]
    rng = np.random.default_rng(SEED); rows = []
    # families that do not depend on the pool: unsupervised clusters, and the MMLU subject
    fams = {}
    fams_va = {}
    for K in (3, 16, 64, 128):
        km = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(Etr); fams[f"cluster{K}"] = (km.labels_, km.predict(Ete)); fams_va[K] = km.predict(Eva)
    if ds == "mmlu":
        subj = lambda P: np.array([(SUBJECT.search(str(p)) or [None, "?"])[1] for p in P]); s_tr, s_te = subj(Ptr), subj(Pte)
        names = {s: i for i, s in enumerate(sorted(set(s_tr)))}; fams["subject"] = (np.array([names[s] for s in s_tr]), None)   # test family predicted below
    for m, cfgs in d["hard"].items():
        for cfg in CONFIGS:
            dd = cfgs[cfg]["data"]; key = "train_label" if ds == "harness_truthfulqa_mc_0" else "train_score"
            Ytr, Yte, Yva = np.asarray(dd[key], float), np.asarray(dd["test_score"], float), np.asarray(dd["val_score"], float)
            bsm = Yte.mean(0).max(); n_tr = Ytr.size
            def add(name, pick, labels):
                rows.append({"dataset": ds, "m": m, "config": cfg, "router": name, "mu": score(pick, Yte), "V_B": score(pick, Yte) / bsm,
                             "V_R": score(pick, Yte) / ACC_REF[ds], "labelled_outcomes": labels, "bsm": bsm, "oracle": float(Yte.max(1).mean())})
            # --- their baselines
            nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(Etr); _, nb = nn.kneighbors(Ete)
            add("PRKnn (theirs)", np.argmax(Ytr[nb].mean(1), 1), n_tr)
            lab_tr, lab_te = fams["cluster3"]; best = np.array([np.argmax(Ytr[lab_tr == c].mean(0)) if (lab_tr == c).any() else 0 for c in range(3)])
            add("C-RoBERTa-cluster (theirs)", best[lab_te], n_tr)
            add("LinearR (theirs, ridge)", np.argmax(Ridge(alpha=1.0).fit(Etr, Ytr).predict(Ete), 1), n_tr)
            add("MLPR (theirs, sklearn)", np.argmax(MLPRegressor(hidden_layer_sizes=(256,), max_iter=100, batch_size=32, random_state=SEED).fit(Etr, Ytr).predict(Ete), 1), n_tr)
            add("random (theirs)", rng.integers(0, Ytr.shape[1], len(Yte)), 0)
            add("oracle (theirs)", np.argmax(Yte, 1), 0)
            add("best single model", np.full(len(Yte), int(np.argmax(Ytr.mean(0)))), n_tr)          # best on train, applied to test
            # --- SOTA (LLMRouterBench 2026 leaders with released code): Avengers top-1 = per-cluster accuracy ranking on ALL train labels
            for K in (16, 64):
                lab_tr, lab_te = fams[f"cluster{K}"]; best = np.array([np.argmax(Ytr[lab_tr == c].mean(0)) if (lab_tr == c).any() else int(np.argmax(Ytr.mean(0))) for c in range(K)])
                add(f"Avengers top-1 K={K} (AAAI26, full labels)", best[lab_te], n_tr)
            add("EmbedLLM MF (ICLR25, dim64 5ep)", embedllm(Etr, Ytr, Ete), n_tr)
            # --- the same routers with hyperparameters selected on THEIR validation split (no failure-mode defaults)
            Eall = np.vstack([Eva, Ete]); nv = len(Eva)
            c = []
            for k in (5, 20, 50):
                nb = NearestNeighbors(n_neighbors=k, metric="cosine").fit(Etr).kneighbors(Eall)[1]; pk = np.argmax(Ytr[nb].mean(1), 1); c.append((f"k={k}", pk[:nv], pk[nv:], n_tr))
            add(*tuned("PRKnn (theirs)", c, Yva, Yte))
            c = []
            for K in (3, 16, 64, 128):
                lab_tr, lab_te = fams[f"cluster{K}"]; best = np.array([np.argmax(Ytr[lab_tr == q].mean(0)) if (lab_tr == q).any() else int(np.argmax(Ytr.mean(0))) for q in range(K)])
                c.append((f"K={K}", best[fams_va[K]], best[lab_te], n_tr))
            add(*tuned("cluster table, full labels (C-RoBERTa-cluster / Avengers top-1)", c, Yva, Yte))
            c = []
            for a in (0.1, 1.0, 10.0):
                pr = Ridge(alpha=a).fit(Etr, Ytr).predict(Eall); c.append((f"alpha={a}", np.argmax(pr[:nv], 1), np.argmax(pr[nv:], 1), n_tr))
            add(*tuned("LinearR (theirs, ridge)", c, Yva, Yte))
            c = []
            for h in (256, 1024):
                pr = MLPRegressor(hidden_layer_sizes=(h,), max_iter=100, batch_size=32, random_state=SEED).fit(Etr, Ytr).predict(Eall); c.append((f"hidden={h}", np.argmax(pr[:nv], 1), np.argmax(pr[nv:], 1), n_tr))
            add(*tuned("MLPR (theirs, sklearn)", c, Yva, Yte))
            c = []
            for dim, ep in ((64, 5), (232, 20)):
                pk = embedllm(Etr, Ytr, Eall, dim=dim, epochs=ep); c.append((f"dim={dim},ep={ep}", pk[:nv], pk[nv:], n_tr))
            add(*tuned("EmbedLLM MF (ICLR25)", c, Yva, Yte))
            c = []
            for fname in ("cluster3", "cluster16"):
                ftr, fte = fams[fname]; fva = fams_va[int(fname[7:])]
                for b in B_PROBES:
                    pk_te, used, est = probe_family(ftr, fte, Ytr, b, np.random.default_rng(SEED)); pk_va = np.argmax(est[fva], axis=1); c.append((f"{fname},b={b}", pk_va, pk_te, used))
            add(*tuned("probe table", c, Yva, Yte))
            # --- ours
            for fname, (ftr, fte) in fams.items():
                if fname == "cluster64": continue                                     # Avengers' partition; probe tables use 3 / 16 / subject
                if fte is None: fte = KNeighborsClassifier(n_neighbors=5).fit(Etr, ftr).predict(Ete)   # subject predicted from text embeddings
                for b in B_PROBES:
                    pick, used, est = probe_family(ftr, fte, Ytr, b, rng); add(f"probe_{fname}_b{b}", pick, used)
                    if b == 10 and fname == "cluster16":
                        sc = est[fte]; assert all(midian_tree_pick(s) == int(np.argmax(s)) for s in sc[:500]), "MIDIAN tree != argmax"
            print(f"[{ds}] m={m} {cfg} done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/{ds}.csv", index=False); return df


def summarise():
    df = pd.concat([pd.read_csv(f"{OUT}/{ds}.csv") for ds in DATASETS if os.path.exists(f"{OUT}/{ds}.csv")])
    df.to_csv(f"{OUT}/rows.csv", index=False)
    md = ["# RouterEval on its own terms (hard setting; mean over the three pool configs, as their harness reports)", ""]
    df["base"] = df.router.str.replace(r" \[val-tuned: .*\]$", " [val-tuned]", regex=True)
    for m in sorted(df.m.unique()):
        t = df[df.m == m].groupby("base").agg(mu=("mu", "mean"), V_B=("V_B", "mean"), V_R=("V_R", "mean"), labels=("labelled_outcomes", "mean"), n=("mu", "size")).sort_values("mu", ascending=False)
        md += [f"## m = {m} candidates ({df[df.m == m].dataset.nunique()} datasets × 3 configs)", "", "| router | μ | V_B | V_R | labelled outcomes | runs |", "|---|---|---|---|---|---|"]
        md += [f"| {r} | {v.mu:.4f} | {v.V_B:.3f} | {v.V_R:.3f} | {v.labels:,.0f} | {int(v.n)} |" for r, v in t.iterrows()] + [""]
    piv = df[df.m == 1000].pivot_table(index="dataset", columns="router", values="mu")
    keep = [c for c in ["oracle (theirs)", "best single model", "PRKnn (theirs)", "C-RoBERTa-cluster (theirs)", "LinearR (theirs, ridge)", "MLPR (theirs, sklearn)", "Avengers top-1 K=64 (AAAI26, full labels)", "EmbedLLM MF (ICLR25, dim64 5ep)", "probe_cluster16_b10", "probe_cluster16_b30", "probe_subject_b30", "random (theirs)"] if c in piv]
    # val-tuned rows carry their chosen config in the name: pool them by base name for the per-dataset table
    df["base"] = df.router.str.replace(r" \[val-tuned: .*\]$", " [val-tuned]", regex=True)
    pv2 = df[df.m == 1000].pivot_table(index="dataset", columns="base", values="mu"); tuned_cols = [c for c in pv2.columns if "[val-tuned]" in c]
    if tuned_cols:
        md += ["", "## m = 1000, per dataset (μ), hyperparameters selected on their validation split", "", "| dataset | " + " | ".join(tuned_cols) + " |", "|---" * (len(tuned_cols) + 1) + "|"]
        md += ["| " + ds + " | " + " | ".join("" if pd.isna(pv2.loc[ds, c]) else f"{pv2.loc[ds, c]:.3f}" for c in tuned_cols) + " |" for ds in pv2.index]
    md += ["## m = 1000, per dataset (μ)", "", "| dataset | " + " | ".join(keep) + " |", "|---" * (len(keep) + 1) + "|"]
    md += ["| " + ds + " | " + " | ".join("" if pd.isna(piv.loc[ds, c]) else f"{piv.loc[ds, c]:.3f}" for c in keep) + " |" for ds in piv.index]
    open(f"{OUT}/summary.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    if sys.argv[1:] == ["--summary"]: summarise()
    else:
        for ds in (sys.argv[1:] or DATASETS): run(ds)
        summarise()
