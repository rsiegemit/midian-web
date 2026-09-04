"""LLMRouterBench (Li et al. 2026, Findings@ACL'26) on its own terms — TARGETS_rte_v3.md part F.
    python scripts/llmrouterbench_terms.py [--prep]      (--prep builds the score matrix + MiniLM embeddings once)

Their performance-oriented setting: 15 datasets (AIME, MATH500, MATHBench, HumanEval, MBPP, LiveCodeBench, BBH, KORBench,
Knights & Knaves, MMLU-Pro, GPQA, FinQA, MedQA, EmoryNLP, MELD) × the same 20 lightweight (~7–9B) models, per-instance
score in [0, 1] from their bench-release JSON; protocol: 70/30 train/test split repeated with their five seeds (42, 999,
2024, 2025, 3407); metrics: AvgAcc = mean over datasets of the routed accuracy, Gain@R / Gain@B = relative gain over the
random / best-single-model router, Gap@O = relative gap to the per-instance oracle. Deviation: embeddings are MiniLM
(they use gte-qwen2-7B-instruct); every embedding router here uses the same one. Routers: random, best single model
(on train), oracle, KNN / linear / MLP predictive routers, EmbedLLM MF, Avengers top-1 (clusters on all train labels), and
our probe table (b probes per dataset per model; the dataset id is the family — their routing records carry task_name,
so every router may use it; a predicted-family variant via 5-NN over probe prompts is reported too). Hyperparameters
of every router are chosen on a 20% validation slice of the train split (no failure-mode defaults). With truthful labels
every MIDIAN variant is the probe table (max-tree = argmax; nothing to audit); MIDIAN with liars is grid
`llmrouterbench_pool` (rte/backends/routereval.py, dataset=llmrouterbench)."""
import os, sys, json, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from routereval_terms import embedllm
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge

R = os.environ.get("RTE_DATA", "/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte")
B, OUT, NPZ = f"{R}/data/llmrouterbench/bench-release", f"{R}/results/llmrouterbench_terms", f"{R}/data/llmrouterbench/perf_matrix.npz"
os.makedirs(OUT, exist_ok=True)
DATASETS = ["aime", "math500", "mathbench", "humaneval", "mbpp", "livecodebench", "bbh", "korbench", "kandk", "mmlupro", "gpqa", "finqa", "medqa", "emorynlp", "meld"]
SPLIT = {"aime": "hybrid", "mmlupro": "test_1000"}                     # every other dataset has one split, "test"
SEEDS, TEST, B_PROBES = [42, 999, 2024, 2025, 3407], 0.3, [3, 10, 30]


def prep():
    """Score matrix over the 20 common models: Y (N, 20), family id per prompt, prompt text, MiniLM embeddings."""
    tabs = {}
    for d in DATASETS:
        sp = SPLIT.get(d, "test"); tabs[d] = {f.split("/")[-2]: json.load(open(f)) for f in glob.glob(f"{B}/{d}/{sp}/*/*.json")}
    models = sorted(set.intersection(*[set(t) for t in tabs.values()])); assert len(models) == 20, models
    Y, fam, prompts = [], [], []
    for k, d in enumerate(DATASETS):
        recs = {m: sorted(tabs[d][m]["records"], key=lambda r: r["index"]) for m in models}
        n = min(len(v) for v in recs.values())
        Y.append(np.stack([[r["score"] for r in recs[m][:n]] for m in models], 1)); fam += [k] * n
        prompts += [str(recs[models[0]][i]["prompt"]) for i in range(n)]
    Y = np.concatenate(Y).astype(np.float32); fam = np.array(fam)
    from rte.methods._learned import embed
    E = embed(prompts)
    np.savez(NPZ, Y=Y, fam=fam, prompts=np.array(prompts, dtype=object), E=E, models=np.array(models), datasets=np.array(DATASETS))
    print("prep:", Y.shape, "families", len(DATASETS), "models", len(models), "mean score", Y.mean().round(3))


def metrics(pick, Yte, fte, rand, bsm, orc):
    acc = np.array([Yte[fte == k][np.arange((fte == k).sum()), pick[fte == k]].mean() for k in range(len(DATASETS))])
    return {"AvgAcc": acc.mean(), "Gain@R": (acc / rand - 1).mean(), "Gain@B": (acc / bsm - 1).mean(), "Gap@O": (1 - acc / orc).mean()}


def main():
    z = np.load(NPZ, allow_pickle=True); Y, fam, E = z["Y"], z["fam"], z["E"]; N, M = Y.shape; rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed); perm = rng.permutation(N); n_te = int(TEST * N); te, tr = perm[:n_te], perm[n_te:]
        n_va = int(0.2 * len(tr)); va, tr_ = tr[:n_va], tr[n_va:]                     # validation slice of the train split for hyperparameters
        Ytr, Yva, Yte, ftr, fva, fte = Y[tr], Y[va], Y[te], fam[tr], fam[va], fam[te]; Etr, Eva, Ete = E[tr], E[va], E[te]
        per_ds = lambda pick, Yx, fx: np.array([Yx[fx == k][np.arange((fx == k).sum()), pick[fx == k]].mean() for k in range(len(DATASETS))])
        rand = np.array([Yte[fte == k].mean() for k in range(len(DATASETS))]); bsm_idx = int(np.argmax(Ytr.mean(0))); bsm = per_ds(np.full(n_te, bsm_idx), Yte, fte); orc = np.array([Yte[fte == k].max(1).mean() for k in range(len(DATASETS))])
        def add(name, pick, labels):
            rows.append({"seed": seed, "router": name, **metrics(pick, Yte, fte, rand, bsm, orc), "labelled_outcomes": labels})
        def tuned(name, cands, labels):                                          # cands: (label, pick_va, pick_te)
            best = max(cands, key=lambda c: per_ds(c[1], Yva, fva).mean()); add(f"{name} [tuned: {best[0]}]", best[2], labels)
        add("random", rng.integers(0, M, n_te), 0); add("best single model", np.full(n_te, bsm_idx), len(tr) * M); add("oracle", np.argmax(Yte, 1), 0)
        Eall = np.vstack([Eva, Ete]); nv = len(va); all_lab = len(tr) * M
        c = []
        for k in (5, 20, 50):
            nb = NearestNeighbors(n_neighbors=k, metric="cosine").fit(Etr).kneighbors(Eall)[1]; pk = np.argmax(Ytr[nb].mean(1), 1); c.append((f"k={k}", pk[:nv], pk[nv:]))
        tuned("KNN router", c, all_lab)
        c = [(f"alpha={a}", *np.split(np.argmax(Ridge(alpha=a).fit(Etr, Ytr).predict(Eall), 1), [nv])) for a in (0.1, 1.0, 10.0)]; tuned("linear router", c, all_lab)
        c = [(f"hidden={h}", *np.split(np.argmax(MLPRegressor(hidden_layer_sizes=(h,), max_iter=100, batch_size=32, random_state=0).fit(Etr, Ytr).predict(Eall), 1), [nv])) for h in (256, 1024)]; tuned("MLP router", c, all_lab)
        c = [(f"dim={dim},ep={ep}", *np.split(embedllm(Etr, Ytr, Eall, dim=dim, epochs=ep), [nv])) for dim, ep in ((64, 5), (232, 20))]; tuned("EmbedLLM MF (ICLR25)", c, all_lab)
        c = []
        for K in (15, 64, 128):
            km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(Etr); lab = km.labels_; best = np.array([np.argmax(Ytr[lab == q].mean(0)) if (lab == q).any() else bsm_idx for q in range(K)]); p = best[km.predict(Eall)]; c.append((f"K={K}", p[:nv], p[nv:]))
        tuned("Avengers top-1 (AAAI26, full labels)", c, all_lab)
        # dataset-as-family table with the given family (their records carry task_name): best model per dataset on ALL train labels
        best = np.array([np.argmax(Ytr[ftr == k].mean(0)) for k in range(len(DATASETS))]); add("dataset table, full labels", best[fte], all_lab)
        c, cp = [], []
        for b in B_PROBES:
            probe = np.concatenate([rng.choice(np.flatnonzero(ftr == k), size=min(b, (ftr == k).sum()), replace=False) for k in range(len(DATASETS))])
            est = np.stack([Ytr[probe][ftr[probe] == k].mean(0) for k in range(len(DATASETS))]); pk = np.argmax(est, 1)
            c.append((f"b={b}", pk[fva], pk[fte])); fam_pred = KNeighborsClassifier(n_neighbors=5).fit(Etr[probe], ftr[probe]).predict(Eall); cp.append((f"b={b}", pk[fam_pred[:nv]], pk[fam_pred[nv:]]))
        tuned("probe table (family given)", c, "b×15×20"); tuned("probe table (family predicted)", cp, "b×15×20")
        print(f"[seed {seed}] done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/rows.csv", index=False); df["base"] = df.router.str.replace(r" \[tuned: .*\]$", "", regex=True)
    g = df.groupby("base").agg(AvgAcc=("AvgAcc", "mean"), sd=("AvgAcc", "std"), GainR=("Gain@R", "mean"), GainB=("Gain@B", "mean"), GapO=("Gap@O", "mean")).sort_values("AvgAcc", ascending=False)
    md = ["# LLMRouterBench on its own terms (performance setting: 15 datasets × 20 lightweight models; 70/30 × their 5 seeds; MiniLM embeddings; hyperparameters tuned on a 20% train slice)", "",
          "| router | AvgAcc | sd (seeds) | Gain@R | Gain@B | Gap@O |", "|---|---|---|---|---|---|"] + [f"| {r} | {v.AvgAcc:.4f} | {v.sd:.4f} | {v.GainR:+.3f} | {v.GainB:+.3f} | {v.GapO:.3f} |" for r, v in g.iterrows()]
    open(f"{OUT}/summary.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    if "--prep" in sys.argv or not os.path.exists(NPZ): prep()
    if "--prep" not in sys.argv: main()
