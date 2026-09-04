"""v3 figures (RESULTS_rte_v3.md): X3 learned routers vs MIDIAN variants by β at n = 100 / 1k / 10k (our benchmark),
X4 RouterEval on its own terms (μ by router and pool size), X5 every arm with liars on RouterEval's real 1,000-LLM pools.
    python scripts/v3_figs.py [X3 X4 X5]      -> $RTE_DATA/results/extra_figs/X*.png (copy to figures/)
Error bars: 95% bootstrap over seeds within fixed cells (mean of per-cell means), as in scripts/extra_figs.py."""
import os, sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__)); from extra_figs import rows, ci, COLOR, O
from rte.analyze import load

ARMS = ["oracle", "sequential_halving_peer", "midian_va", "midian_v", "midian_a", "midian", "flat_probe_argmax_online", "mlp_router", "knn_router", "fw_autogen"]
COLOR.update({"mlp_router": "#2ecc71", "knn_router": "#16a085", "knn_router_online": "#1abc9c", "fw_autogen": "#2980b9"})


def line(ax, df, label, **kw):
    g = df.groupby("beta")
    m = g.success.mean(); lo, hi = zip(*[ci(x.groupby("seed").success.mean().to_numpy()) for _, x in g])   # per-seed mean over cells, bootstrap over seeds
    ax.errorbar(m.index, m.values, yerr=[m.values - np.array(lo), np.array(hi) - m.values], marker="o", ms=3, capsize=2, label=label, color=COLOR.get(label), **kw)


def X3():
    grids = {100: ["learned_n100", "fw_live_n100"], 1000: ["learned_f1", "variants_f1", "live_f1_n1000", "live_f1_core_s6_10", "fw_live_n1000"], 10000: ["learned_n10k", "live_n10k_v2"]}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
    for ax, (n, gs) in zip(axes, grids.items()):
        df = load(gs); df = df[(df.n == n) & (df.declared_source == "self_described") & df.label.isin(ARMS)]
        for l in ARMS:
            sub = df[df.label == l]
            if len(sub): line(ax, sub, l, lw=2.2 if l == "midian_va" else 1.2, ls="--" if l in ("knn_router", "mlp_router") else "-")
        ax.set_title(f"n = {n:,}" + (" (specialist, 3 seeds)" if n == 10000 else " (3 shapes × 2 liar selections × 10 seeds)")); ax.set_xlabel("β (liar fraction)"); ax.grid(alpha=.3)
    axes[0].set_ylabel("success"); axes[2].legend(fontsize=7)
    fig.suptitle("X3  RouterBench's learned routers (dashed) inside our benchmark vs MIDIAN variants, on identical cells"); fig.savefig(f"{O}/X3_learned_vs_midian.png", dpi=300, bbox_inches="tight"); print("[X3] written")


def X4():
    R = os.environ.get("RTE_DATA", "/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte"); df = pd.read_csv(f"{R}/results/routereval_terms/rows.csv")
    df["router"] = df.router.str.replace(r" \[val-tuned: .*\]$", " [tuned]", regex=True)
    show = ["oracle (theirs)", "LinearR (theirs, ridge) [tuned]", "EmbedLLM MF (ICLR25) [tuned]", "MLPR (theirs, sklearn) [tuned]", "cluster table, full labels (C-RoBERTa-cluster / Avengers top-1) [tuned]", "PRKnn (theirs) [tuned]", "PRKnn (theirs)", "Avengers top-1 K=64 (AAAI26, full labels)", "best single model", "probe table [tuned]", "probe_cluster16_b30", "probe_cluster16_b10", "random (theirs)"]
    show = [s for s in show if s in set(df.router)]
    fig, ax = plt.subplots(figsize=(9.5, 5.5)); t = df.groupby(["router", "m"]).mu.mean().unstack(); ax.set_prop_cycle(color=plt.cm.tab20.colors)
    for s in show: ax.plot(t.columns, t.loc[s], marker="o", lw=2 if "probe" in s else 1.1, ls="--" if "probe" in s else "-", label=s)
    ax.set_xscale("log"); ax.set_xlabel("candidate pool size m (real LLMs)"); ax.set_ylabel("μ = mean test score of the routed model (12 datasets × 3 pools)"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    ax.set_title("X4  RouterEval on its own terms (validation-tuned where marked): their baselines, SOTA routers, probe table"); fig.savefig(f"{O}/X4_routereval_terms.png", dpi=300, bbox_inches="tight"); print("[X4] written")


def X5():
    df = load(["routereval_mmlu", "routereval_mmlu5k"]); arms = ["oracle", "sequential_halving_peer", "midian_va", "midian_v", "midian_a", "midian", "flat_probe_argmax_online", "mlp_router", "knn_router", "warm_start_bandit", "declared_argmax"]
    COLOR.setdefault("declared_argmax", "#5d6d7e")
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey="row")
    for j, n in enumerate([10, 100, 1000, 5000]):
        for i, ls in enumerate(["random", "low_skill_first"]):
            ax = axes[i][j]; sub = df[(df.n == n) & (df.liar_select == ls)]
            for l in arms:
                x = sub[sub.label == l]
                if len(x): line(ax, x, l, lw=2.2 if l == "midian_va" else 1.2, ls="--" if l in ("knn_router", "mlp_router") else "-")
            ax.set_title(f"n = {n:,} real LLMs, liars = {ls}" + (" (3 pools × 5 seeds)" if n < 5000 else " (leaderboard, 3 seeds)"), fontsize=10); ax.set_xlabel("β"); ax.grid(alpha=.3)
    axes[0][0].set_ylabel("success (MMLU test prompts)"); axes[1][0].set_ylabel("success"); axes[0][3].legend(fontsize=7)
    fig.suptitle("X5  Every arm with liars on RouterEval's real LLM pools (MMLU, 16 subjects, b = 3)"); fig.savefig(f"{O}/X5_routereval_liars.png", dpi=300, bbox_inches="tight"); print("[X5] written")


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["X3", "X4", "X5"]):
        try: globals()[name]()
        except Exception as e: print(f"[{name}] FAILED: {type(e).__name__}: {e}")
