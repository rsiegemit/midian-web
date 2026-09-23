"""SAMPLE condensed shortlist figures in the style of condensed A / B (one panel, one row each).
    python scripts/shortlist_condensed.py   -> figures/condensed_sample/{E_shortlists_by_n,F_shortlists_1e5}.{png,pdf,csv}
Reads figures/shortlist/live.csv (per framework x shortlist x cell: seed mean and 95% CI; written by shortlist_figs.py).
  E  live specialist, n = 10^2 .. 10^5: per n, one bar per shortlist = the MEAN over frameworks (solid honest, hatched
     beta = 0.5 low-skill cartel); oracle dotted and MIDIAN-VA (whole population) solid, as lines over each group.
  F  n = 10^5 (every shortlist ran there), shortlists sorted by the honest mean; a black dot = the BEST single framework.
  G  n = 10^5: each shortlist's gain over the pre-registered TF-IDF shortlist, paired WITHIN each framework, averaged over
     frameworks (whisker = 95% t-interval across frameworks). At 10^5 TF-IDF is the clone shortlist (0.379 for every
     framework, erratum 25), so G is the gain over that floor.
Framework numbers with an erratum-28 rerun outstanding are marked in the csv (star) and by * after the title while any remain."""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extra_figs  # noqa: F401  (installs the legend rule: best top-left, row-major)
from shortlist_figs import SOURCES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/figures/condensed_sample"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "legend.fontsize": 6,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6, "figure.constrained_layout.use": True})
REG = {"beta0": "honest", "cartel": "β=0.5 cartel"}
NAME = {k: v for k, v, _ in SOURCES}; COL = {k: c for k, _, c in SOURCES}; ORDER = [k for k, _, _ in SOURCES]
SHORT = {"tfidf": "TF-IDF (pre-reg.)", "embed": "MiniLM", "dense_icomp": "dense, I-comp", "dense_idemo": "dense, I-demo",
         "sota": "rerank", "sota_icomp": "rerank, I-comp", "sota_idemo": "rerank, I-demo", "declared": "declared top-k",
         "va_cohort": "VA cohort", "bm25": "BM25", "dense": "dense"}
MAIN = [k for k in ORDER if k not in ("bm25", "dense")]     # E: the shortlists run (or queued) at every n; BM25 / plain dense are 10^3 ablation only


def load(family="live"):
    d = pd.read_csv(f"{ROOT}/figures/shortlist/{family}.csv")
    keep = (d.dist == "specialist") if family == "live" else ((d.dist == "strong_to_weak") | (d.n == 5000))   # RouterEval: the mixed pool; 5,000 = the leaderboard
    return d[keep & d.regime.isin(list(REG))]


def summarise(d):
    """(n, regime, shortlist) -> mean over frameworks, best framework, any rerun outstanding; plus the reference lines."""
    fw = d[d.shortlist != "-"]
    s = fw.groupby(["n", "regime", "shortlist"]).agg(mean=("mean", "mean"), best=("mean", "max"), k=("arm", "nunique"),
                                                     star=("rerun_outstanding", "any")).reset_index()
    ref = d[d.shortlist == "-"].pivot_table(index=["n", "regime"], columns="arm", values="mean")
    return s, ref


def lines(ax, ref, n, x0, x1, first):
    if (n, "beta0") not in ref.index: return
    o = ref.loc[(n, "beta0")]
    ax.hlines(o.get("oracle"), x0, x1, colors="#7f8c8d", linestyles=":", lw=1.2, zorder=4, label="oracle" if first else None)
    ax.hlines(o.get("MIDIAN-VA (whole population)"), x0, x1, colors="#2ecc71", lw=1.4, zorder=4, label="MIDIAN-VA (whole population)" if first else None)


def pair(ax, x, w, q, src, label, dots=True):
    """solid honest + hatched cartel bar for one (n, shortlist); a black dot on each = the best framework. A cell not run
    yet is an EMPTY slot with * at the baseline; a bar whose framework reruns are outstanding gets * above it. Returns
    whether the labelled (honest) bar was drawn, so a legend entry is never spent on an empty slot."""
    drawn = False
    for h, r in enumerate(REG):
        xx = x + (h - 0.5) * w
        if r not in q.index:
            ax.text(xx, 0.205, "*", ha="center", va="bottom", fontsize=6, zorder=5); continue
        v = q.loc[r]
        ax.bar(xx, v["mean"], w * 0.95, color=COL[src], edgecolor="black", lw=0.3, hatch="////" if h else None, alpha=0.75 if h else 1,
               label=label if h == 0 else None, zorder=2)
        if h == 0: drawn = True
        if dots: ax.plot(xx, v["best"], "o", ms=2.2, color="black", zorder=3)
        if v["star"]: ax.text(xx, (max(v["mean"], v["best"]) if dots else v["mean"]) + 0.008, "*", ha="center", va="bottom", fontsize=6, zorder=5)
    return drawn


def fig_E(s, ref, name="E_shortlists_by_n", title=None, srcs=None, xlab="n"):
    """x = n; fixed slots per n (a cell not in yet stays visible, starred), one colour per shortlist."""
    fig, ax = plt.subplots(figsize=(7.2, 2.8)); ns = sorted(s.n.unique()); seen = set(); srcs = srcs or MAIN
    for i, n in enumerate(ns):
        w = 0.86 / (2 * len(srcs))
        for j, src in enumerate(srcs):
            q = s[(s.n == n) & (s.shortlist == src)].set_index("regime")
            if pair(ax, i + (2 * j + 1 - len(srcs)) * w, w, q, src, None if src in seen else SHORT[src], dots=False): seen.add(src)
        lines(ax, ref, n, i - 0.46, i + 0.46, i == 0)
    ax.set_xticks(range(len(ns))); ax.set_xticklabels([f"{xlab} = {n:,}" for n in ns]); finish(ax, fig, s,
        title or "E  frameworks by shortlist (live, specialist): bar = mean of frameworks; solid honest, hatched cartel; * = not in yet", name, 6)


def fig_F(s, ref, n=100000):
    """x = shortlist at one n, sorted by the honest mean; the ticks name the shortlists."""
    q = s[s.n == n]; order = q[q.regime == "beta0"].sort_values("mean", ascending=False).shortlist.tolist()
    fig, ax = plt.subplots(figsize=(7.2, 2.8)); w = 0.38
    for i, src in enumerate(order): pair(ax, i, w, q[q.shortlist == src].set_index("regime"), src, None)
    lines(ax, ref, n, -0.5, len(order) - 0.5, True)
    ax.plot([], [], "o", ms=2.5, color="black", label="best single framework")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([NAME[x] for x in order], rotation=28, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(order) - 0.4)
    finish(ax, fig, q, f"F  n = {n:,}: every shortlist, best to worst; solid honest, hatched β = 0.5 cartel", "F_shortlists_1e5", 3, inside=True)


def finish(ax, fig, s, title, name, ncol, inside=False):
    ax.set_ylim(0.2, 0.95); ax.set_ylabel("success"); ax.grid(axis="y", lw=0.3, alpha=0.35); ax.set_axisbelow(True)
    ax.set_title(title, pad=14 if inside else 6)
    if inside: ax.legend(ncol=ncol, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.82))   # the band between bars and lines
    else: ax.legend(ncol=ncol, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.07))
    fig.savefig(f"{OUT}/{name}.png", dpi=250); fig.savefig(f"{OUT}/{name}.pdf"); plt.close(fig); print(f"[{name}] written")
    s.to_csv(f"{OUT}/{name}.csv", index=False)


def fig_G(d, n=100000):
    fw = d[(d.shortlist != "-") & (d.n == n)]
    base = fw[fw.shortlist == "tfidf"].set_index(["regime", "arm"])["mean"]
    rows = []
    for (r, src), q in fw[fw.shortlist != "tfidf"].groupby(["regime", "shortlist"]):
        diff = (q.set_index(["regime", "arm"])["mean"] - base).dropna()
        k = len(diff); half = 1.96 * diff.std(ddof=1) / np.sqrt(k) if k > 1 else np.nan
        rows.append(dict(regime=r, shortlist=src, lift=diff.mean(), half=half, frameworks=k, star=bool(q.rerun_outstanding.any())))
    t = pd.DataFrame(rows); order = t[t.regime == "beta0"].sort_values("lift", ascending=False).shortlist.tolist()
    fig, ax = plt.subplots(figsize=(7.2, 2.8)); w = 0.38
    for i, src in enumerate(order):
        for h, r in enumerate(REG):
            q = t[(t.shortlist == src) & (t.regime == r)]
            if q.empty: ax.text(i + (h - 0.5) * w, 0.002, "*", ha="center", va="bottom", fontsize=6); continue
            v = q.iloc[0]; xx = i + (h - 0.5) * w
            ax.bar(xx, v.lift, w * 0.95, color=COL[src], edgecolor="black", lw=0.3, hatch="////" if h else None, alpha=0.75 if h else 1, zorder=2)
            ax.errorbar(xx, v.lift, yerr=v.half, fmt="none", ecolor="#222", elinewidth=0.5, capsize=1.2, zorder=3)
            if v.star: ax.text(xx, v.lift + (v.half if np.isfinite(v.half) else 0) + 0.004, "*", ha="center", va="bottom", fontsize=6)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([NAME[x] for x in order], rotation=28, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(order) - 0.4); ax.set_ylabel("success gain over TF-IDF"); ax.grid(axis="y", lw=0.3, alpha=0.35); ax.set_axisbelow(True)
    ax.set_title(f"G  gain over the pre-registered TF-IDF shortlist at n = {n:,}, paired within framework: solid honest, hatched cartel")
    fig.savefig(f"{OUT}/G_shortlist_lift_1e5.png", dpi=250); fig.savefig(f"{OUT}/G_shortlist_lift_1e5.pdf"); plt.close(fig)
    t.to_csv(f"{OUT}/G_shortlist_lift_1e5.csv", index=False); print("[G_shortlist_lift_1e5] written")


if __name__ == "__main__":
    d = load(); s, ref = summarise(d)
    fig_E(s, ref); fig_F(s, ref); fig_G(d)
    s, ref = summarise(load("routereval"))                                  # the re_sl_* grids bring declared, Qwen3 dense and rerank; * until they land
    fig_E(s, ref, "H_routereval_shortlists", "H  RouterEval (strong-to-weak pools, leaderboard at 5,000): mean of frameworks; hatched = cartel; * = not in yet", MAIN, "pool m")
