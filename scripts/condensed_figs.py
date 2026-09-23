"""SAMPLE condensed replacements for the 48 figures/bars files -- four figures, one panel and one row each.
    python scripts/condensed_figs.py      -> figures/condensed_sample/{A,B,C,D}_*.{png,pdf,csv}
Reads only figures/bars/<family>.csv (per-arm seed means and 95% seed-bootstrap CIs, written by bar_figs.py).
  A  live headline: n = 10^2..10^5 (specialist), a few arms, solid = honest, hatched = beta = 0.5 low-skill cartel.
  B  every family on one axis: success / oracle, same arms, filled = honest, hollow = cartel.
  C  scripts/paired_gaps.py: MIDIAN-VA minus each FIXED rival, seed-paired, every condition.
  D  appendix heatmap: every arm x every (family, n, regime) cell, success / oracle.
"Best learned router" / "best bandit" are chosen PER CONDITION (the most flattering choice for the rivals); the chosen arm
is written to the csv. Frameworks are not drawn (figures/shortlist). The do-not-add list applies, and with it the
TEMPORARY extra_figs.HIDE_HALVING switch."""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extra_figs import excluded

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BARS, OUT = f"{ROOT}/figures/bars", f"{ROOT}/figures/condensed_sample"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "legend.fontsize": 6,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6, "figure.constrained_layout.use": True})
LEARNED = ["knn_router", "knn_router_online", "mlp_router", "flat_nsw_router", "cluster_head_router", "disrouter_cascade"]
BANDIT = ["ucb_per_family", "thompson_per_family", "warm_start_bandit", "linucb_honest", "trueskill_per_family"]
ARMS = [("midian_va", "MIDIAN-VA", "#2ecc71"), ("midian", "MIDIAN", "#c0392b"), ("flat_probe_argmax_online", "flat probe argmax (online)", "#3498db"),
        ("best_learned", "best learned router", "#ff7f0e"), ("best_bandit", "best bandit", "#9467bd"),
        ("declared_argmax", "declared argmax", "#5d6d7e"), ("random", "random", "#bbbbbb")]
PRIMARY = {"live": "specialist", "routereval": "strong_to_weak"}          # one population shape per family in A / B / D
FAMILY = {"live": "live", "bernoulli": "bernoulli", "replay": "RouterBench replay", "routereval": "RouterEval", "llmrouterbench": "LLMRouterBench"}
REG = {"beta0": "honest", "cartel": "β=0.5 cartel"}


def load():
    d = pd.concat([pd.read_csv(f"{BARS}/{f}.csv") for f in FAMILY], ignore_index=True)
    return d[~d.label.astype(str).str.startswith("fw_")]


def cells(d):
    """(family, group, n, regime) -> {arm key -> (mean, lo, hi, chosen label)} for the arms in ARMS, plus the oracle."""
    out = {}
    for key, q in d.groupby(["family", "group", "n", "regime"]):
        s = q.set_index("label")
        if "oracle" not in s.index: continue
        row = {"oracle": (s.at["oracle", "mean"], s.at["oracle", "ci_lo"], s.at["oracle", "ci_hi"], "oracle")}
        for k, pool in (("best_learned", LEARNED), ("best_bandit", BANDIT)):
            have = [a for a in pool if a in s.index and not excluded(a)]
            if have: b = s.loc[have, "mean"].idxmax(); row[k] = (s.at[b, "mean"], s.at[b, "ci_lo"], s.at[b, "ci_hi"], b)
        for k, _, _ in ARMS:
            if k in s.index and not excluded(k): row[k] = (s.at[k, "mean"], s.at[k, "ci_lo"], s.at[k, "ci_hi"], k)
        out[key] = row
    return out


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=250); fig.savefig(f"{OUT}/{name}.pdf"); plt.close(fig); print(f"[{name}] written")


def fig_A(C):
    ns = sorted(n for (f, g, n, r) in C if f == "live" and g == "specialist" and r == "beta0")
    arms = [a for a in ARMS if any(a[0] in C.get(("live", "specialist", n, r), {}) for n in ns for r in REG)]
    w = 0.84 / (2 * len(arms)); fig, ax = plt.subplots(figsize=(7.2, 2.8)); rec = []
    for i, n in enumerate(ns):
        for j, (k, lbl, col) in enumerate(arms):
            for h, r in enumerate(REG):
                v = C.get(("live", "specialist", n, r), {}).get(k)
                if v is None: continue
                x = i + (2 * j + h - (2 * len(arms) - 1) / 2) * w
                ax.bar(x, v[0], w * 0.95, color=col, edgecolor="black", lw=0.3, hatch="////" if h else None, alpha=0.75 if h else 1,
                       label=(lbl if (i == 0 and h == 0) else None), zorder=2)
                ax.errorbar(x, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="none", ecolor="#222", elinewidth=0.4, capsize=0.8, zorder=3)
                rec.append(dict(n=n, regime=REG[r], arm=lbl, chosen=v[3], mean=v[0], ci_lo=v[1], ci_hi=v[2]))
        o = C.get(("live", "specialist", n, "beta0"), {}).get("oracle")
        if o: ax.hlines(o[0], i - 0.45, i + 0.45, colors="#7f8c8d", linestyles=":", lw=1.2, zorder=4, label="oracle" if i == 0 else None)
    ax.set_xticks(range(len(ns))); ax.set_xticklabels([f"n = {n:,}" for n in ns]); ax.set_ylim(0.2, 0.95); ax.set_ylabel("success")
    ax.grid(axis="y", lw=0.3, alpha=0.35); ax.set_axisbelow(True)
    ax.set_title("A  live RTE, specialist: solid = honest, hatched = β = 0.5 low-skill cartel (b = 3)")
    ax.legend(ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.07)); save(fig, "A_live_headline")
    pd.DataFrame(rec).to_csv(f"{OUT}/A_live_headline.csv", index=False)


def primary(f, g): return g == PRIMARY.get(f, g)


def fig_B(C):
    conds = sorted({(f, n) for (f, g, n, r) in C if primary(f, g) and r in REG}, key=lambda t: (list(FAMILY).index(t[0]), t[1]))
    group = {f: PRIMARY.get(f) or g for (f, g, n, r) in C}                  # families with one shape use it
    fig, ax = plt.subplots(figsize=(7.2, 2.8)); rec = []
    for j, (k, lbl, col) in enumerate(ARMS):
        for h, r in enumerate(REG):
            xs, ys, lo, hi = [], [], [], []
            for i, (f, n) in enumerate(conds):
                c = C.get((f, group[f], n, r), {})
                if k not in c or "oracle" not in c: continue
                o = c["oracle"][0]; v = c[k]
                xs.append(i + (j - (len(ARMS) - 1) / 2) * 0.1); ys.append(v[0] / o); lo.append((v[0] - v[1]) / o); hi.append((v[2] - v[0]) / o)
                rec.append(dict(family=FAMILY[f], n=n, regime=REG[r], arm=lbl, chosen=v[3], success_over_oracle=v[0] / o))
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=3, color=col, mfc=col if h == 0 else "white", mec=col, elinewidth=0.4,
                        capsize=0, label=lbl if h == 0 else None, zorder=3)
    ax.axhline(1.0, color="#7f8c8d", ls=":", lw=1.2, label="oracle")
    edges = [i for i in range(1, len(conds)) if conds[i][0] != conds[i - 1][0]]
    for e in edges: ax.axvline(e - 0.5, color="#999", lw=0.5)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels([f"{FAMILY[f]} {n:,}" for f, n in conds], rotation=40, ha="right", rotation_mode="anchor")
    ax.set_ylabel("success / oracle"); ax.grid(axis="y", lw=0.3, alpha=0.35); ax.set_axisbelow(True)
    ax.plot([], [], "o", mfc="white", mec="black", ms=3, ls="", label="hollow = β=0.5 cartel")
    ax.set_title("B  every experiment family on one axis (success / oracle): filled = honest, hollow = β = 0.5 low-skill cartel")
    ax.legend(ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.07)); save(fig, "B_all_families")
    pd.DataFrame(rec).to_csv(f"{OUT}/B_all_families.csv", index=False)


def fig_D(d):
    q = d[np.array([primary(f, g) for f, g in zip(d.family, d.group)]) & d.regime.isin(list(REG)) & ~d.label.map(excluded)]
    o = q[q.label == "oracle"].set_index(["family", "n", "regime"])["mean"]
    q = q[q.label != "oracle"].assign(rel=lambda x: x["mean"].values / o.reindex(pd.MultiIndex.from_frame(x[["family", "n", "regime"]])).values)
    cols = sorted({(f, n, r) for f, n, r in zip(q.family, q.n, q.regime)}, key=lambda t: (list(FAMILY).index(t[0]), t[1], t[2] != "beta0"))
    M = q.pivot_table(index="label", columns=["family", "n", "regime"], values="rel").reindex(columns=pd.MultiIndex.from_tuples(cols))
    M = M.loc[M.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(7.2, 0.14 * len(M) + 1.4))
    im = ax.imshow(M.values, aspect="auto", cmap="viridis", vmin=0.3, vmax=1.0, interpolation="nearest")
    ax.set_yticks(range(len(M))); ax.set_yticklabels([l.replace("_", " ") for l in M.index], fontsize=5.5)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{FAMILY[f]} {n:,} {'H' if r == 'beta0' else 'C'}" for f, n, r in cols], rotation=60, ha="right", fontsize=5, rotation_mode="anchor")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="success / oracle")
    ax.set_title("D  every arm × every cell (H = honest, C = β = 0.5 low-skill cartel), success / oracle; blank = not run")
    save(fig, "D_heatmap_all_arms"); M.to_csv(f"{OUT}/D_heatmap_all_arms.csv")


if __name__ == "__main__":
    d = load(); C = cells(d)
    fig_A(C); fig_B(C); fig_D(d)          # C: scripts/paired_gaps.py (needs per-seed rows)
