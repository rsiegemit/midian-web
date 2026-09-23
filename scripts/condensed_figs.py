"""SAMPLE condensed replacements for the 48 figures/bars files -- one panel and one row each.
    python scripts/condensed_figs.py      -> figures/condensed_sample/{A,B}_{allb,nested}, D_*.{png,pdf,csv}
b = 3 cells from figures/bars/<family>.csv; b = 1 / 5 from the va_b_* (MIDIAN-VA) and rivals_b_* (budget-matched rivals)
rows and, for bernoulli / replay b = 1, their scale matrices. b is NEVER pooled: every bar is one budget.
  A  live headline: n = 10^2..10^5 (specialist), solid = honest, hatched = beta = 0.5 low-skill cartel.
  B  every family at its largest population, success / oracle, same arms and style.
  Two versions of each: _allb  -- one bar per (arm, regime, b), shade light -> dark = b = 1, 3, 5;
                        _stacked -- one full-width bar per (arm, regime): b = 1 at the bottom, then the gain to b = 3,
                                    then to b = 5 (drawn tallest-first, so a non-monotone arm shows out-of-order shades).
  declared argmax and random spend no probes: one bar (b does not apply). * = a budget not in yet.
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
BUDGETLESS = {"declared_argmax", "random"}
BS = (1, 3, 5)
PRIMARY = {"live": "specialist", "routereval": "strong_to_weak"}          # one population shape per family in A / B / D
FAMILY = {"live": "live", "bernoulli": "bernoulli", "replay": "RouterBench replay", "routereval": "RouterEval", "llmrouterbench": "LLMRouterBench"}
REG = {"beta0": "honest", "cartel": "β=0.5 cartel"}


def load():
    d = pd.concat([pd.read_csv(f"{BARS}/{f}.csv") for f in FAMILY], ignore_index=True)
    return d[~d.label.astype(str).str.startswith("fw_")]


def cells(d):
    """(family, group, n, regime) -> {"raw": {b: {label: (mean, lo, hi)}}, "oracle": (...)}; b = 3 from the bar CSVs."""
    out = {}
    for key, q in d.groupby(["family", "group", "n", "regime"]):
        s = q.set_index("label")
        if "oracle" not in s.index: continue
        out[key] = {"oracle": (s.at["oracle", "mean"], s.at["oracle", "ci_lo"], s.at["oracle", "ci_hi"], "oracle"),
                    "raw": {3: {l: (s.at[l, "mean"], s.at[l, "ci_lo"], s.at[l, "ci_hi"]) for l in s.index if l != "oracle" and not excluded(l)}}}
    return out


def arms_at(cell):
    """{arm key: {b: (mean, lo, hi, chosen label)}}; best learned router / best bandit are chosen per b, per cell."""
    out = {}
    for b, raw in cell["raw"].items():
        for k, pool in (("best_learned", LEARNED), ("best_bandit", BANDIT + ["warm_start_bandit[n0=0.5]"])):
            have = [a for a in pool if a in raw]
            if have: c = max(have, key=lambda a: raw[a][0]); out.setdefault(k, {})[b] = (*raw[c], c)
        for k, _, _ in ARMS:
            if k in raw and (b == 3 or k not in BUDGETLESS): out.setdefault(k, {})[b] = (*raw[k], k)
    return out


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=250); fig.savefig(f"{OUT}/{name}.pdf"); plt.close(fig); print(f"[{name}] written")


def shade(col, b):
    import matplotlib.colors as mc
    c = np.array(mc.to_rgb(col)); return tuple(c + (1 - c) * 0.5) if b == 1 else tuple(c * 0.6) if b == 5 else tuple(c)


def budget_bars(C, groups, title, name, norm=False, nested=False, stacked=False):
    """One group per (x label, cell). _allb: a bar per (arm, regime, b). _nested: one slot per (arm, regime) holding the
    b = 5, 3, 1 bars widest to narrowest (success rises with b, so each stays visible). A budget not in yet: * in its place."""
    A = {g: arms_at(C[key[:3] + (r,)]) for _, key in groups for g, r in [((key, r), r) for r in REG] if key[:3] + (r,) in C}
    arms = [a for a in ARMS if any(a[0] in v for v in A.values())]
    slots = [(k, r) for k, _, _ in arms for r in REG]                      # nested: one slot each
    nested = nested or stacked                                          # both: one slot per (arm, regime)
    if not nested: slots = [(k, r, b) for k, _, _ in arms for r in REG for b in ((3,) if k in BUDGETLESS else BS)]
    w = 0.86 / len(slots); fig, ax = plt.subplots(figsize=(7.2, 2.8)); rec = []; col = {k: c for k, _, c in ARMS}; lab = {k: l for k, l, _ in ARMS}
    for i, (xl, key) in enumerate(groups):
        o = C.get(key[:3] + ("beta0",), {}).get("oracle"); z = o[0] if (norm and o) else 1.0
        for j, sl in enumerate(slots):
            k, r = sl[0], sl[1]; x = i + (j - (len(slots) - 1) / 2) * w; h = r == "cartel"
            got = A.get((key, r), {}).get(k, {})
            bs = [sl[2]] if not nested else ((3,) if k in BUDGETLESS else (5, 3, 1))
            if stacked: bs = sorted(bs, key=lambda b: -got[b][0] if b in got else 0)   # tallest behind: each level at its true height
            for depth, b in enumerate(bs):
                ww = w * 0.95 * (1.0 if (not nested or stacked) else (1.0, 0.66, 0.36)[depth])
                if b not in got:
                    if k in A.get((key, r), {}): ax.text(x, 0.205 if not norm else 0.205, "*", ha="center", va="bottom", fontsize=5.5)
                    continue
                m, lo, hi, chosen = got[b]; m, lo, hi = m / z, lo / z, hi / z
                first = i == 0 and not h and (b == 3)
                ax.bar(x, m, ww, color=shade(col[k], b) if k not in BUDGETLESS else col[k], edgecolor="black", lw=0.3,
                       hatch="////" if h else None, alpha=0.8 if h else 1, label=lab[k] if first else None, zorder=2 + depth)
                if not nested or (b == 3 and not stacked):          # stacked: an interval inside the stack misreads; see _allb
                    ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="#222", elinewidth=0.4, capsize=0.6, zorder=6)
                rec.append(dict(group=xl.replace("\n", " "), regime=REG[r], arm=lab[k], b=b if k not in BUDGETLESS else "-", chosen=chosen, value=m, ci_lo=lo, ci_hi=hi))
        if o: ax.hlines(1.0 if norm else o[0], i - 0.46, i + 0.46, colors="#7f8c8d", linestyles=":", lw=1.2, zorder=7, label="oracle" if i == 0 else None)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([xl for xl, _ in groups])
    ax.set_ylim(0.2, 1.05 if norm else 0.95); ax.set_ylabel("success / oracle" if norm else "success")
    ax.grid(axis="y", lw=0.3, alpha=0.35); ax.set_axisbelow(True); ax.set_title(title)
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.07)); save(fig, name)
    pd.DataFrame(rec).to_csv(f"{OUT}/{name}.csv", index=False)


KEY_ALL = "light / mid / dark = probe budget b = 1 / 3 / 5"
KEY_NEST = "in each slot: wide = b 5, mid = b 3, narrow = b 1"
KEY_STACK = "stack: b = 1, +gain to b = 3, +gain to b = 5 (CIs in _allb)"


def fig_A(C):
    ns = sorted(n for (f, g, n, r) in C if f == "live" and g == "specialist" and r == "beta0")
    g = [(f"n = {n:,}", ("live", "specialist", n)) for n in ns]
    budget_bars(C, g, f"A  live RTE, specialist; solid honest, hatched β = 0.5 cartel; {KEY_ALL}", "A_live_allb")
    budget_bars(C, g, f"A  live, specialist; hatched = β 0.5 cartel; {KEY_STACK}", "A_live_stacked", stacked=True)


def primary(f, g): return g == PRIMARY.get(f, g)


def fig_B(C):
    """Each family at its LARGEST population with both regimes, divided by that cell's oracle."""
    groups = []
    for f in FAMILY:
        ns = [n for (ff, g, n, r) in C if ff == f and primary(f, g) and r == "cartel" and (ff, g, n, "beta0") in C]
        if not ns: continue
        n = max(ns); g = next(g for (ff, g, nn, r) in C if ff == f and nn == n and primary(f, g))
        groups.append((f"{FAMILY[f]}\nn = {n:,}", (f, g, n)))
    budget_bars(C, groups, f"B  each family at its largest n, success / oracle; hatched = cartel; {KEY_ALL}", "B_families_allb", norm=True)
    budget_bars(C, groups, f"B  each family at its largest n, / oracle; hatched = cartel; {KEY_STACK}", "B_families_stacked", norm=True, stacked=True)


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


def label(method, params):
    import json
    from rte.analyze import ALIAS
    p = json.loads(params) if isinstance(params, str) and params.startswith("{") else {}
    short = ",".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(p.items()))
    l = method if not p else f"{method}[{short}]"; return ALIAS.get(l, l)


def add_budgets(C):
    """b = 1 / 5 for every arm: live / RouterEval / LLMRouterBench / bernoulli 10^7 / replay 10^6 from the va_b_* (MIDIAN-VA)
    and rivals_b_* (budget-matched rivals) rows (per-seed means, seed-bootstrap CI); bernoulli / replay b = 1 from their
    scale matrices. A cell the runs have not reached simply has no bar (a * marks it)."""
    sys.path.insert(0, ROOT)
    from fw_variant_numbers import load as rows, regime
    from extra_figs import ci
    R = os.environ.get("RTE_DATA", "/scratch/rte") + "/results"
    fams = [("live", "specialist", "n100"), ("live", "specialist", "n1000"), ("live", "specialist", "n10k"), ("live", "specialist", "n100k"),
            ("routereval", "strong_to_weak", "routereval5k"), ("llmrouterbench", "20 models", "llmrouterbench"),
            ("bernoulli", "specialist", "bernoulli_1e7"), ("replay", "all shapes pooled", "replay_1e6")]
    for fam, grp, tag in fams:
        for g in (f"va_b_{tag}", f"rivals_b_{tag}"):
            df = rows(g)
            if df.empty: continue
            df = df.assign(label=[label(m, p) for m, p in zip(df.method, df.params)])
            for (n, b, beta, ls, l), q in df.groupby(["n", "b", "beta", "liar_select", "label"]):
                key = (fam, grp, int(n), regime(beta, ls))
                if key not in C or int(b) not in (1, 5) or l == "oracle" or excluded(l): continue
                per = q.groupby("seed").success.mean(); lo, hi = ci(per.values)
                C[key]["raw"].setdefault(int(b), {})[l] = (float(per.mean()), float(lo), float(hi))
    names = {"beta=0 (no liars)": "beta0", "beta=0.5 CARTEL (low-skill-first)": "cartel"}
    for fam, grp, g in (("bernoulli", "specialist", "bernoulli_scale_v5"), ("replay", "all shapes pooled", "replay_scale_v5")):
        m = pd.read_csv(f"{R}/{g}/matrix_success.csv"); m = m[(m.b == 1) & (m.metric == "success") & m.regime.isin(list(names))]
        for r in m.itertuples():
            key = (fam, grp, int(r.n), names[r.regime])
            if key in C and r.label != "oracle" and not excluded(r.label):
                C[key]["raw"].setdefault(1, {})[r.label] = (float(r.mean), float(r.ci_lo), float(r.ci_hi))


if __name__ == "__main__":
    d = load(); C = cells(d); add_budgets(C)
    fig_A(C); fig_B(C); fig_D(d)          # C: scripts/paired_gaps.py (needs per-seed rows)
