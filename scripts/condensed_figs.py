"""SAMPLE condensed replacements for the 48 figures/bars files -- one panel and one row each.
    python scripts/condensed_figs.py      -> figures/condensed_sample/{A,B}_{allb,stacked}.{png,pdf,csv}
b = 3 cells from figures/bars/<family>.csv; b = 1 / 5 from the va_b_* (MIDIAN) and rivals_b_* (budget-matched rivals)
rows and, for bernoulli / replay b = 1, their scale matrices. b is NEVER pooled: every bar is one budget.
  A  live headline: n = 10^2..10^5 (specialist), solid = honest, hatched = beta = 0.5 low-skill cartel.
  B  every family at its largest population, success / oracle, same arms and style.
  Two versions of each: _allb  -- one bar per (arm, regime, b), shade light -> dark = b = 1, 3, 5;
                        _stacked -- one full-width bar per (arm, regime): b = 1 at the bottom, then the gain to b = 3,
                                    then to b = 5 (drawn tallest-first, so a non-monotone arm shows out-of-order shades).
  declared argmax and random spend no probes: one bar (b does not apply). No markers on bars: a budget not in yet is an
  empty slot, and the title ends in ONE * while any bar of the figure is missing or its pool is incomplete.
  C / D (routing work vs n; energy per query) are scripts/efficiency_figs.py.
"Best learned router" / "best bandit" are CROSS-FITTED (scripts/seed_tables.py): for each seed the arm is chosen on the
OTHER seeds and scored on this one, so no bar is the maximum of noisy means over the seeds it reports (no winner's curse).
The pool is the same at every b in a cell (POOL minus the arms that cannot run there, NOT_RUNNABLE); a bar whose pool is
still missing a candidate (or a seed of one) at that b puts the one * in the figure title. The csv lists how often each arm was chosen. Frameworks are not drawn (figures/shortlist). The do-not-add list applies, and with it the
TEMPORARY extra_figs.HIDE_HALVING switch."""
from __future__ import annotations
import os, re, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extra_figs import excluded, se
from seed_tables import tables, crossfit, label, switched

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BARS, OUT = f"{ROOT}/figures/bars", f"{ROOT}/figures/condensed_sample"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "legend.fontsize": 6,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6, "figure.constrained_layout.use": True})
LEARNED = ["knn_router", "knn_router_online", "mlp_router", "flat_nsw_router", "cluster_head_router", "disrouter_cascade"]
BANDIT = ["ucb_per_family", "thompson_per_family", "warm_start_bandit", "linucb_honest", "trueskill_per_family"]
ARMS = [("midian", "MIDIAN", "#2ecc71"), ("midian_wo_defenses", "MIDIAN w/o defenses", "#c0392b"), ("flat_probe_argmax_online", "flat probe argmax (online)", "#3498db"),
        ("best_learned", "best learned router", "#ff7f0e"), ("best_bandit", "best bandit", "#9467bd"),
        ("declared_argmax", "declared argmax", "#5d6d7e"), ("random", "random", "#bbbbbb")]
POOLS = {"best_learned": LEARNED,                      # the bandit pool takes the TUNED warm-start bandit (n0 = 0.5) and drops the
         "best_bandit": [a for a in BANDIT if a != "linucb_honest"] + ["linucb_honest[bonus=own]", "warm_start_bandit[n0=0.5]"]}
# linucb_honest -> the fixed bonus (post hoc, audit 2026-09-23: the context bonus picks the WEAKEST tied agent).
# The pre-registered warm-start bandit (n0 = 5) trusts the declared claims; on the non-live backends those are true skill
# + 5 % noise (an answer key), so it counts there only on the calibrated-claims reruns (erratum 30), not on these rows.
CLAIM_KEY = {"bernoulli", "replay", "routereval", "llmrouterbench"}
SW = set()                                             # families on the erratum-30 rows (seed_tables.switched): n0 = 5 counts there
B_INVARIANT = {"declared_argmax", "random", "cluster_head_router", "disrouter_cascade"}   # never probe (needs has no probe / reports)
NOT_RUNNABLE = lambda fam, n: ({"trueskill_per_family"} if n >= 10 ** 5 else set()) | ({"mlp_router"} if n >= 5000 else set()) \
                              | ({"knn_router", "knn_router_online", "mlp_router"} if fam in ("bernoulli", "replay") else set())   # grid.yaml pool_fill_*
BUDGETLESS = {"declared_argmax", "random"}
BS = (1, 3, 5)
PRIMARY = {"live": "specialist", "routereval": "strong_to_weak"}          # one population shape per family in A / B
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
        out[key] = {"key": key, "oracle": (s.at["oracle", "mean"], s.at["oracle", "ci_lo"], s.at["oracle", "ci_hi"], "oracle"),
                    "raw": {3: {l: (s.at[l, "mean"], s.at[l, "ci_lo"], s.at[l, "ci_hi"]) for l in s.index if l != "oracle" and not excluded(l)}}}
    return out


def arms_at(cell):
    """{arm key: {b: (mean, lo, hi, chosen)}}; best learned router / best bandit are cross-fitted from the per-seed tables."""
    out = {}
    fam, n = cell["key"][0], cell["key"][2]
    for b, T in cell.get("seeds", {}).items():
        for k, pool in POOLS.items():
            want = [a for a in pool if a not in NOT_RUNNABLE(fam, n) and not (a == "warm_start_bandit" and fam in CLAIM_KEY - SW)]
            vals, picks = crossfit(T, want)
            if len(vals) < 2: continue                       # one seed cannot be cross-fitted: no bar, never the biased pick
            lo, hi = se(vals.values); miss = [a for a in want if a not in T.columns or T.loc[vals.index, a].isna().any()]   # absent, or missing a seed
            chosen = "; ".join(f"{a} x{c}" for a, c in picks.value_counts().items()) + (f" | INCOMPLETE POOL, missing or partial {', '.join(miss)}" if miss else "")
            out.setdefault(k, {})[b] = (float(vals.mean()), float(lo), float(hi), chosen)
    for b, raw in cell["raw"].items():
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
    w = 0.86 / len(slots); fig, ax = plt.subplots(figsize=(7.2, 2.8)); rec = []; incomplete = False; col = {k: c for k, _, c in ARMS}; lab = {k: l for k, l, _ in ARMS}
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
                    incomplete = True; continue
                m, lo, hi, chosen = got[b]; m, lo, hi = m / z, lo / z, hi / z
                incomplete |= "INCOMPLETE" in chosen
                first = i == 0 and not h and (b == 3)
                ax.bar(x, m, ww, color=shade(col[k], b) if k not in BUDGETLESS else col[k], edgecolor="black", lw=0.3,
                       hatch="////" if h else None, alpha=0.8 if h else 1, label=lab[k] if first else None, zorder=2 + depth)
                if not nested or (b == 3 and not stacked):          # stacked: an interval inside the stack misreads; see _allb
                    ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="#222", elinewidth=0.4, capsize=0.6, zorder=6)
                rec.append(dict(group=xl.replace("\n", " "), regime=REG[r], arm=lab[k], b=b if k not in BUDGETLESS else "-", chosen=chosen, value=m, ci_lo=lo, ci_hi=hi,
                                b_invariant=k in B_INVARIANT or (k in POOLS and set(re.findall(r"([\w\[\]=.]+) x\d+", chosen.split("|")[0])) <= B_INVARIANT)))
        if o: ax.hlines(1.0 if norm else o[0], i - 0.46, i + 0.46, colors="#7f8c8d", linestyles=":", lw=1.2, zorder=7, label="oracle" if i == 0 else None)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([xl for xl, _ in groups])
    ax.set_ylim(0.2, 1.05 if norm else 0.95); ax.set_ylabel("success / oracle" if norm else "success")
    ax.grid(axis="y", lw=0.3, alpha=0.35); ax.set_axisbelow(True); ax.set_title(title + (" *" if incomplete else ""))
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.07)); save(fig, name)
    pd.DataFrame(rec).to_csv(f"{OUT}/{name}.csv", index=False)


KEY_ALL = "light / mid / dark = probe budget b = 1 / 3 / 5; whiskers ±1 s.e. over seeds"
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


def add_budgets(C):
    """b = 1 / 5 for every arm: live / RouterEval / LLMRouterBench / bernoulli 10^7 / replay 10^6 from the va_b_* (MIDIAN)
    and rivals_b_* (budget-matched rivals) rows (per-seed means, whiskers set by narrow()); bernoulli / replay b = 1 from their
    scale matrices. A cell the runs have not reached simply has no bar (a * marks it)."""
    sys.path.insert(0, ROOT)
    from fw_variant_numbers import load as rows, regime
    R = os.environ.get("RTE_DATA", "/scratch/rte") + "/results"
    fams = [("live", "specialist", "n100"), ("live", "specialist", "n1000"), ("live", "specialist", "n10k"), ("live", "specialist", "n100k"),
            ("routereval", "strong_to_weak", "routereval5k"), ("llmrouterbench", "20 models", "llmrouterbench"),
            ("bernoulli", "specialist", "bernoulli_1e7"), ("replay", "all shapes pooled", "replay_1e6")]
    for fam, grp, tag in fams:
        if fam in SW: continue                            # every b comes from the erratum-30 per-seed tables (from_tables)
        for g in (f"va_b_{tag}", f"rivals_b_{tag}"):
            df = rows(g)
            if df.empty: continue
            df = df.assign(label=[label(m, p) for m, p in zip(df.method, df.params)])
            for (n, b, beta, ls, l), q in df.groupby(["n", "b", "beta", "liar_select", "label"]):
                key = (fam, grp, int(n), regime(beta, ls))
                if key not in C or int(b) not in (1, 5) or l == "oracle" or excluded(l): continue
                if grp == "all shapes pooled":            # pooled shapes: only seeds that have EVERY shape, else a partial
                    by = q.groupby(["seed", "dist"]).success.mean().unstack()   # run compares different populations
                    by = by.dropna() if by.shape[1] == 3 else by.iloc[0:0]
                    if by.empty: continue
                    per = by.mean(axis=1)
                else: per = q.groupby("seed").success.mean()
                lo, hi = se(per.values)
                C[key]["raw"].setdefault(int(b), {})[l] = (float(per.mean()), float(lo), float(hi))
    names = {"beta=0 (no liars)": "beta0", "beta=0.5 CARTEL (low-skill-first)": "cartel"}
    for fam, grp, g in (("bernoulli", "specialist", "bernoulli_scale_v5"), ("replay", "all shapes pooled", "replay_scale_v5")):
        if fam in SW: continue
        m = pd.read_csv(f"{R}/{g}/matrix_success.csv"); m = m[(m.b == 1) & (m.metric == "success") & m.regime.isin(list(names))]
        for r in m.itertuples():
            key = (fam, grp, int(r.n), names[r.regime])
            if key in C and r.label != "oracle" and not excluded(r.label):
                C[key]["raw"].setdefault(1, {})[r.label] = (float(r.mean), float(r.ci_lo), float(r.ci_hi))


def from_tables(C, T):
    """A switched family's cell: every arm, b and the oracle from its erratum-30 per-seed tables (whiskers +/- 1 s.e.)."""
    for key, bb in T.items():
        if key[0] not in SW or key not in C: continue
        if 3 not in bb or "oracle" not in bb[3]:              # never fall back to the old rows: an empty (incomplete) cell
            C[key]["raw"] = {}; continue
        cell = lambda v: (float(v.mean()), *map(float, se(v.values)))
        C[key]["oracle"] = (*cell(bb[3]["oracle"].dropna()), "oracle")
        C[key]["raw"] = {b: {l: cell(t[l].dropna()) for l in t.columns if l != "oracle" and not excluded(l) and t[l].notna().any()} for b, t in bb.items()}


def narrow(C, T):
    """Every whisker = +/- 1 s.e. over seeds, from the per-seed tables (the bar CSVs and scale matrices carry bootstrap CIs).
    A bar whose per-seed values are not in the tables keeps its mean and gets no whisker; the means must agree."""
    for key, c in C.items():
        bb = T.get(key, {})
        def fix(v, b, l):
            if l != "oracle" and l not in {k for k, _, _ in ARMS}: return v   # not drawn (pool members come from the tables)
            t = bb.get(b)
            if t is None or l not in t or t[l].notna().sum() == 0: return (v[0], v[0], v[0], *v[3:])
            assert abs(t[l].mean() - v[0]) < 0.01 or l == "oracle" and b != 3, f"{key} b={b} {l}: {t[l].mean():.3f} vs {v[0]:.3f}"
            return (v[0], *map(float, se(t[l].dropna().values)), *v[3:])
        c["raw"] = {b: {l: fix(v, b, l) for l, v in r.items()} for b, r in c["raw"].items()}
        c["oracle"] = fix(c["oracle"], 3, "oracle")


if __name__ == "__main__":
    d = load(); C = cells(d); SW |= switched(); T = tables(); add_budgets(C); from_tables(C, T); narrow(C, T)
    for key, bb in T.items():                        # per-seed tables for the cross-fitted pooled arms
        if key in C: C[key]["seeds"] = bb
    fig_A(C); fig_B(C)                    # C / D: scripts/efficiency_figs.py
