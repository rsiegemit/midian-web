"""Figures 1 / 2 of the submission (A_live_stacked, B_families_stacked) and their appendix versions (_allb); style,
names, legend order and saving from scripts/figures/lib/figspec.py (docs/FIGURE_SPEC.md).
    python scripts/figures/condensed_figs.py [--out DIR] [--from-csv]
      -> <out>/{A,B}_{allb,stacked}.{png,pdf,csv}; <out> = $RTE_FIG_OUT or figures/paper
      + results/aggregates/figures/<name>.{csv,draw.csv}: the figure's CSV and its display list (every bar, whisker,
        oracle segment, tick and legend key in drawing order); --from-csv redraws from those alone (no $RTE_DATA).
b = 3 cells from results/aggregates/bars/<family>.csv; b = 1 / 5 from the va_b_* (MIDIAN) and rivals_b_* (budget-matched
rivals) rows and, for bernoulli / replay b = 1, their scale matrices. b is NEVER pooled: every bar is one budget.
  A  live headline: n = 10^2..10^5 (specialist), solid = honest, hatched = beta = 0.5 low-skill cartel.
  B  every family at its largest population, success / oracle, same arms and style.
  Two versions of each:
    _allb     one bar per (arm, regime, b), shade light -> dark = b = 1, 3, 5, whiskers +/- 1 s.e.;
    _stacked  one bar per (arm, regime) at b = 3; MIDIAN alone carries the budget overlay (b = 1 light in front, 3 mid,
              5 dark behind, drawn tallest-first), no whiskers (see _allb).
  declared argmax and random spend no probes: one bar (b does not apply). No markers on bars and no titles: a budget not
  in yet is an empty slot, and the script prints INCOMPLETE while any bar is missing or its pool is incomplete. A also
  carries "best framework, best text shortlist" (b = 3; add_best_framework), cross-fitted per seed like the pooled arms.
  C / D (routing work vs n; energy per query) are efficiency_figs.py.
"Best learned router" / "best bandit" are CROSS-FITTED (lib/stats.crossfit over lib/seed_tables): for each seed the arm
is chosen on the OTHER seeds and scored on this one, so no bar is the maximum of noisy means over the seeds it reports
(no winner's curse). The pool is the same at every b in a cell (POOL minus the arms that cannot run there,
NOT_RUNNABLE); a bar whose pool is still missing a candidate (or a seed of one) at that b is reported as INCOMPLETE
(cluster/ops/figure_status.py; figures carry no marks). The csv lists how often each arm was chosen. Frameworks are not
drawn (figures/shortlist). The do-not-add list applies, and with it the permanent halving exclusion (lib/exclusions.py).
Rendered under figspec.LEGACY_RC (the rcParams these figures always inherited from the retired paper_figs.py), inside an
rc_context."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from scripts.figures.lib import AGG, fig_out  # noqa: E402
from scripts.figures.lib import figspec as S  # noqa: E402
from scripts.figures.lib.exclusions import excluded  # noqa: E402
from scripts.figures.lib.grids import BUDGET_GRIDS, MATRICES  # noqa: E402
from scripts.figures.lib.regimes import AB as REG, MATRIX_REGIME  # noqa: E402
from scripts.figures.lib.stats import crossfit, se  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

BARS, DRAW = f"{AGG}/bars", f"{AGG}/figures"
LEARNED = [
    "knn_router",
    "knn_router_online",
    "mlp_router",
    "flat_nsw_router",
    "cluster_head_router",
    "disrouter_cascade",
]
BANDIT = ["ucb_per_family", "thompson_per_family", "warm_start_bandit", "linucb_honest", "trueskill_per_family"]
ARMS = [k for k in S.ORDER if k not in ("oracle", "best_framework")]  # the arms read from the rows, in legend order
TEXT = {
    "tfidf",
    "bm25",
    "embed",
    "dense",
    "dense_icomp",
    "dense_idemo",
    "sota",
    "sota_icomp",
    "sota_idemo",
}  # text shortlists (not declared / cohort)
POOLS = {
    "best_learned": LEARNED,  # the bandit pool takes the TUNED warm-start bandit (n0 = 0.5) and drops the
    "best_bandit": [a for a in BANDIT if a != "linucb_honest"]
    + ["linucb_honest[bonus=own]", "warm_start_bandit[n0=0.5]"],
}
# linucb_honest -> the fixed bonus (post hoc, audit 2026-09-23: the context bonus picks the WEAKEST tied agent).
# The pre-registered warm-start bandit (n0 = 5) trusts the declared claims; on the non-live backends those are true
# skill
# + 5 % noise (an answer key), so it counts there only on the calibrated-claims reruns (erratum 30), not on these rows.
CLAIM_KEY = {"bernoulli", "replay", "routereval", "llmrouterbench"}
SW = set()  # families on the erratum-30 rows (seed_tables.switched): n0 = 5 counts there
B_INVARIANT = {
    "declared_argmax",
    "random",
    "cluster_head_router",
    "disrouter_cascade",
}  # never probe (needs has no probe / reports)
NOT_RUNNABLE = (
    lambda fam, n: ({"trueskill_per_family"} if n >= 10**5 else set())
    | ({"mlp_router"} if n >= 5000 else set())
    | ({"knn_router", "knn_router_online", "mlp_router"} if fam in ("bernoulli", "replay") else set())
)  # configs/grids pool_fill_*
BUDGETLESS = {"declared_argmax", "random"}
BS = (1, 3, 5)
PRIMARY = {"live": "specialist", "routereval": "strong_to_weak"}  # one population shape per family in A / B
FAMILY = list(S.BACKEND)
FIGS = {
    "A_live_allb": (False, False),
    "A_live_stacked": (False, True),  # name -> (norm, stacked)
    "B_families_allb": (True, False),
    "B_families_stacked": (True, True),
}


def load():
    d = pd.concat([pd.read_csv(f"{BARS}/{f}.csv") for f in FAMILY], ignore_index=True)
    return d[~d.label.astype(str).str.startswith("fw_")]


def cells(d):
    """(family, group, n, regime) -> {"raw": {b: {label: (mean, lo, hi)}}, "oracle": (...)}; b = 3 from the bar CSVs."""
    out = {}
    for key, q in d.groupby(["family", "group", "n", "regime"]):
        s = q.set_index("label")
        if "oracle" not in s.index:
            continue
        out[key] = {
            "key": key,
            "oracle": (s.at["oracle", "mean"], s.at["oracle", "ci_lo"], s.at["oracle", "ci_hi"], "oracle"),
            "raw": {
                3: {
                    l: (s.at[l, "mean"], s.at[l, "ci_lo"], s.at[l, "ci_hi"])
                    for l in s.index
                    if l != "oracle" and not excluded(l)
                }
            },
        }
    return out


def arms_at(cell):
    """{arm key: {b: (mean, lo, hi, chosen)}}; best learned router / best bandit are cross-fitted from the per-seed
    tables."""
    out = {}
    fam, n = cell["key"][0], cell["key"][2]
    for b, T in cell.get("seeds", {}).items():
        for k, pool in POOLS.items():
            want = [
                a
                for a in pool
                if a not in NOT_RUNNABLE(fam, n) and not (a == "warm_start_bandit" and fam in CLAIM_KEY - SW)
            ]
            vals, picks = crossfit(T, want)
            if len(vals) < 2:
                continue  # one seed cannot be cross-fitted: no bar, never the biased pick
            lo, hi = se(vals.values)
            miss = [
                a for a in want if a not in T.columns or T.loc[vals.index, a].isna().any()
            ]  # absent, or missing a seed
            chosen = "; ".join(f"{a} x{c}" for a, c in picks.value_counts().items()) + (
                f" | INCOMPLETE POOL, missing or partial {', '.join(miss)}" if miss else ""
            )
            out.setdefault(k, {})[b] = (float(vals.mean()), float(lo), float(hi), chosen)
    if cell.get("fw_seeds") is not None:  # best framework, best text shortlist: cross-fitted over the pairs
        vals, picks = crossfit(cell["fw_seeds"], list(cell["fw_seeds"].columns))
        if len(vals) >= 2:
            lo, hi = se(vals.values)
            out["best_framework"] = {
                3: (
                    float(vals.mean()),
                    float(lo),
                    float(hi),
                    "; ".join(f"{a} x{c}" for a, c in picks.value_counts().items()),
                )
            }
    for b, raw in cell["raw"].items():
        for k in ARMS:
            if k in raw and (b == 3 or k not in BUDGETLESS):
                out.setdefault(k, {})[b] = (*raw[k], k)
    return out


def budget_bars(C, groups, name, norm=False, stacked=False, skip=()):
    """One group per (x label, cell). _allb: a bar per (arm, regime, b) with whiskers. _stacked: one slot per (arm,
    regime) at b = 3; MIDIAN's slot holds its b = 5, 3, 1 bars tallest-first (each level at its true height), no
    whiskers. Returns (the figure CSV, the display list: one row per drawing call, in order)."""
    A = {(key, r): arms_at(C[key[:3] + (r,)]) for _, key in groups for r in REG if key[:3] + (r,) in C}
    arms = [k for k in S.ORDER if k not in skip and any(k in v for v in A.values())]
    one = lambda k: k in BUDGETLESS or k == "best_framework"  # a single bar: no probes, or frameworks (b = 3 only)
    slots = (
        [(k, r, None) for k in arms for r in REG]
        if stacked
        else [(k, r, b) for k in arms for r in REG for b in ((3,) if one(k) else BS)]
    )
    w = 0.86 / len(slots)
    rec, dl = [], []
    incomplete = False
    for i, (xl, key) in enumerate(groups):
        o = C.get(key[:3] + ("beta0",), {}).get("oracle")
        z = o[0] if (norm and o) else 1.0
        for j, (k, r, sb) in enumerate(slots):
            x = i + (j - (len(slots) - 1) / 2) * w
            got = A.get((key, r), {}).get(k, {})
            bs = (
                [sb]
                if not stacked
                else sorted((5, 3, 1) if k == "midian" else (3,), key=lambda b: -got[b][0] if b in got else 0)
            )
            for depth, b in enumerate(bs):
                if b not in got:
                    incomplete = True
                    continue
                m, lo, hi, chosen = got[b]
                m, lo, hi = m / z, lo / z, hi / z
                incomplete |= "INCOMPLETE" in chosen
                dl.append(
                    dict(
                        el="bar",
                        x=x,
                        y=m,
                        w=w * 0.95,
                        key=k,
                        cartel=r == "cartel",
                        b=None if one(k) else b,
                        z=2 + depth,
                    )
                )
                if not stacked:
                    dl.append(dict(el="whisker", x=x, y=m, lo=lo, hi=hi))
                rec.append(
                    dict(
                        group=(S.BACKEND[key[0]] + " " if norm else "") + f"n = {key[2]:,}",
                        regime=REG[r],
                        arm=S.NAME[k],
                        key=k,
                        b=b if not one(k) else "-",
                        chosen=chosen,
                        value=m,
                        ci_lo=lo,
                        ci_hi=hi,
                        b_invariant=k in B_INVARIANT
                        or (k in POOLS and set(re.findall(r"([\w\[\]=.]+) x\d+", chosen.split("|")[0])) <= B_INVARIANT),
                    )
                )
        if o:
            dl.append(dict(el="oracle", x=i - 0.46, x1=i + 0.46, y=1.0 if norm else o[0]))
    dl += [dict(el="tick", x=i, text=xl) for i, (xl, _) in enumerate(groups)] + [
        dict(el="legend", key=k) for k in ["oracle"] + arms
    ]
    if incomplete:
        print(f"[{name}] INCOMPLETE: a bar is missing or its pool is incomplete (see the csv)")
    return pd.DataFrame(rec), pd.DataFrame(dl)


def render(dl, name, out):
    """Draw one A / B figure from its display list (figspec.LEGACY_RC inside an rc_context) -> <out>/<name>.{pdf,png}.
    """
    norm, stacked = FIGS[name]
    num = lambda v: None if pd.isna(v) else v
    with plt.rc_context(S.LEGACY_RC):
        fig, ax = S.figure("body" if stacked else "appendix")
        for d in dl.itertuples():
            if d.el == "bar":
                S.bar(
                    ax,
                    d.x,
                    d.y,
                    d.w,
                    d.key,
                    cartel=str(d.cartel) == "True",
                    b=None if num(d.b) is None else int(d.b),
                    z=d.z,
                )
            elif d.el == "whisker":
                S.whisker(ax, d.x, d.y, d.lo, d.hi)
            elif d.el == "oracle":
                S.oracle(ax, d.x, d.x1, d.y)
        ticks = dl[dl.el == "tick"]
        ax.set_xticks(range(len(ticks)))
        ax.set_xticklabels(list(ticks.text))
        ax.set_xlim(-0.5, len(ticks) - 0.5)
        S.finish(ax, ylabel=S.AXIS["rel" if norm else "success"], ylim=(0.2, 1.05 if norm else 0.95))
        keys = list(dl[dl.el == "legend"].key)  # _allb: three rows, the budget swatches in the last column
        S.legend(
            ax,
            keys,
            ncol=-(-len(keys) // 2) if stacked else -(-len(keys) // 3) + 1,  # _stacked: two rows
            extra=None if stacked else S.budget_handles(-len(keys) % 3),
            **(dict(columnspacing=0.5, handlelength=1.0) if stacked else {}),
        )
        S.save(fig, name, out)


def publish(rec, dl, name, out):
    """The figure CSV to <out> and to results/aggregates/figures (with the display list), then the figure itself."""
    os.makedirs(DRAW, exist_ok=True)
    rec.to_csv(f"{out}/{name}.csv", index=False)
    rec.to_csv(f"{DRAW}/{name}.csv", index=False)
    dl.to_csv(f"{DRAW}/{name}.draw.csv", index=False)
    render(dl, name, out)


def from_csv(out):
    """Redraw A / B from results/aggregates/figures alone."""
    for name in FIGS:
        render(pd.read_csv(f"{DRAW}/{name}.draw.csv"), name, out)
        shutil.copyfile(f"{DRAW}/{name}.csv", f"{out}/{name}.csv")


def fig_A(C, out):
    ns = sorted(n for (f, g, n, r) in C if f == "live" and g == "specialist" and r == "beta0")
    g = list(zip(S.pow10_ticks(ns, "n"), [("live", "specialist", n) for n in ns]))
    publish(*budget_bars(C, g, "A_live_allb"), "A_live_allb", out)
    publish(*budget_bars(C, g, "A_live_stacked", stacked=True), "A_live_stacked", out)


def primary(f, g):
    return g == PRIMARY.get(f, g)


def fig_B(C, out):
    """Each family at its LARGEST population with both regimes, divided by that cell's oracle."""
    groups = []
    for f in FAMILY:
        ns = [n for (ff, g, n, r) in C if ff == f and primary(f, g) and r == "cartel" and (ff, g, n, "beta0") in C]
        if not ns:
            continue
        n = max(ns)
        g = next(g for (ff, g, nn, r) in C if ff == f and nn == n and primary(f, g))
        groups.append((f"{S.BACKEND[f]}\n{S.pow10(n, 'n')}", (f, g, n)))
    publish(*budget_bars(C, groups, "B_families_allb", norm=True, skip=("best_framework",)), "B_families_allb", out)
    publish(
        *budget_bars(C, groups, "B_families_stacked", norm=True, stacked=True, skip=("best_framework",)),
        "B_families_stacked",
        out,
    )


def add_best_framework(C):
    """Per live n and regime, the seed x (framework | text shortlist) success table of every pair with the full seed
    count at that cell (framework rows via shortlist_figs.collect: b = 3, specialist); arms_at cross-fits the best pair
    per seed."""
    from scripts.figures.shortlist_figs import collect

    for (n, dist, reg), cell in collect("live").items():
        key = ("live", "specialist", n, reg)
        if dist != "specialist" or key not in C:
            continue
        T = pd.DataFrame({f"{m} | {src}": v for m, d in cell["fw"].items() for src, v in d.items() if src in TEXT})
        if T.empty:
            continue
        full = T.notna().sum()
        C[key]["fw_seeds"] = T[full.index[full == full.max()]]


def add_budgets(C):
    """b = 1 / 5 for every arm: live / RouterEval / LLMRouterBench / bernoulli 10^7 / replay 10^6 from the va_b_*
    (MIDIAN) and rivals_b_* (budget-matched rivals) rows (per-seed means, whiskers set by narrow()); bernoulli / replay
    b = 1 from their scale matrices. A cell the runs have not reached simply has no bar (figure_status.py lists it)."""
    from scripts.figures.lib.regimes import tag as regime
    from scripts.figures.lib.rows import RESULTS as R, label, load_fw as rows

    for fam, grp, tag in BUDGET_GRIDS:
        if fam in SW:
            continue  # every b comes from the erratum-30 per-seed tables (from_tables)
        for g in (f"va_b_{tag}", f"rivals_b_{tag}"):
            df = rows(g)
            if df.empty:
                continue
            df = df.assign(label=[label(m, p) for m, p in zip(df.method, df.params)])
            for (n, b, beta, ls, l), q in df.groupby(["n", "b", "beta", "liar_select", "label"]):
                key = (fam, grp, int(n), regime(beta, ls))
                if key not in C or int(b) not in (1, 5) or l == "oracle" or excluded(l):
                    continue
                if grp == "all shapes pooled":  # pooled shapes: only seeds that have EVERY shape, else a partial
                    by = q.groupby(["seed", "dist"]).success.mean().unstack()  # run compares different populations
                    by = by.dropna() if by.shape[1] == 3 else by.iloc[0:0]
                    if by.empty:
                        continue
                    per = by.mean(axis=1)
                else:
                    per = q.groupby("seed").success.mean()
                lo, hi = se(per.values)
                C[key]["raw"].setdefault(int(b), {})[l] = (float(per.mean()), float(lo), float(hi))
    names = {MATRIX_REGIME[k]: k for k in ("beta0", "cartel")}
    for fam, grp, g in (
        ("bernoulli", "specialist", MATRICES["bernoulli"]),
        ("replay", "all shapes pooled", MATRICES["replay"]),
    ):
        if fam in SW:
            continue
        m = pd.read_csv(f"{R}/{g}/matrix_success.csv")
        m = m[(m.b == 1) & (m.metric == "success") & m.regime.isin(list(names))]
        for r in m.itertuples():
            key = (fam, grp, int(r.n), names[r.regime])
            if key in C and r.label != "oracle" and not excluded(r.label):
                C[key]["raw"].setdefault(1, {})[r.label] = (float(r.mean), float(r.ci_lo), float(r.ci_hi))


def from_tables(C, T):
    """A switched family's cell: every arm, b and the oracle from its erratum-30 per-seed tables (whiskers +/- 1 s.e.).
    """
    for key, bb in T.items():
        if key[0] not in SW or key not in C:
            continue
        if 3 not in bb or "oracle" not in bb[3]:  # never fall back to the old rows: an empty (incomplete) cell
            C[key]["raw"] = {}
            continue
        cell = lambda v: (float(v.mean()), *map(float, se(v.values)))
        C[key]["oracle"] = (*cell(bb[3]["oracle"].dropna()), "oracle")
        C[key]["raw"] = {
            b: {l: cell(t[l].dropna()) for l in t.columns if l != "oracle" and not excluded(l) and t[l].notna().any()}
            for b, t in bb.items()
        }


def narrow(C, T):
    """Every whisker = +/- 1 s.e. over seeds, from the per-seed tables (the bar CSVs and scale matrices carry bootstrap
    CIs). A bar whose per-seed values are not in the tables keeps its mean and gets no whisker; the means must agree."""
    for key, c in C.items():
        bb = T.get(key, {})

        def fix(v, b, l):
            if l != "oracle" and l not in ARMS:
                return v  # not drawn (pool members come from the tables)
            t = bb.get(b)
            if t is None or l not in t or t[l].notna().sum() == 0:
                return (v[0], v[0], v[0], *v[3:])
            assert (
                abs(t[l].mean() - v[0]) < 0.01 or l == "oracle" and b != 3
            ), f"{key} b={b} {l}: {t[l].mean():.3f} vs {v[0]:.3f}"
            return (v[0], *map(float, se(t[l].dropna().values)), *v[3:])

        c["raw"] = {b: {l: fix(v, b, l) for l, v in r.items()} for b, r in c["raw"].items()}
        c["oracle"] = fix(c["oracle"], 3, "oracle")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", help="output directory (default: $RTE_FIG_OUT or figures/paper)")
    p.add_argument("--from-csv", action="store_true", help="redraw from results/aggregates/figures only")
    a = p.parse_args(argv)
    out = fig_out(a.out)
    if a.from_csv:
        return from_csv(out)
    from scripts.figures.lib.seed_tables import switched, tables

    d = load()
    C = cells(d)
    SW.update(switched())
    T = tables()
    add_budgets(C)
    from_tables(C, T)
    narrow(C, T)
    for key, bb in T.items():  # per-seed tables for the cross-fitted pooled arms
        if key in C:
            C[key]["seeds"] = bb
    add_best_framework(C)
    fig_A(C, out)
    fig_B(C, out)  # C / D: efficiency_figs.py


if __name__ == "__main__":
    main()
