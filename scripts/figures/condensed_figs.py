"""Figures 1 / 2 of the submission (A_live_stacked, B_families_stacked) and their appendix versions (_allb); style, names,
legend order and saving from scripts/figspec.py (docs/FIGURE_SPEC.md).
    python scripts/condensed_figs.py      -> figures/condensed_sample/{A,B}_{allb,stacked}.{png,pdf,csv}
b = 3 cells from figures/bars/<family>.csv; b = 1 / 5 from the va_b_* (MIDIAN) and rivals_b_* (budget-matched rivals)
rows and, for bernoulli / replay b = 1, their scale matrices. b is NEVER pooled: every bar is one budget.
  A  live headline: n = 10^2..10^5 (specialist), solid = honest, hatched = beta = 0.5 low-skill cartel.
  B  every family at its largest population, success / oracle, same arms and style.
  Two versions of each: _allb  -- one bar per (arm, regime, b), shade light -> dark = b = 1, 3, 5, whiskers +/- 1 s.e.;
                        _stacked -- one bar per (arm, regime) at b = 3; MIDIAN alone carries the budget overlay (b = 1 light
                                    in front, 3 mid, 5 dark behind, drawn tallest-first), no whiskers (see _allb).
  declared argmax and random spend no probes: one bar (b does not apply). No markers on bars and no titles: a budget not in
  yet is an empty slot, and the script prints INCOMPLETE while any bar is missing or its pool is incomplete.
  A also carries "best framework, best text shortlist" (b = 3; add_best_framework), cross-fitted per seed like the pooled arms.
  C / D (routing work vs n; energy per query) are scripts/efficiency_figs.py.
"Best learned router" / "best bandit" are CROSS-FITTED (scripts/seed_tables.py): for each seed the arm is chosen on the
OTHER seeds and scored on this one, so no bar is the maximum of noisy means over the seeds it reports (no winner's curse).
The pool is the same at every b in a cell (POOL minus the arms that cannot run there, NOT_RUNNABLE); a bar whose pool is
still missing a candidate (or a seed of one) at that b is reported as INCOMPLETE (scripts/ops/figure_status.py; figures carry no marks). The csv lists how often each arm was chosen. Frameworks are not drawn (figures/shortlist). The do-not-add list applies, and with it the
TEMPORARY extra_figs.HIDE_HALVING switch."""
from __future__ import annotations
import os, re, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figspec as S
from extra_figs import excluded, se
from seed_tables import tables, crossfit, label, switched

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BARS, OUT = f"{ROOT}/figures/bars", f"{ROOT}/figures/condensed_sample"; os.makedirs(OUT, exist_ok=True)
LEARNED = ["knn_router", "knn_router_online", "mlp_router", "flat_nsw_router", "cluster_head_router", "disrouter_cascade"]
BANDIT = ["ucb_per_family", "thompson_per_family", "warm_start_bandit", "linucb_honest", "trueskill_per_family"]
ARMS = [k for k in S.ORDER if k not in ("oracle", "best_framework")]      # the arms read from the rows, in legend order
TEXT = {"tfidf", "bm25", "embed", "dense", "dense_icomp", "dense_idemo", "sota", "sota_icomp", "sota_idemo"}   # text shortlists (not declared / cohort)
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
FAMILY = list(S.BACKEND)
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
    if cell.get("fw_seeds") is not None:                 # best framework, best text shortlist: cross-fitted over the pairs
        vals, picks = crossfit(cell["fw_seeds"], list(cell["fw_seeds"].columns))
        if len(vals) >= 2:
            lo, hi = se(vals.values)
            out["best_framework"] = {3: (float(vals.mean()), float(lo), float(hi), "; ".join(f"{a} x{c}" for a, c in picks.value_counts().items()))}
    for b, raw in cell["raw"].items():
        for k in ARMS:
            if k in raw and (b == 3 or k not in BUDGETLESS): out.setdefault(k, {})[b] = (*raw[k], k)
    return out


def budget_bars(C, groups, name, norm=False, stacked=False, skip=()):
    """One group per (x label, cell). _allb: a bar per (arm, regime, b) with whiskers. _stacked: one slot per (arm, regime)
    at b = 3; MIDIAN's slot holds its b = 5, 3, 1 bars tallest-first (each level at its true height), no whiskers."""
    A = {(key, r): arms_at(C[key[:3] + (r,)]) for _, key in groups for r in REG if key[:3] + (r,) in C}
    arms = [k for k in S.ORDER if k not in skip and any(k in v for v in A.values())]
    one = lambda k: k in BUDGETLESS or k == "best_framework"            # a single bar: no probes, or frameworks (b = 3 only)
    slots = [(k, r, None) for k in arms for r in REG] if stacked else [(k, r, b) for k in arms for r in REG for b in ((3,) if one(k) else BS)]
    fig, ax = S.figure("body" if stacked else "appendix"); w = 0.86 / len(slots); rec = []; incomplete = False
    for i, (xl, key) in enumerate(groups):
        o = C.get(key[:3] + ("beta0",), {}).get("oracle"); z = o[0] if (norm and o) else 1.0
        for j, (k, r, sb) in enumerate(slots):
            x = i + (j - (len(slots) - 1) / 2) * w; got = A.get((key, r), {}).get(k, {})
            bs = [sb] if not stacked else sorted((5, 3, 1) if k == "midian" else (3,), key=lambda b: -got[b][0] if b in got else 0)
            for depth, b in enumerate(bs):
                if b not in got:
                    incomplete = True; continue
                m, lo, hi, chosen = got[b]; m, lo, hi = m / z, lo / z, hi / z
                incomplete |= "INCOMPLETE" in chosen
                S.bar(ax, x, m, w * 0.95, k, cartel=r == "cartel", b=None if one(k) else b, z=2 + depth)
                if not stacked: S.whisker(ax, x, m, lo, hi)
                rec.append(dict(group=(S.BACKEND[key[0]] + " " if norm else "") + f"n = {key[2]:,}", regime=REG[r], arm=S.NAME[k], key=k,
                                b=b if not one(k) else "-", chosen=chosen, value=m, ci_lo=lo, ci_hi=hi,
                                b_invariant=k in B_INVARIANT or (k in POOLS and set(re.findall(r"([\w\[\]=.]+) x\d+", chosen.split("|")[0])) <= B_INVARIANT)))
        if o: S.oracle(ax, i - 0.46, i + 0.46, 1.0 if norm else o[0])
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([xl for xl, _ in groups]); ax.set_xlim(-0.5, len(groups) - 0.5)
    S.finish(ax, ylabel=S.AXIS["rel" if norm else "success"], ylim=(0.2, 1.05 if norm else 0.95))
    keys = ["oracle"] + arms                                            # _allb: three rows, the budget swatches in the last column
    S.legend(ax, keys, ncol=-(-len(keys) // 2) if stacked else -(-len(keys) // 3) + 1,                # _stacked: two rows
             extra=None if stacked else S.budget_handles(-len(keys) % 3), **(dict(columnspacing=0.5, handlelength=1.0) if stacked else {}))
    if incomplete: print(f"[{name}] INCOMPLETE: a bar is missing or its pool is incomplete (see the csv)")
    S.save(fig, name, OUT); pd.DataFrame(rec).to_csv(f"{OUT}/{name}.csv", index=False)


def fig_A(C):
    ns = sorted(n for (f, g, n, r) in C if f == "live" and g == "specialist" and r == "beta0")
    g = list(zip(S.pow10_ticks(ns, "n"), [("live", "specialist", n) for n in ns]))
    budget_bars(C, g, "A_live_allb"); budget_bars(C, g, "A_live_stacked", stacked=True)


def primary(f, g): return g == PRIMARY.get(f, g)


def fig_B(C):
    """Each family at its LARGEST population with both regimes, divided by that cell's oracle."""
    groups = []
    for f in FAMILY:
        ns = [n for (ff, g, n, r) in C if ff == f and primary(f, g) and r == "cartel" and (ff, g, n, "beta0") in C]
        if not ns: continue
        n = max(ns); g = next(g for (ff, g, nn, r) in C if ff == f and nn == n and primary(f, g))
        groups.append((f"{S.BACKEND[f]}\n{S.pow10(n, 'n')}", (f, g, n)))
    budget_bars(C, groups, "B_families_allb", norm=True, skip=("best_framework",))
    budget_bars(C, groups, "B_families_stacked", norm=True, stacked=True, skip=("best_framework",))


def add_best_framework(C):
    """Per live n and regime, the seed x (framework | text shortlist) success table of every pair with the full seed count
    at that cell (framework rows via shortlist_figs.collect: b = 3, specialist); arms_at cross-fits the best pair per seed."""
    from shortlist_figs import collect
    for (n, dist, reg), cell in collect("live").items():
        key = ("live", "specialist", n, reg)
        if dist != "specialist" or key not in C: continue
        T = pd.DataFrame({f"{m} | {src}": v for m, d in cell["fw"].items() for src, v in d.items() if src in TEXT})
        if T.empty: continue
        full = T.notna().sum(); C[key]["fw_seeds"] = T[full.index[full == full.max()]]


def add_budgets(C):
    """b = 1 / 5 for every arm: live / RouterEval / LLMRouterBench / bernoulli 10^7 / replay 10^6 from the va_b_* (MIDIAN)
    and rivals_b_* (budget-matched rivals) rows (per-seed means, whiskers set by narrow()); bernoulli / replay b = 1 from their
    scale matrices. A cell the runs have not reached simply has no bar (figure_status.py lists it)."""
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
            if l != "oracle" and l not in ARMS: return v   # not drawn (pool members come from the tables)
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
    add_best_framework(C)
    fig_A(C); fig_B(C)                    # C / D: scripts/efficiency_figs.py
