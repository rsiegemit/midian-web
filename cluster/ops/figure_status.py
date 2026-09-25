"""What the figures still lack, read from their own CSVs, plus planned-vs-landed rows of the grids still running.
    RTE_DATA=... python cluster/ops/figure_status.py [--out DIR]
A / B: every (group, arm, regime, b) slot the figure draws (declared argmax / random: b = 3 only) -> missing slots, and bars whose
pool is INCOMPLETE (a member absent or short of seeds). E-H: every (n, regime, shortlist) slot -> missing, rerun outstanding
(star), fewer than MIN_FW frameworks. Grids: set comparison of planned vs landed (lib/seed_tables.planned / rows). DIR: the figures' directory ($RTE_FIG_OUT or figures/paper).."""
import itertools, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import scripts.figures.lib.seed_tables as t                                                                   # noqa: E402
from scripts.figures.lib import fig_out                                                                       # noqa: E402
from scripts.figures.lib.grids import H30_FW, H30_REF                                                         # noqa: E402
from scripts.figures.lib.rows import label                                                                    # noqa: E402
from scripts.figures.shortlist_condensed import MAIN, MIN_FW                                                  # noqa: E402

OUT = fig_out(sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None)
GRIDS = ["replay_1e6_split_cal", "bernoulli_1e7_cal", "routereval5k_norep_cal", "llmrouterbench_norep_cal", "rivals_b_n100k",
         "pool_fill_n100k", "linucb_fix_n100k", "linucb_fix_bernoulli_1e7", "trueskill_fix_n100", "trueskill_fix_n1000",
         "trueskill_fix_n10k", "linucb_fix_n100", "linucb_fix_n1000", "linucb_fix_n10k"]


def ab(name):
    d = pd.read_csv(f"{OUT}/{name}.csv")
    d["b"] = d.b.astype(str)
    arms, groups, regs = d.arm.unique(), d.group.unique(), d.regime.unique()
    budgetless = ("declared argmax", "random", "best framework")   # spend no probes: one bar, b does not apply
    want = {(g, a, r, b) for g in groups for a in arms for r in regs for b in (("-",) if a in budgetless else ("1", "3", "5"))}
    miss = sorted(want - set(zip(d.group, d.arm, d.regime, d.b)))
    inc = d[d.chosen.astype(str).str.contains("INCOMPLETE")]
    print(f"{name}: {len(d)} bars, {len(miss)} missing slots, {len(inc)} incomplete-pool bars")
    for m in miss: print("   MISSING", m)
    for (g, m), q in inc.assign(m=inc.chosen.str.extract(r"partial (.*)")[0]).groupby(["group", "m"]):
        print(f"   INCOMPLETE {g}: {m}  ({len(q)} bars)")


def shortlist(name, srcs=None):
    d = pd.read_csv(f"{OUT}/{name}.csv"); k = "frameworks" if "frameworks" in d else "k"
    keys = ["regime", "shortlist"] + (["n"] if "n" in d else [])
    if "n" in d and srcs:
        want = set(itertools.product(d.regime.unique(), srcs, d.n.unique()))
        miss = sorted(want - set(zip(*[d[c] for c in keys])))
    else: miss = []
    print(f"{name}: {len(d)} bars, {len(miss)} missing slots, {int(d.star.sum())} rerun-outstanding, {int((d[k] < MIN_FW).sum())} under MIN_FW")
    for m in miss: print("   MISSING", m)


def grids():
    for g in GRIDS + H30_FW + sorted({x for v in H30_REF.values() for x in v}):
        w = t.planned(g)
        if g.startswith(("fw_", "re_sl_")):                     # framework grids: rows() drops fw_ arms, so read them directly
            from scripts.figures.shortlist_figs import rows as fwrows
            df = fwrows(g); df = df if df.empty else df[df.method.astype(str).str.startswith("fw_") | (df.method == "oracle")]
        else: df = pd.concat([t.rows(g, n) for n in {x[0] for x in w}], ignore_index=True)
        have = set() if df.empty else {(int(n), int(b), str(d), float(be), str(ls), int(s), label(m, p)) for n, b, d, be, ls, s, m, p
                                       in zip(df.n, df.b, df.dist, df.beta, df.liar_select, df.seed, df.method, df.params.astype(str))}
        print(f"   {g:28s} {len(w & have):5d}/{len(w):5d}{'  DONE' if w <= have else ''}")


if __name__ == "__main__":
    for f in ("A_live_allb", "B_families_allb"): ab(f)
    shortlist("E_shortlists_by_n", MAIN); shortlist("F_shortlists_1e5"); shortlist("G_shortlist_lift_1e5")
    shortlist("H_routereval_shortlists", ["tfidf", "embed", "dense", "sota", "declared", "va_cohort"])   # body: best instruction variant per family
    print("switched to erratum-30 rows:", sorted(t.switched()))
    print("grids planned vs landed:"); grids()
