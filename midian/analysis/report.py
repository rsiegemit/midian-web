"""summary.md for a grid (tables, targets, figures list) and the `python -m midian.analyze` entry point."""
import argparse
import os
import warnings

import pandas as pd

from .legacy_figures import figures
from .legacy_targets import targets, targets_v2
from .load import BUILD, COST, FLOOR, REF, RTE_DATA, STATS, cells, load, log, reads_declared
from .stats import aggregate, boot, paired


def md(df, f="{:.4f}"):
    if df is None or df.empty:
        return "_(no rows)_\n"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f.format(v))
    head = "| " + " | ".join(map(str, d.columns)) + " |\n| " + " | ".join("---" for _ in d.columns) + " |\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)) + "\n"
def by_method(df, extra=()):
    """Mean +- bootstrap CI of success per method, grouped by method class, framework accountings and costs beside
    it."""
    g = df.groupby(["group", "label"])
    cols = [c for c in ("success_late", *STATS, "regret", "misroute_to_liar", *extra) if c in df.columns]
    return (g.success.apply(lambda s: pd.Series(dict(zip(("success", "lo", "hi"), boot(s))))).unstack()
            # variance across (cell, seed) units
            .join(g.success.std().rename("sd_units")).join(g.size().rename("units"))
            .join(g[cols].mean()).sort_values("success", ascending=False).reset_index())
def roll_up(cmp_):
    """Per rival, across cells: mean paired delta, its CI, sign-test p and how often the reference (MIDIAN w/o
    defenses) wins."""
    n = lambda tag: ("verdict", lambda s: int((s == tag).sum()))
    return cmp_.groupby(["group", "rival"]).agg(
        cells=("delta_mean", "size"), delta_mean=("delta_mean", "mean"), delta_lo=("delta_lo", "mean"),
        delta_hi=("delta_hi", "mean"), delta_min=("delta_mean", "min"), delta_max=("delta_mean", "max"),
        midian_better=n("midian_better"), rival_better=n("rival_better"), within_floor=n(FLOOR),
        min_sign_p=("sign_p", "min")).sort_values("delta_mean").reset_index()
def strict(df):
    """Frameworks under the strict accounting (a task the framework did not delegate scores 0), paired vs MIDIAN w/o
    defenses."""
    f = df[(df.group == "framework") & df.success_strict.notna()]
    if f.empty:
        return []
    d = pd.concat([f.assign(success=f.success_strict), df[df.label == REF]])
    c = paired(d)
    if c.empty:
        return []
    return sec("Frameworks, STRICT accounting (no fallback credit), paired vs MIDIAN w/o defenses", md(roll_up(c)),
               "Lenient (headline) routes the fallback pick and scores it; strict scores every non-delegated task 0. "
               "`fallback_rate` = 1 - picks/tasks under either accounting.")
UPPER = "programmatic = upper bound (S + N(0,0.05)): an honest declaration no live agent produces"
def sec(t, table, *prose, level=2):
    return ["", f"{'#' * level} {t}", "", *prose, *([""] if prose else []), table]
def by_channel(df, cmp_):
    """Success tables per declaration channel: declared-channel readers are never pooled across channels
    (x beta, x dist and the paired-vs-MIDIAN w/o defenses roll-up per channel); probe-only methods once, identical by
    construction."""
    piv = lambda d, col: md(d.pivot_table(index=["group", "label"], columns=col, values="success").reset_index())
    dec = df.method.map(reads_declared)
    chans = sorted(df.declared_source.unique(), reverse=True)          # self_described first
    L = []
    for ch in chans if chans[1:] else []:
        d = df[dec & (df.declared_source == ch)]
        cap = f"declaration = {ch}; " + (UPPER if ch == "programmatic"
                                         else "the live channel: agents' own self-descriptions")
        for col in ("beta", "dist"):
            L += sec(f"Declared-channel readers, success x {col} [{ch}]", piv(d, col), cap, level=3)
        c = cmp_[cmp_.rival.isin(d.label.unique()) & (cmp_.declared_source == ch)] if not cmp_.empty else cmp_
        if not c.empty:
            L += sec(f"MIDIAN w/o defenses vs declared-channel readers, paired by seed [{ch}]", md(roll_up(c)), cap,
                     level=3)
    rest = df if not chans[1:] else df[~dec]
    note = (f"single declaration channel: {chans[0]}" + (f"; {UPPER}" if chans == ["programmatic"] else "")
            if not chans[1:] else "probe-only methods: identical across channels")
    return L + sec("Success x beta", piv(rest, "beta"), note, level=3)
def latency(df):
    """The one wall-clock table: frameworks' supervisor call per task. Their clients call vLLM directly and are
    never memoised, so these are cache-consistent; they are latencies under shared-fleet load, not compute costs."""
    f = df[(df.group == "framework") & (df.wall_clock_per_task > 0)] if "wall_clock_per_task" in df else pd.DataFrame()
    if f.empty:
        return []
    q = f.groupby(["label", "n"]).wall_clock_per_task.quantile([.25, .5, .75]).unstack()
    q.columns = ["q25_s", "median_s", "q75_s"]
    return sec("Frameworks' supervisor latency per task (seconds; cache-consistent, under shared-fleet load)",
               md(q.reset_index(), "{:.2f}"),
               "Every other wall-clock column is omitted: memo hits and misses are mixed and say nothing about cost.")
def summary(out, grids, df, agg, cmp_, fits, tgts, tgts2, figs):
    L = [f"# RTE results -- {', '.join(grids)}", "",
         f"{len(df)} rows | {df.label.nunique()} methods | "
         f"{df.groupby(cells(df), dropna=False).ngroups} cells | seeds {sorted(map(int, df.seed.unique()))}",
         f"Backends: {', '.join(sorted(df.backend.astype(str).unique()))}.", "",
         "Intervals are 95% percentile bootstrap over seeds; MIDIAN w/o defenses-vs-rival deltas are paired by",
         f"seed. `{FLOOR}` means the delta does not exceed MIDIAN w/o defenses' own seed envelope in that cell.",
         *sec("Pre-registered targets (TARGETS_rte.md)",
              md(pd.DataFrame(tgts)[["target", "verdict", "name", "detail"]])),
         *sec("Pre-registered targets, v2 (TARGETS_rte_v2.md)",
              md(pd.DataFrame(tgts2)[["target", "verdict", "name", "detail"]])),
         *sec("HEADLINE: frameworks vs MIDIAN, by method class (SPEC 6A)",
              md(by_method(df, COST + BUILD), "{:.4g}"),
              "Frameworks are the systems practitioners deploy, run through their own libraries; every",
              "one reads names and self-descriptions only. `fallback_rate` / `success_strict` (0.2) are the framework",
              "accountings; the other groups are SPEC 6 mechanism controls."),
         *strict(df), "", "## Success by method, per declaration channel", *by_channel(df, cmp_), *latency(df)]
    if not cmp_.empty:
        L += sec("MIDIAN w/o defenses vs every rival, paired by seed", md(roll_up(cmp_)))
        L += ["", "### Per-cell detail", "", md(cmp_.drop(columns="ref"))]
    if not fits.empty:
        L += sec("Cost exponents (cost ~ n^k)", md(fits.sort_values(["metric", "exponent"]), "{:.3f}"),
                 "MIDIAN's per-task messages are 2*ceil(log_r n): logarithmic, so its fitted exponent sits",
                 "near 0. Flat table lookups send 0 messages per task and CNP 2n: exponents 0 and 1.")
    if figs:
        L += ["", "## Figures", ""] + [f"- {k}: `{os.path.relpath(v, out)}`" for k, v in sorted(figs.items())]
    L += sec("Per-cell aggregate (success)", md(agg))
    path, tmp = f"{out}/summary.md", f"{out}/summary.md.tmp{os.getpid()}"
    open(tmp, "w").write("\n".join(L) + "\n")
    os.replace(tmp, path)
    return path
def main(argv=None):
    p = argparse.ArgumentParser("midian.analyze")
    p.add_argument("--grid", required=True)
    p.add_argument("--grids")
    p.add_argument("--metric", default="success")
    p.add_argument("--out")
    a = p.parse_args(argv)
    grids = [a.grid] + [g.strip() for g in (a.grids or "").split(",") if g.strip() and g.strip() != a.grid]
    df = load(grids)
    out = a.out or f"{RTE_DATA}/results/{a.grid}"
    os.makedirs(out, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        figs, fits = figures(df, f"{out}/figs")
    agg, cmp_ = aggregate(df, a.metric), paired(df[df.method != "oracle"], a.metric)
    tgts, tgts2 = targets(df, fits), targets_v2(df, fits)
    log("\n=== pre-registered targets (v1, v2) ===")
    for t in tgts + tgts2:
        log(f"  [{t['verdict']:>12}] {t['target']}. {t['name']}\n                 {t['detail']}")
    agg.to_csv(f"{out}/aggregate.csv", index=False)
    cmp_.to_csv(f"{out}/paired_vs_midian.csv", index=False)
    if not fits.empty:
        fits.to_csv(f"{out}/cost_exponents.csv", index=False)
    log(f"\n[analyze] wrote {summary(out, grids, df, agg, cmp_, fits, tgts, tgts2, figs)}")
