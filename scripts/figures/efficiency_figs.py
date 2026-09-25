"""Efficiency figures C (appendix) and D (Figure 5); style, names and saving from scripts/figspec.py (docs/FIGURE_SPEC.md).
    python scripts/figures/efficiency_figs.py [--out DIR] [--refresh] [--from-csv]
      -> <out>/{C_routing_work_vs_n,D_energy_per_query}.{png,pdf,csv}   (<out> = $RTE_FIG_OUT or figures/paper)
Two figures, one panel each (they replace the old C paired-gap and D heatmap figures):
  C   routing work per query vs n (10^2..10^7, calibrated bernoulli, b = 3, honest; exact ledger counts: messages +
     comparisons per routed task): MIDIAN, MIDIAN w/o defenses, "any flat scan" (flat probe argmax, declared argmax and
     every bandit scan all n agents: their counts coincide, checked), the learned routers as a band; legend: growth() of
     the log-log slope over n >= 10^3 (log n, n^s).
  D   energy per routed query vs queries served: (build J) / T + marginal J per query. MIDIAN at n = 10^3, 10^5, 10^7
     (its build probes from the same ledger); the frameworks' per-query supervisor energy (live n = 1,000 measurement,
     scripts/analysis/energy.py) as a band. Energy model = scripts/analysis/energy.py (probe 4.04 J on specialist, 7B supervisor call
     20.6 J, message 1e-3 J, comparison 1e-8 J; 700 W). The routed task's own execution is common to all and excluded.
Costs come from bernoulli_scale_v5 rows, cached to results/aggregates/cost_by_n.csv (--refresh re-reads the rows).
--from-csv: C from that cache, D from results/aggregates/figures/D_energy_per_query.csv + refs.csv (no $RTE_DATA)."""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from scripts.figures.condensed_figs import BANDIT, LEARNED, NOT_RUNNABLE, POOLS                               # noqa: E402
from scripts.figures.lib import AGG, fig_out                                                                   # noqa: E402
from scripts.figures.lib import figspec as S                                                                   # noqa: E402
from scripts.figures.lib.grids import MATRICES, VA_GRIDS                                                       # noqa: E402
from scripts.figures.lib.rows import label, read_rows                                                          # noqa: E402

COST, DRAW = f"{AGG}/cost_by_n.csv", f"{AGG}/figures"
COLS = ["n", "b", "beta", "method", "params", "messages_per_task", "comparisons_per_task", "build_probes", "build_messages"]
FIXED = ["midian", "midian_wo_defenses", "flat_probe_argmax_online", "declared_argmax"]      # C: the lines; the pools as FLAT_SCAN / the band
SHOW = set(FIXED) | set(LEARNED) | set(BANDIT)                                      # + every member of the two pools


def costs(refresh=False):
    if os.path.exists(COST) and not refresh: return pd.read_csv(COST)
    keep = lambda c: c[((c.b == 3) | ((c.method == "midian") & (c.params == "{}"))) & (c.beta == 0)]   # MIDIAN at every b: D's build ledger
    d = read_rows(MATRICES["bernoulli"], COLS, where=keep, rowsd=False).reset_index(drop=True)
    d["label"] = [label(m, str(p)) for m, p in zip(d.method, d.params)]
    d = d[d.label.isin(list(SHOW))].groupby(["label", "n", "b"])[COLS[5:]].median().reset_index()
    d.to_csv(COST, index=False); return d


def va_build(d):
    """{(n, b): (build probes, source)} for MIDIAN at n = 10^3, 10^5, 10^7 and b = 1, 3, 5, from the exact ledger
    wherever that cell ran (the live / va_b grids first, then the bernoulli sweep); else the measured build / (n K b)
    ratio at that b (no cell needs it today)."""
    L = pd.concat([f for f in (read_rows(g, ("rid", "n", "b", "method", "params", "build_probes")) for g in VA_GRIDS) if not f.empty])
    L = L.drop_duplicates("rid")                                             # a row can sit in both rows.d and rows.csv
    L = L[(L.method == "midian") & (L.params.fillna("{}").astype(str) == "{}")].groupby(["n", "b"]).build_probes.median()   # the full method
    for (n, b), v in d[d.label == "midian"].set_index(["n", "b"]).build_probes.items(): L.loc[(n, b)] = L.get((n, b), v)
    ratio = {b: float((L.xs(b, level="b") / (L.xs(b, level="b").index * 16 * b)).median()) for b in (1, 3, 5)}
    return {(n, b): ((float(L[(n, b)]), "ledger") if (n, b) in L.index else (ratio[b] * n * 16 * b, f"ratio {ratio[b]:.3f} x nKb"))
            for n in (1000, 100000, 10 ** 7) for b in (1, 3, 5)}


def slope(q):
    q = q[(q.n >= 1000) & (q.work > 0)]
    return np.polyfit(np.log10(q.n), np.log10(q.work), 1)[0] if len(q) > 1 else np.nan


def growth(s):
    """Legend wording for a fitted log-log slope: MIDIAN's work is linear in tree depth (log n), not a power law."""
    return "constant" if abs(s) < 0.02 else r"$\propto \log n$" if s < 0.3 else rf"$\propto n^{{{s:.2f}}}$"


FLAT_SCAN = ["flat_probe_argmax_online", "declared_argmax"] + POOLS["best_bandit"]   # one line: every one scans all n agents


def draw_C(ax, d):
    """MIDIAN, MIDIAN w/o defenses, any flat scan (lines) and the learned routers (the band between the cheapest and the
    costliest runnable member at each n; constant-cost members, flat_nsw, left out). Random does no routing work."""
    d = d[d.b == 3].assign(work=lambda x: x.messages_per_task + x.comparisons_per_task); rec = []; hs, ls = [], []
    runnable = lambda q: q[~pd.Series([l in NOT_RUNNABLE("bernoulli", n) for l, n in zip(q.label, q.n)], index=q.index)]
    flat = runnable(d[d.label.isin(FLAT_SCAN) & (d.n >= 100)]).groupby("n").work
    dev = float(((flat.max() - flat.min()) / flat.min()).max())
    print(f"[C] any flat scan: members {sorted(set(d[d.label.isin(FLAT_SCAN)].label))} differ by at most {dev:.1%} at any n")
    for k in ("midian", "midian_wo_defenses", "flat_scan"):
        q = d[(d.label == ("flat_probe_argmax_online" if k == "flat_scan" else k)) & (d.n >= 100)].sort_values("n")
        s = slope(q); rec += [dict(arm=S.NAME[k], key=k, n=int(n), work=w, slope=s) for n, w in zip(q.n, q.work)]
        ax.plot(q.n, q.work, "o-", ms=3, lw=1.4, color=S.COLOR[k])
        hs.append(Line2D([], [], color=S.COLOR[k], marker="o", ms=3, lw=1.4)); ls.append(f"{S.NAME[k]} ({growth(s)})")
    k, c = "best_learned", S.COLOR["best_learned"]
    q = runnable(d[d.label.isin(POOLS[k])])
    const = [l for l, g in q[q.n >= 100].groupby("label") if growth(slope(g)) == "constant"]
    q = q[(q.n >= 100) & ~q.label.isin(const)].groupby("n").work.agg(["min", "max"]).reset_index()
    lo, hi = slope(q.rename(columns={"min": "work"})), slope(q.rename(columns={"max": "work"}))
    ax.fill_between(q.n, q["min"], q["max"], color=c, alpha=0.3, lw=0)
    for e in ("min", "max"): ax.plot(q.n, q[e], "-", lw=1.0, color=c)
    a, z = sorted((lo, hi))
    span = growth(lo) if growth(lo) == growth(hi) else rf"$\propto n^{{{a:.2f}}}$ to $n^{{{z:.2f}}}$" if a >= 0.3 else f"{growth(lo)} to {growth(hi)}"
    hs.append(Patch(facecolor=c, alpha=0.3, edgecolor=c, lw=1.0)); ls.append(f"{S.NAME[k]} ({span})")
    rec += [dict(arm=S.NAME[k], key=k, n=int(n), work=w, work_max=m, slope=lo, slope_max=hi) for n, w, m in zip(q.n, q["min"], q["max"])]
    ax.set_xscale("log"); ax.set_yscale("log")
    S.finish(ax, ylabel=S.AXIS["work"], xlabel=S.AXIS["n"])
    S.legend(ax, [], labels=ls, handles=hs, where="upper left")
    return pd.DataFrame(rec)


N_SHADE = {1000: 1, 100000: 3, 10 ** 7: 5}                 # n = 10^3 light, 10^5 mid, 10^7 dark (figspec.shade levels)


def compute_D(d):
    """D's table (one row per (n, b, framework edge)) and the names of the cheapest / costliest framework."""
    from scripts.analysis import energy
    fw = energy.table(); fw = fw[fw.index.str.startswith("fw_") & ~fw.index.str.contains("supervisor")]
    fwJ = fw.per_task_J                                  # supervisor call energy + the framework's messages / comparisons
    PJ = energy.probe_cost["specialist"] * 700
    rec = []
    va = d[(d.label == "midian") & (d.b == 3)].set_index("n"); VB = va_build(d)
    for n in N_SHADE:
        r = va.loc[n]; marg = r.messages_per_task * energy.J_MSG + r.comparisons_per_task * energy.J_CMP
        for b in S.B_LINESTYLE:
            probes, src = VB[(n, b)]
            build = probes * PJ + r.build_messages * energy.J_MSG          # build messages: the b = 3 ledger (a rounding term)
            for lo, hi in ((fwJ.min(), "cheapest"), (fwJ.max(), "costliest")):
                rec.append(dict(n=n, b=b, build_probes=probes, build_source=src, build_J=build, marginal_J=marg, vs=hi, fw_J=lo,
                                break_even_queries=build / (lo - marg)))
    refs = pd.DataFrame([dict(figure="D_energy_per_query", key=k, value=v, text=S.FW_NAME.get(m, m))
                         for k, v, m in (("cheapest", fwJ.min(), fwJ.idxmin()), ("costliest", fwJ.max(), fwJ.idxmax()))])
    return pd.DataFrame(rec), refs


def render_D(ax, rec, refs):
    """The framework band with both edges named, MIDIAN's (build J) / T + marginal J per (n, b), break-evens as x."""
    T = np.logspace(2, 9, 200)
    e = refs.set_index("key")
    ax.fill_between(T, e.value["cheapest"], e.value["costliest"], color=S.COLOR["fw_band"], lw=0, zorder=1)
    edge = blended_transform_factory(ax.transAxes, ax.transData)
    for k in ("cheapest", "costliest"):                  # both edges drawn and named at the right margin
        v = e.value[k]
        ax.axhline(v, color=S.COLOR["grey"], lw=0.6, zorder=2)
        ax.text(1.01, v, f"{e.text[k]}, {v:.3g} J", transform=edge, va="center", ha="left", fontsize=8)
    for i in range(0, len(rec), 2):                      # one curve per (n, b), then its two break-evens
        r = rec.iloc[i]
        ax.plot(T, r.build_J / T + r.marginal_J, S.B_LINESTYLE[int(r.b)], lw=1.2, color=S.shade(S.COLOR["midian"], N_SHADE[int(r.n)]), zorder=3)
        for _, q in rec.iloc[i:i + 2].iterrows():
            ax.plot(q.break_even_queries, q.fw_J, "x", color="black", ms=3, zorder=5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(T[0], T[-1])
    ax.set_ylim(1e-6, None)                              # room below the curves for the legend (bottom left)
    S.finish(ax, ylabel=S.AXIS["energy"], xlabel=S.AXIS["T"])
    hs = [Line2D([], [], color=S.shade(S.COLOR["midian"], lvl), lw=2.5) for lvl in N_SHADE.values()] + \
         [Line2D([], [], color="black", ls=ls, lw=1.2) for ls in S.B_LINESTYLE.values()]
    S.legend(ax, [], labels=[S.pow10(n, "n") for n in N_SHADE] + [f"b = {b}" for b in S.B_LINESTYLE], handles=hs, where="lower left", ncol=2, labelspacing=0.2, borderaxespad=0.2)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", help="output directory (default: $RTE_FIG_OUT or figures/paper)")
    p.add_argument("--refresh", action="store_true", help="re-read the bernoulli rows into results/aggregates/cost_by_n.csv")
    p.add_argument("--from-csv", action="store_true", help="draw from results/aggregates only")
    a = p.parse_args(argv)
    out = fig_out(a.out)
    d = costs(a.refresh and not a.from_csv)
    f, ax = S.figure("appendix")
    draw_C(ax, d).to_csv(f"{out}/C_routing_work_vs_n.csv", index=False); S.save(f, "C_routing_work_vs_n", out)
    name, refs_csv = "D_energy_per_query", f"{DRAW}/refs.csv"
    if a.from_csv:
        rec, refs = pd.read_csv(f"{DRAW}/{name}.csv"), pd.read_csv(refs_csv)
        refs = refs[refs.figure == name]
        shutil.copyfile(f"{DRAW}/{name}.csv", f"{out}/{name}.csv")
    else:
        rec, refs = compute_D(d)
        os.makedirs(DRAW, exist_ok=True)
        rec.to_csv(f"{out}/{name}.csv", index=False); rec.to_csv(f"{DRAW}/{name}.csv", index=False)
        old = pd.read_csv(refs_csv) if os.path.exists(refs_csv) else pd.DataFrame(columns=refs.columns)
        pd.concat([old[old.figure != name], refs], ignore_index=True).to_csv(refs_csv, index=False)
    f, ax = S.figure("d")
    render_D(ax, rec, refs); S.save(f, name, out)


if __name__ == "__main__":
    main()
