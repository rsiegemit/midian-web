"""Efficiency figures C (appendix) and D (Figure 5); style, names and saving from scripts/figspec.py (docs/FIGURE_SPEC.md).
    python scripts/efficiency_figs.py   -> figures/condensed_sample/{C_routing_work_vs_n,D_energy_per_query}.{png,pdf,csv}
Two figures, one panel each (they replace the old C paired-gap and D heatmap figures):
  C   routing work per query vs n (10^2..10^7, calibrated bernoulli, b = 3, honest; exact ledger counts: messages +
     comparisons per routed task): MIDIAN, MIDIAN w/o defenses, "any flat scan" (flat probe argmax, declared argmax and
     every bandit scan all n agents: their counts coincide, checked), the learned routers as a band; legend: growth() of
     the log-log slope over n >= 10^3 (log n, n^s).
  D   energy per routed query vs queries served: (build J) / T + marginal J per query. MIDIAN at n = 10^3, 10^5, 10^7
     (its build probes from the same ledger); the frameworks' per-query supervisor energy (live n = 1,000 measurement,
     scripts/energy.py) as a band. Energy model = scripts/energy.py (probe 4.04 J on specialist, 7B supervisor call
     20.6 J, message 1e-3 J, comparison 1e-8 J; 700 W). The routed task's own execution is common to all and excluded.
Costs come from bernoulli_scale_v5 rows, cached to figures/condensed_sample/cost_by_n.csv (delete it to re-read)."""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figspec as S
from seed_tables import label
from rte.methods import keys
from condensed_figs import BANDIT, LEARNED, NOT_RUNNABLE, OUT, POOLS

R = os.environ.get("RTE_DATA", "/scratch/rte") + "/results"
COST = f"{OUT}/cost_by_n.csv"
COLS = ["n", "b", "beta", "method", "params", "messages_per_task", "comparisons_per_task", "build_probes", "build_messages"]
FIXED = ["midian", "midian_wo_defenses", "flat_probe_argmax_online", "declared_argmax"]      # C: the lines; the pools as FLAT_SCAN / the band
SHOW = set(FIXED) | set(LEARNED) | set(BANDIT)                                      # + every member of the two pools


def costs():
    if os.path.exists(COST): return pd.read_csv(COST)
    parts = [c[((c.b == 3) | ((c.method == "midian") & (c.params == "{}"))) & (c.beta == 0)]   # MIDIAN at every b: D's build ledger
             for c in (keys.normalize(c, f"{R}/bernoulli_scale_v5")
                       for c in pd.read_csv(f"{R}/bernoulli_scale_v5/rows.csv", usecols=COLS, chunksize=500000, low_memory=False))]
    d = pd.concat(parts, ignore_index=True)
    d["label"] = [label(m, str(p)) for m, p in zip(d.method, d.params)]
    d = d[d.label.isin(list(SHOW))].groupby(["label", "n", "b"])[COLS[5:]].median().reset_index()
    d.to_csv(COST, index=False); return d


VA_GRIDS = ["va_b_bernoulli_1e7", "va_b_n1000", "va_b_n100k", "fw_live_n1000", "live_n100k"]   # exact MIDIAN build ledgers by (n, b)


def va_build(d):
    """{(n, b): (build probes, source)} for MIDIAN at n = 10^3, 10^5, 10^7 and b = 1, 3, 5, from the exact ledger
    wherever that cell ran (the live / va_b grids first, then the bernoulli sweep); else the measured build / (n K b)
    ratio at that b (no cell needs it today)."""
    import glob, json
    fr = []
    for g in VA_GRIDS:
        fr += [keys.normalize(pd.DataFrame([{**json.load(open(f)), "rid": os.path.basename(f)[:-5]} for f in glob.glob(f"{R}/{g}/rows.d/*.json")]), f"{R}/{g}")]
        if os.path.exists(f"{R}/{g}/rows.csv"):
            head = pd.read_csv(f"{R}/{g}/rows.csv", nrows=0).columns
            fr.append(keys.normalize(pd.read_csv(f"{R}/{g}/rows.csv", usecols=[c for c in ("rid", "n", "b", "method", "params", "build_probes") if c in head],
                                                 low_memory=False), f"{R}/{g}"))
    L = pd.concat([f for f in fr if not f.empty])
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


def draw_D(ax, d):
    import energy
    from doc_tables import NAMES
    fw = energy.table(); fw = fw[fw.index.str.startswith("fw_") & ~fw.index.str.contains("supervisor")]
    fwJ = fw.per_task_J                                  # supervisor call energy + the framework's messages / comparisons
    PJ = energy.probe_cost["specialist"] * 700
    T = np.logspace(2, 9, 200); rec = []
    ax.fill_between(T, fwJ.min(), fwJ.max(), color=S.COLOR["fw_band"], lw=0, zorder=1)
    edge = blended_transform_factory(ax.transAxes, ax.transData)
    for v, m in ((fwJ.min(), fwJ.idxmin()), (fwJ.max(), fwJ.idxmax())):     # both edges drawn and named at the right margin
        ax.axhline(v, color=S.COLOR["grey"], lw=0.6, zorder=2)
        ax.text(1.01, v, f"{NAMES.get(m, m)}, {v:.3g} J", transform=edge, va="center", ha="left", fontsize=8)
    va = d[(d.label == "midian") & (d.b == 3)].set_index("n"); VB = va_build(d)
    for n, lvl in N_SHADE.items():
        r = va.loc[n]; marg = r.messages_per_task * energy.J_MSG + r.comparisons_per_task * energy.J_CMP
        for b, ls in S.B_LINESTYLE.items():
            probes, src = VB[(n, b)]
            build = probes * PJ + r.build_messages * energy.J_MSG          # build messages: the b = 3 ledger (a rounding term)
            ax.plot(T, build / T + marg, ls, lw=1.2, color=S.shade(S.COLOR["midian"], lvl), zorder=3)
            for lo, hi in ((fwJ.min(), "cheapest"), (fwJ.max(), "costliest")):
                t = build / (lo - marg); ax.plot(t, lo, "x", color="black", ms=3, zorder=5)
                rec.append(dict(n=n, b=b, build_probes=probes, build_source=src, build_J=build, marginal_J=marg, vs=hi, fw_J=lo, break_even_queries=t))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(T[0], T[-1])
    ax.set_ylim(1e-6, None)                              # room below the curves for the legend (bottom left)
    S.finish(ax, ylabel=S.AXIS["energy"], xlabel=S.AXIS["T"])
    hs = [Line2D([], [], color=S.shade(S.COLOR["midian"], lvl), lw=2.5) for lvl in N_SHADE.values()] + \
         [Line2D([], [], color="black", ls=ls, lw=1.2) for ls in S.B_LINESTYLE.values()]
    S.legend(ax, [], labels=[S.pow10(n, "n") for n in N_SHADE] + [f"b = {b}" for b in S.B_LINESTYLE], handles=hs, where="lower left", ncol=2, labelspacing=0.2, borderaxespad=0.2)
    return pd.DataFrame(rec)


def fig(draw, name, d, kind):
    f, ax = S.figure(kind)
    draw(ax, d).to_csv(f"{OUT}/{name}.csv", index=False); S.save(f, name, OUT)


if __name__ == "__main__":
    d = costs(); fig(draw_C, "C_routing_work_vs_n", d, "appendix"); fig(draw_D, "D_energy_per_query", d, "d")
