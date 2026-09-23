"""SAMPLE efficiency figures in the condensed A / B style (one panel, one row each).
    python scripts/efficiency_figs.py   -> figures/condensed_sample/I_efficiency.{png,pdf} + I_routing_work_vs_n.csv, I_energy_per_query.csv
One figure, two panels in one row:
  (a) routing work per query vs n (10^2..10^7, calibrated bernoulli, b = 3, honest; exact ledger counts: messages +
     comparisons per routed task), one line per arm, fitted log-log slope over n >= 10^3 in the legend.
  (b) energy per routed query vs queries served: (build J) / T + marginal J per query. MIDIAN-VA at n = 10^3, 10^5, 10^7
     (its build probes from the same ledger); the frameworks' per-query supervisor energy (live n = 1,000 measurement,
     scripts/energy.py) as a band. Energy model = scripts/energy.py (probe 4.04 J on specialist, 7B supervisor call
     20.6 J, message 1e-3 J, comparison 1e-8 J; 700 W). The routed task's own execution is common to all and excluded.
Costs come from bernoulli_scale_v5 rows, cached to figures/condensed_sample/cost_by_n.csv (delete it to re-read)."""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extra_figs  # noqa: F401  (legend rule: best top-left, row-major)
from seed_tables import label
from condensed_figs import ARMS, OUT, save

R = os.environ.get("RTE_DATA", "/scratch/rte") + "/results"
COST = f"{OUT}/cost_by_n.csv"
COLS = ["n", "b", "beta", "method", "params", "messages_per_task", "comparisons_per_task", "build_probes", "build_messages"]
SHOW = {"midian_va": "MIDIAN-VA", "midian": "MIDIAN", "flat_probe_argmax_online": "flat probe argmax (online)",
        "warm_start_bandit": "warm-start bandit", "ucb_per_family": "UCB bandit", "flat_nsw_router": "flat NSW index router",
        "declared_argmax": "declared argmax"}
COL = {k: c for k, _, c in ARMS} | {"warm_start_bandit": "#9467bd", "ucb_per_family": "#c5b0d5", "flat_nsw_router": "#ff7f0e"}


def costs():
    if os.path.exists(COST): return pd.read_csv(COST)
    parts = [c[(c.b == 3) & (c.beta == 0)] for c in pd.read_csv(f"{R}/bernoulli_scale_v5/rows.csv", usecols=COLS, chunksize=500000, low_memory=False)]
    d = pd.concat(parts, ignore_index=True)
    d["label"] = [label(m, str(p)) for m, p in zip(d.method, d.params)]
    d = d[d.label.isin(list(SHOW))].groupby(["label", "n"])[COLS[5:]].median().reset_index()
    d.to_csv(COST, index=False); return d


VA_GRIDS = ["va_b_bernoulli_1e7", "va_b_n1000", "va_b_n100k", "fw_live_n1000", "live_n100k"]   # exact MIDIAN-VA build ledgers by (n, b)


def va_build(d):
    """{(n, b): (build probes, source)} for MIDIAN-VA at n = 10^3, 10^5, 10^7 and b = 1, 3, 5, from the exact ledger
    wherever that cell ran; else the measured build / (n K b) ratio at that b (the only such cell: n = 10^7, b = 1)."""
    import glob, json
    fr = []
    for g in VA_GRIDS:
        fr += [pd.DataFrame([json.load(open(f)) for f in glob.glob(f"{R}/{g}/rows.d/*.json")])]
        if os.path.exists(f"{R}/{g}/rows.csv"): fr.append(pd.read_csv(f"{R}/{g}/rows.csv", usecols=["n", "b", "method", "build_probes"], low_memory=False))
    L = pd.concat([f for f in fr if not f.empty]); L = L[L.method == "midian_va"].groupby(["n", "b"]).build_probes.median()
    for n, v in d[d.label == "midian_va"].set_index("n").build_probes.items(): L.loc[(n, 3)] = L.get((n, 3), v)
    ratio = {b: float((L.xs(b, level="b") / (L.xs(b, level="b").index * 16 * b)).median()) for b in (1, 3, 5)}
    return {(n, b): ((float(L[(n, b)]), "ledger") if (n, b) in L.index else (ratio[b] * n * 16 * b, f"ratio {ratio[b]:.3f} x nKb"))
            for n in (1000, 100000, 10 ** 7) for b in (1, 3, 5)}


def slope(q):
    q = q[(q.n >= 1000) & (q.work > 0)]
    return np.polyfit(np.log10(q.n), np.log10(q.work), 1)[0] if len(q) > 1 else np.nan


def draw_I(ax, d):
    d = d.assign(work=d.messages_per_task + d.comparisons_per_task); rec = []
    for l, name in SHOW.items():
        q = d[(d.label == l) & (d.n >= 100)].sort_values("n")
        if q.empty or (q.work <= 0).all(): continue
        s = slope(q); rec += [dict(arm=name, n=int(n), work=w, slope=s) for n, w in zip(q.n, q.work)]
        ax.plot(q.n, q.work, "o-", ms=3, lw=1.6, color=COL[l], label=f"{name}  (∝ n^{s:.2f})")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("population n (agents)"); ax.set_ylabel("messages + comparisons per query")
    ax.grid(which="major", lw=0.3, alpha=0.4)
    ax.set_title("(a) routing work per query: MIDIAN ~ log n, flat methods ~ n (bernoulli, b = 3, exact ledger)")
    ax.legend(ncol=2, frameon=False, loc="upper left", fontsize=6, rank="asc")
    return pd.DataFrame(rec)


def draw_J(ax, d):
    import energy
    fw = energy.table(); fw = fw[fw.index.str.startswith("fw_") & ~fw.index.str.contains("supervisor")]
    fwJ = fw.per_task_J                                  # supervisor call energy + the framework's messages / comparisons
    PJ = energy.probe_cost["specialist"] * 700
    T = np.logspace(2, 9, 200); rec = []
    ax.fill_between(T, fwJ.min(), fwJ.max(), color="#bbbbbb", alpha=0.5, lw=0, label=f"frameworks: one supervisor call per query ({fwJ.min():.0f}-{fwJ.max():.0f} J)")
    va = d[d.label == "midian_va"].set_index("n"); VB = va_build(d)
    for n, shade in ((1000, 0.55), (100000, 0.8), (10 ** 7, 1.0)):
        r = va.loc[n]; marg = r.messages_per_task * energy.J_MSG + r.comparisons_per_task * energy.J_CMP
        for b, ls in ((1, ":"), (3, "-"), (5, "--")):
            probes, src = VB[(n, b)]
            build = probes * PJ + r.build_messages * energy.J_MSG          # build messages: the b = 3 ledger (a rounding term)
            c = tuple(np.array(matplotlib.colors.to_rgb("#2ecc71")) * (1.35 - shade) ** 1.0)
            ax.plot(T, build / T + marg, ls, lw=1.4, color=c, label=f"MIDIAN-VA n = 10^{int(np.log10(n))}, b = {b}")
            for lo, hi in ((fwJ.min(), "cheapest"), (fwJ.max(), "costliest")):
                t = build / (lo - marg); ax.plot(t, lo, "x", color="black", ms=3, zorder=5)
                rec.append(dict(n=n, b=b, build_probes=probes, build_source=src, build_J=build, marginal_J=marg, vs=hi, fw_J=lo, break_even_queries=t))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("queries served T"); ax.set_ylabel("energy per query, J (build amortised)")
    ax.grid(which="major", lw=0.3, alpha=0.4)
    ax.set_title("(b) energy per query (*): one probing build, then ~0.01 J / query vs an LLM call per query")
    ax.legend(ncol=1, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=5.5, rank="asc")
    return pd.DataFrame(rec)


def fig_IJ(d):
    """I: both panels in ONE row (the one-row rule): asymptotic routing work and amortised energy."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 3.0), gridspec_kw={"width_ratios": [1, 1.05]})
    draw_I(a, d).to_csv(f"{OUT}/I_routing_work_vs_n.csv", index=False)
    draw_J(b, d).to_csv(f"{OUT}/I_energy_per_query.csv", index=False)
    fig.suptitle("I  efficiency: MIDIAN routes in O(log n) work and, after a linear one-off build, ~no energy per query "
                 "(* energy = scripts/energy.py estimate)", fontsize=7.5)
    save(fig, "I_efficiency")


if __name__ == "__main__":
    fig_IJ(costs())
