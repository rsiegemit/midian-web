"""SAMPLE efficiency figures in the condensed A / B style (one panel, one row each).
    python scripts/efficiency_figs.py   -> figures/condensed_sample/{C_routing_work_vs_n,D_energy_per_query}.{png,pdf,csv}
Two figures, one panel each (they replace the old C paired-gap and D heatmap figures):
  C   routing work per query vs n (10^2..10^7, calibrated bernoulli, b = 3, honest; exact ledger counts: messages +
     comparisons per routed task), one line per arm; legend: growth() of the log-log slope over n >= 10^3 (log n, n^s).
  D   energy per routed query vs queries served: (build J) / T + marginal J per query. MIDIAN at n = 10^3, 10^5, 10^7
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
from rte.methods import keys
from condensed_figs import ARMS, BANDIT, LEARNED, NOT_RUNNABLE, OUT, POOLS, save

R = os.environ.get("RTE_DATA", "/scratch/rte") + "/results"
COST = f"{OUT}/cost_by_n.csv"
COLS = ["n", "b", "beta", "method", "params", "messages_per_task", "comparisons_per_task", "build_probes", "build_messages"]
FIXED = ["midian", "midian_wo_defenses", "flat_probe_argmax_online", "declared_argmax"]      # C draws exactly A / B's arms (ARMS)
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
    return "constant" if abs(s) < 0.02 else "∝ log n" if s < 0.3 else f"∝ n^{s:.2f}"


def draw_I(ax, d):
    """Exactly A / B's arms, labels and colours. A pool ("best learned router", "best bandit") picks a different member per
    cell and seed, so it is drawn as the band between its cheapest and costliest runnable member at each n."""
    d = d[d.b == 3].assign(work=lambda x: x.messages_per_task + x.comparisons_per_task); rec = []
    for k, name, c in ARMS:
        if k == "random":                                    # routes blindly: no routing work, nothing to draw on a log axis
            ax.plot([], [], "o-", ms=3, lw=1.6, color=c, label=f"{name}  (0: no routing work)"); continue
        if k in POOLS:
            q = d[d.label.isin(POOLS[k]) & ~pd.Series([l in NOT_RUNNABLE("bernoulli", n) for l, n in zip(d.label, d.n)], index=d.index)]
            flat = [l for l, g in q[q.n >= 100].groupby("label") if growth(slope(g)) == "constant"]   # constant-cost members (flat_nsw) are left out of the band
            q = q[(q.n >= 100) & ~q.label.isin(flat)].groupby("n").work.agg(["min", "max"]).reset_index()
            lo, hi = slope(q.rename(columns={"min": "work"})), slope(q.rename(columns={"max": "work"}))
            ax.fill_between(q.n, q["min"], q["max"], color=c, alpha=0.3, lw=0)
            for e in ("min", "max"): ax.plot(q.n, q[e], "-", lw=1.0, color=c)
            a, z = sorted((lo, hi))
            span = growth(lo) if growth(lo) == growth(hi) else f"∝ n^{a:.2f}–{z:.2f}" if a >= 0.3 else f"{growth(lo)} to {growth(hi)} across its pool"
            ax.plot([], [], "o-", ms=3, lw=1.6, color=c, label=f"{name}  ({span})")
            rec += [dict(arm=name, n=int(n), work=w, work_max=m, slope=lo, slope_max=hi) for n, w, m in zip(q.n, q["min"], q["max"])]
            continue
        q = d[(d.label == k) & (d.n >= 100)].sort_values("n")
        s = slope(q); rec += [dict(arm=name, n=int(n), work=w, slope=s) for n, w in zip(q.n, q.work)]
        ax.plot(q.n, q.work, "o-", ms=3, lw=1.6, color=c, label=f"{name}  ({growth(s)})")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("population n (agents)"); ax.set_ylabel("messages + comparisons per query")
    ax.grid(which="major", lw=0.3, alpha=0.4)
    ax.set_title("C  routing work per query, the arms of A / B: MIDIAN grows like log n (calibrated bernoulli, b = 3, exact ledger)")
    ax.legend(ncol=2, frameon=False, loc="upper left", fontsize=6, rank="asc")
    return pd.DataFrame(rec)


def draw_J(ax, d):
    import energy
    fw = energy.table(); fw = fw[fw.index.str.startswith("fw_") & ~fw.index.str.contains("supervisor")]
    fwJ = fw.per_task_J                                  # supervisor call energy + the framework's messages / comparisons
    PJ = energy.probe_cost["specialist"] * 700
    T = np.logspace(2, 9, 200); rec = []
    ce = fw.sup_call_equiv                               # supervisor call-equivalents: latency ratio to one-call AutoGen
    ax.fill_between(T, fwJ.min(), fwJ.max(), color="#bbbbbb", alpha=0.5, lw=0,
                    label=f"frameworks: {ce.min():.0f}-{ce.max():.1f} supervisor-call equivalents per query ({fwJ.min():.0f}-{fwJ.max():.0f} J)")
    va = d[(d.label == "midian") & (d.b == 3)].set_index("n"); VB = va_build(d)
    for n, shade in ((1000, 0.55), (100000, 0.8), (10 ** 7, 1.0)):
        r = va.loc[n]; marg = r.messages_per_task * energy.J_MSG + r.comparisons_per_task * energy.J_CMP
        for b, ls in ((1, ":"), (3, "-"), (5, "--")):
            probes, src = VB[(n, b)]
            build = probes * PJ + r.build_messages * energy.J_MSG          # build messages: the b = 3 ledger (a rounding term)
            c = tuple(np.array(matplotlib.colors.to_rgb("#2ecc71")) * (1.35 - shade) ** 1.0)
            ax.plot(T, build / T + marg, ls, lw=1.4, color=c, label=f"MIDIAN n = 10^{int(np.log10(n))}, b = {b}")
            for lo, hi in ((fwJ.min(), "cheapest"), (fwJ.max(), "costliest")):
                t = build / (lo - marg); ax.plot(t, lo, "x", color="black", ms=3, zorder=5)
                rec.append(dict(n=n, b=b, build_probes=probes, build_source=src, build_J=build, marginal_J=marg, vs=hi, fw_J=lo, break_even_queries=t))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("queries served T"); ax.set_ylabel("estimated energy per query, J (build amortised)")
    ax.grid(which="major", lw=0.3, alpha=0.4)
    ax.set_title("D  estimated energy per query: MIDIAN pays one probing build, then ~0.01 J / query; frameworks pay supervisor LLM calls every query", pad=38)
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=5.5, rank="asc")
    return pd.DataFrame(rec)


def fig(draw, name, d):
    """One panel, one row, the A / B size."""
    f, ax = plt.subplots(figsize=(7.2, 2.8))
    draw(ax, d).to_csv(f"{OUT}/{name}.csv", index=False); save(f, name)


if __name__ == "__main__":
    d = costs(); fig(draw_I, "C_routing_work_vs_n", d); fig(draw_J, "D_energy_per_query", d)
