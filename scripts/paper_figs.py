"""Paper figures (vector PDF + 300-dpi PNG + CSV sidecar per figure) drawn at printed size for a 13.97 cm text width.
    python scripts/paper_figs.py [M1 M1_full M2 M3 F1_energy F1_shortlist]     -> figures/<name>.{pdf,png,csv}
Every value comes through rte.analyze.load (one name per arm); CIs are the 95% bootstrap over seeds used everywhere else
(paired over seeds where a difference is plotted). Nothing is interpolated: a series is dropped where its cell does not exist
and the gap is printed."""
import json, os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rte.analyze import load as _load, FLAT_ON
from extra_figs import COLOR, HAL, HALP, MAG14, ci as _ci, stat

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"); os.makedirs(OUT, exist_ok=True)
CACHE = os.environ.get("PAPER_CACHE")                                   # optional dir of <grid>.pkl from rte.analyze.load
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
                     "lines.markersize": 3, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "grid.linewidth": 0.4, "figure.constrained_layout.use": True, "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02})
cm = lambda x: x / 2.54
COLOR = {**COLOR, "knn_router": "#1abc9c", "mlp_router": "#795548", "fw_band": "#bbbbbb"}      # figures/COLOURS.md
NAME = {"oracle": "oracle", HALP: "seq. halving (peer)", HAL: "seq. halving (trusted)", "midian_va": "MIDIAN-VA", "midian_v": "MIDIAN-V", "midian_a": "MIDIAN-A",
        "midian": "MIDIAN", FLAT_ON: "flat probe argmax (online)", "knn_router": "RouterBench KNN router", "mlp_router": "RouterBench MLP router",
        "declared_argmax": "declared argmax", "warm_start_bandit": "warm-start bandit", "random": "random"}
SHORT = {**NAME, HALP: "halving (peer)", HAL: "halving (trusted)", FLAT_ON: "flat online", "knn_router": "KNN router", "mlp_router": "MLP router", "warm_start_bandit": "warm-start bandit"}
FWS = ["fw_autogen", "fw_camel_workforce", "fw_crewai", "fw_google_adk", "fw_langgraph", "fw_llamaindex", "fw_maf", "fw_magentic_one", "fw_openai_agents", "fw_smolagents"]
ABBR = {"fw_autogen": "autogen", "fw_camel_workforce": "camel", "fw_crewai": "crewai", "fw_google_adk": "adk", "fw_langgraph": "langgraph", "fw_llamaindex": "llama",
        "fw_maf": "maf", "fw_magentic_one": "magentic", "fw_openai_agents": "openai", "fw_smolagents": "smol"}
_mem = {}


def rows(*grids):
    out = []
    for g in grids:
        if g not in _mem:
            p = f"{CACHE}/{g}.pkl" if CACHE else None
            _mem[g] = pd.read_pickle(p) if p and os.path.exists(p) else _load([g])
        out.append(_mem[g])
    return pd.concat(out, ignore_index=True)


def units(df, labels, dist=None, beta=None, liar=None, n=None):
    """Success per (dist, beta, liar_select, seed) unit for the requested labels, on the cells where EVERY label exists."""
    q = df[df.label.isin(labels)]
    for k, v in (("dist", dist), ("beta", beta), ("liar_select", liar), ("n", n)):
        if v is not None: q = q[np.isclose(q[k], v) if k == "beta" else q[k] == v]
    return q.pivot_table(index=["dist", "beta", "liar_select", "seed"], columns="label", values="success")


def mean_ci(s):
    """Mean and 95% seed-bootstrap CI of a unit series indexed by (..., seed)."""
    s = s.dropna(); lo, hi = _ci(s); return float(s.mean()), float(lo), float(hi), int(s.index.get_level_values("seed").nunique())


def csv(name, recs):
    pd.DataFrame(recs).to_csv(f"{OUT}/{name}.csv", index=False)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png", dpi=300); plt.close(fig)
    w, h = fig.get_size_inches() * 2.54; print(f"[{name}] written ({w:.2f} x {h:.2f} cm)")


# ----------------------------------------------------------------------------------------------------- M1
M1_SERIES = ["oracle", HALP, "midian_va", "midian", FLAT_ON, "knn_router", "random"]
STYLE = {"oracle": dict(ls=":", color=COLOR["oracle"]), "random": dict(ls="--", color=COLOR["random"]), "midian_va": dict(lw=2.0)}


def m1_points(cartel):
    """(series, n, source) -> (mean, lo, hi, units, grid) on specialist cells; frameworks -> (min, max, grid). Missing cells are reported."""
    beta, liar = (0.5, "low_skill_first") if cartel else (0.0, "random")
    pts, band, miss = {}, {}, []
    def put(s, n, src, w, grid, allshape=None):
        if s not in w or w[s].dropna().empty: miss.append(f"{s} n={n} {src} ({grid})"); return
        m, lo, hi, k = mean_ci(w[s]); pts[(s, n, src)] = dict(mean=m, lo=lo, hi=hi, units=k, grid=grid, all_shape=allshape)
    # n = 100 / 1,000: MIDIAN-side arms and their KNN on identical units; random from the framework grids (same cells)
    for n, g_arms, g_fw, g_fwc in ((100, ["learned_n100"], "fw_live_n100", "fw_live_n100_lowskill"), (1000, ["variants_f1", "learned_f1", "live_f1_n1000"], "fw_live_n1000", "fw_live_n1000_lowskill")):
        d = rows(*g_arms); w = units(d, M1_SERIES, "specialist", beta, liar); wa = units(d, M1_SERIES, None, beta, liar)
        for s in M1_SERIES:
            if s == "random": continue
            put(s, n, "live", w, "+".join(g_arms), allshape=float(wa[s].mean()) if s in wa else None)
        f = rows(g_fwc if cartel else g_fw); wf = units(f, FWS + ["random"], "specialist", beta, liar)
        put("random", n, "live", wf, g_fwc if cartel else g_fw, allshape=float(units(f, ["random"], None, beta, liar)["random"].mean()))
        fm = wf[[c for c in FWS if c in wf]].mean(); band[n] = dict(min=float(fm.min()), max=float(fm.max()), grid=g_fwc if cartel else g_fw, units=int(len(wf)))
    # n = 10,000 live: learned_n10k (Q = 300, 3 seeds) for the arms; live_n10k_v2 (β = 0 only) for random and the frameworks
    d = rows("learned_n10k"); w = units(d, M1_SERIES, "specialist", beta, liar)
    for s in M1_SERIES:
        if s != "random": put(s, 10000, "live", w, "learned_n10k")
    v2 = rows("live_n10k_v2"); wv = units(v2, FWS + ["random"], "specialist", beta, liar)
    put("random", 10000, "live", wv, "live_n10k_v2")
    if len(wv) and any(c in wv for c in FWS):
        fm = wv[[c for c in FWS if c in wv]].mean(); band[10000] = dict(min=float(fm.min()), max=float(fm.max()), grid="live_n10k_v2", units=int(len(wv)))
    else: miss.append(f"ten frameworks n=10000 live (live_n10k_v2 has β = 0 random-liar cells only)")
    # simulation: scale_100k specialist cells
    d = rows("scale_100k")
    for n in (10000, 100000):
        w = units(d, M1_SERIES, "specialist", beta, liar, n); wa = units(d, M1_SERIES, None, beta, liar, n)
        for s in M1_SERIES:
            if s != "knn_router": put(s, n, "sim", w, "scale_100k", allshape=float(wa[s].mean()) if s in wa else None)
    return pts, band, miss


def draw_m1(ax, pts, band, legend):
    ns_live = sorted({n for (s, n, src) in pts if src == "live"}); ns_sim = sorted({n for (s, n, src) in pts if src == "sim"})
    if band:
        bn = sorted(band); ax.fill_between(bn, [band[n]["min"] for n in bn], [band[n]["max"] for n in bn], color=COLOR["fw_band"], alpha=.45, lw=0, label="ten frameworks", zorder=1)
    handles = {}
    for s in M1_SERIES:
        st = dict(color=COLOR[s], lw=1.0); st.update(STYLE.get(s, {}))
        live = [(n, pts[(s, n, "live")]) for n in ns_live if (s, n, "live") in pts]; sim = [(n, pts[(s, n, "sim")]) for n in ns_sim if (s, n, "sim") in pts]
        if live:
            handles[s] = ax.errorbar([n for n, _ in live], [p["mean"] for _, p in live], yerr=[[p["mean"] - p["lo"] for _, p in live], [p["hi"] - p["mean"] for _, p in live]],
                        marker="o", ms=3.2, capsize=1.5, elinewidth=0.6, label=SHORT[s], zorder=3, **st)
        if sim:
            st2 = {**st, "ls": "--" if s not in ("oracle", "random") else st["ls"]}
            h = ax.errorbar([n for n, _ in sim], [p["mean"] for _, p in sim], yerr=[[p["mean"] - p["lo"] for _, p in sim], [p["hi"] - p["mean"] for _, p in sim]],
                        marker="o", ms=3.2, mfc="white", capsize=1.5, elinewidth=0.6, label=SHORT[s] + " (sim.)", zorder=3, **st2)
            handles.setdefault(s, h)
    ax.set_xscale("log"); ax.set_xlabel("n (agents)"); ax.set_ylim(0.2, 0.9); ax.grid(alpha=.3, lw=0.4)
    if legend:
        hs = [handles[s] for s in M1_SERIES if s in handles] + ([h for h in ax.collections if h.get_label() == "ten frameworks"][:1] if band else [])
        ax.legend(hs, [h.get_label() for h in hs], loc="lower right", ncol=2, fontsize=6.0, frameon=False, handlelength=1.2, columnspacing=0.6, labelspacing=0.15, handletextpad=0.4, borderaxespad=0.15)


def M1():
    fig, axes = plt.subplots(1, 2, figsize=(cm(13.97), cm(4.6)), sharey=True); recs = []
    for ax, cartel, ttl in zip(axes, (False, True), ("honest, β = 0", "colluding low-skill cartel, β = 0.5")):
        pts, band, miss = m1_points(cartel); draw_m1(ax, pts, band, legend=cartel)          # legend in panel 2: its lower right is free (the framework band has no 10k cartel cell)
        for (s, n, src), p in pts.items(): recs.append(dict(panel=ttl, series=NAME[s], n=n, source=src, mean=p["mean"], ci_lo=p["lo"], ci_hi=p["hi"], units=p["units"], grid=p["grid"], all_shape_mean=p["all_shape"]))
        for n, b in band.items(): recs.append(dict(panel=ttl, series="ten frameworks (min)", n=n, source="live", mean=b["min"], grid=b["grid"], units=b["units"])); recs.append(dict(panel=ttl, series="ten frameworks (max)", n=n, source="live", mean=b["max"], grid=b["grid"], units=b["units"]))
        if miss: print(f"[M1 {ttl}] cells that do not exist (omitted): " + "; ".join(miss))
    axes[0].set_ylabel("success"); csv("M1_success_vs_n", recs); save(fig, "M1_success_vs_n")


M1F_SERIES = ["oracle", HAL, HALP, "declared_argmax", "warm_start_bandit", "midian_va", "midian_v", "midian", "mlp_router", FLAT_ON, "knn_router", "random"]


def M1_full():
    fig, axes = plt.subplots(2, 2, figsize=(cm(13.97), cm(9.5)), sharey="row"); recs = []
    for ax, cartel, ttl in zip(axes[0], (False, True), ("RTE, honest, β = 0", "RTE, colluding low-skill cartel, β = 0.5")):
        pts, band, miss = m1_points(cartel); draw_m1(ax, pts, band, legend=cartel)
        for (s, n, src), p in pts.items(): recs.append(dict(panel=ttl, series=NAME[s], n=n, source=src, mean=p["mean"], ci_lo=p["lo"], ci_hi=p["hi"], units=p["units"], grid=p["grid"]))
    re = rows("routereval_mmlu", "routereval_mmlu5k")
    for ax, cartel, ttl in zip(axes[1], (False, True), ("RouterEval real LLM pools, β = 0", "RouterEval real LLM pools, β = 0.5 low-skill cartel")):
        beta, liar = (0.5, "low_skill_first") if cartel else (0.0, "random")
        for s in M1F_SERIES:
            xs, ys, lo, hi = [], [], [], []
            for n in (10, 100, 1000, 5000):
                w = units(re, [s], None, beta, liar, n)
                if s not in w or w[s].dropna().empty: print(f"[M1_full {ttl}] no cell: {s} n={n}"); continue
                m, l, h, k = mean_ci(w[s]); xs.append(n); ys.append(m); lo.append(m - l); hi.append(h - m)
                recs.append(dict(panel=ttl, series=NAME[s], n=n, source="real pools", mean=m, ci_lo=l, ci_hi=h, units=k, grid="routereval_mmlu" if n < 5000 else "routereval_mmlu5k"))
            if xs:
                st = dict(color=COLOR[s], lw=1.0); st.update(STYLE.get(s, {})); ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", ms=3.2, capsize=1.5, elinewidth=0.6, label=SHORT[s], **st)
        ax.set_xscale("log"); ax.set_xlabel("m (candidate LLMs)"); ax.set_ylim(0.3, 1.0); ax.grid(alpha=.3, lw=0.4)
    h, l = axes[1][0].get_legend_handles_labels(); fig.legend(h, l, loc="outside lower center", ncol=6, fontsize=6.0, frameon=False, handlelength=1.4, columnspacing=0.8)
    axes[0][0].set_ylabel("success"); axes[1][0].set_ylabel("success"); csv("M1_success_vs_n_full", recs); save(fig, "M1_success_vs_n_full")


# ----------------------------------------------------------------------------------------------------- M2
M2_SERIES = ["oracle", HALP, "midian_va", "midian_v", "midian_a", "midian", FLAT_ON, "mlp_router", "knn_router", "declared_argmax"]
BAND = {"midian_va", "midian_v", HALP, "declared_argmax"}
REPORT_FREE = {FLAT_ON, "mlp_router", "knn_router", "declared_argmax", "oracle"}


def M2():
    d = rows("variants_f1", "learned_f1", "live_f1_n1000"); fig, axes = plt.subplots(1, 2, figsize=(cm(13.97), cm(4.4)), sharey=True); recs = []
    betas = [0.0, 0.1, 0.25, 0.5]
    for ax, liar, ttl in zip(axes, ("random", "low_skill_first"), ("random liars", "low-skill-first liars")):
        for s in M2_SERIES:
            ys, lo, hi = [], [], []
            for b in betas:
                w = units(d, [s], None, b, liar)
                if s not in w or w[s].dropna().empty: print(f"[M2 {ttl}] no cell: {s} β={b}"); ys.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
                m, l, h, k = mean_ci(w[s]); ys.append(m); lo.append(l); hi.append(h)
                recs.append(dict(panel=ttl, series=NAME[s], beta=b, mean=m, ci_lo=l, ci_hi=h, units=k, seeds=int(w[s].dropna().index.get_level_values("seed").nunique()), grid="variants_f1+learned_f1+live_f1_n1000"))
            x = np.arange(len(betas)); ls = ":" if s == "oracle" else ("--" if s in REPORT_FREE else "-")
            lw = 2.0 if s == "midian_va" else (0.8 if s in REPORT_FREE else 1.1)
            ax.plot(x, ys, ls=ls, lw=lw, color=COLOR[s], marker="o", ms=2.8 if s not in REPORT_FREE else 2.2, label=SHORT[s])
            if s in BAND: ax.fill_between(x, lo, hi, color=COLOR[s], alpha=.15, lw=0)
        ax.set_xticks(np.arange(len(betas))); ax.set_xticklabels([str(b) for b in betas]); ax.set_xlabel("β (liar fraction)"); ax.set_ylim(0.35, 0.75); ax.grid(alpha=.3, lw=0.4)
    axes[0].set_ylabel("success (n = 1,000)"); axes[0].legend(loc="lower left", ncol=2, fontsize=6.2, frameon=False, handlelength=2.2, columnspacing=0.7, labelspacing=0.3)
    csv("M2_beta_profile", recs); save(fig, "M2_beta_profile")


# ----------------------------------------------------------------------------------------------------- M3
def M3():
    d = rows("fw_live_n1000"); d = d[d.label.isin(["fw_magentic_one", MAG14])].copy(); d["strict"] = stat(d, "success_strict"); d["fallback"] = stat(d, "fallback_rate")
    vt = rows("fw_live_n1000_verified"); has14 = MAG14 in set(vt.label) or any("14B" in l for l in vt.label.unique())
    if not has14: print("[M3] fw_live_n1000_verified has no 14B Magentic-One arm: one panel")
    fig, ax = plt.subplots(figsize=(cm(6.8), cm(4.2))); recs = []; x = np.arange(3); wbar = 0.36
    for j, (lab, col, arm) in enumerate((("7B orchestrator", "#2980b9", "fw_magentic_one"), ("14B orchestrator", "#e67e22", MAG14))):
        for i, m in enumerate(("success", "strict", "fallback")):
            w = d[d.label == arm].pivot_table(index=["dist", "beta", "seed"], values=m)[m]; mu, lo, hi, k = mean_ci(w)
            ax.bar(x[i] + (j - .5) * wbar, mu, wbar, color=col, yerr=[[mu - lo], [hi - mu]], capsize=2, error_kw=dict(elinewidth=0.6), label=lab if i == 0 else None)
            recs.append(dict(metric=m, arm=lab, mean=mu, ci_lo=lo, ci_hi=hi, units=int(len(w)), seeds=k, grid="fw_live_n1000"))
    for i, m in enumerate(("success", "strict", "fallback")):
        a = d[d.label == "fw_magentic_one"].set_index(["dist", "beta", "seed"])[m]; b = d[d.label == MAG14].set_index(["dist", "beta", "seed"])[m]
        diff = (b - a).dropna(); mu, lo, hi, k = mean_ci(diff); cells = diff.groupby(level=["dist", "beta"]).mean()
        top = max(r["ci_hi"] for r in recs if r["metric"] == m and r["arm"] in ("7B orchestrator", "14B orchestrator"))
        ax.text(x[i], top + 0.02, f"{mu:+.3f}\n[{lo:+.3f}, {hi:+.3f}]", ha="center", va="bottom", fontsize=5.8)
        recs.append(dict(metric=m, arm="14B − 7B (paired)", mean=mu, ci_lo=lo, ci_hi=hi, units=int(len(diff)), seeds=k, cells_14B_lower=int((cells < 0).sum()), cells=int(len(cells)), grid="fw_live_n1000"))
    ax.set_xticks(x); ax.set_xticklabels(["lenient\nsuccess", "strict\nsuccess", "fallback\nrate"]); ax.set_ylim(0, 1.02); ax.set_ylabel("fraction of tasks"); ax.grid(axis="y", alpha=.3, lw=0.4)
    ax.legend(loc="upper left", fontsize=6.2, frameon=False, handlelength=1.2); csv("M3_orchestrator_7b_vs_14b", recs); save(fig, "M3_orchestrator_7b_vs_14b")


# ----------------------------------------------------------------------------------------------------- F1
def F1_energy():
    import energy
    t = pd.read_pickle(f"{CACHE}/energy_table.pkl") if CACHE and os.path.exists(f"{CACHE}/energy_table.pkl") else energy.table()
    T = np.logspace(2, 5, 300); fig = plt.figure(figsize=(cm(3.6), cm(3.2)), layout="none"); ax = fig.add_axes([0.27, 0.23, 0.67, 0.75]); recs = []
    fws = [m for m in t.index if m.startswith("fw_") and "14B" not in m]; light = plt.cm.Blues(np.linspace(0.35, 0.8, len(fws)))
    for m, c in zip(fws, light):
        r = t.loc[m]; ax.plot(T, r.build_J + T * r.per_task_J, color=c, lw=0.7); recs.append(dict(series=m, build_J=float(r.build_J), per_task_J=float(r.per_task_J), grid="fw_live_n1000 (ledger) × scripts/energy.py cost model"))
    r = t.loc["midian"]; ax.plot(T, r.build_J + T * r.per_task_J, color=COLOR["midian"], lw=1.4); recs.append(dict(series="midian", build_J=float(r.build_J), per_task_J=float(r.per_task_J), grid="live_f1_n1000+variants_f1 (ledger) × scripts/energy.py cost model"))
    for b, lab, xy, ha in (("fw_magentic_one", "Magentic", (-2, -8), "right"), ("fw_crewai", "CrewAI", (0, 5), "center"), ("fw_autogen", "AutoGen", (2, -8), "left")):
        xj = energy.crossing(t, "midian", b, "build_J", "per_task_J"); xg = energy.crossing(t, "midian", b); y = r.build_J + xj * r.per_task_J
        ax.plot([xj], [y], "kx", ms=4, mew=0.9); ax.annotate(f"{lab}\n{xj:,.0f}", (xj, y), textcoords="offset points", xytext=xy, ha=ha, va="top" if xy[1] < 0 else "bottom", fontsize=5.5, linespacing=0.9)
        recs.append(dict(series=f"crossing midian vs {b}", tasks_joules=float(xj), tasks_gpu_s=float(xg), label_uses=round(xj)))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([1e2, 1e3, 1e4, 1e5]); ax.set_xlim(1e2, 1e5); ax.set_xlabel("tasks routed", fontsize=7); ax.set_ylabel("cumulative J", fontsize=7)
    ax.tick_params(labelsize=6, pad=1); ax.minorticks_off(); ax.grid(alpha=.25, lw=0.3, which="major"); ax.xaxis.labelpad = 1; ax.yaxis.labelpad = 1; csv("F1_energy_crossings", recs); save(fig, "F1_energy_crossings")


def F1_shortlist():
    d = rows("fw_live_n1000", "fw_live_n1000_verified"); d = d[d.declared_source == "self_described"] if "declared_source" in d else d
    cols = [f + s for f in FWS for s in ("", "[r=10,retrieval=midian]", "[r=5,retrieval=midian]")] + ["midian_v", "oracle"]
    w = d[d.label.isin(cols)].pivot_table(index=["dist", "beta", "seed"], columns="label", values="success")[cols].dropna()
    fig = plt.figure(figsize=(cm(3.9), cm(3.2)), layout="none"); ax = fig.add_axes([0.25, 0.31, 0.73, 0.67]); x = np.arange(len(FWS)); recs = []
    for off, suf, lab, c in ((-.28, "", "own", "#2980b9"), (0, "[r=10,retrieval=midian]", "+V r=10", "#e67e22"), (.28, "[r=5,retrieval=midian]", "+V r=5", "#f1c40f")):
        vals = [mean_ci(w[f + suf]) for f in FWS]
        ax.bar(x + off, [v[0] for v in vals], .28, color=c, label=lab, yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], capsize=0.8, error_kw=dict(elinewidth=0.4))
        for f, v in zip(FWS, vals): recs.append(dict(framework=f, arm=lab, mean=v[0], ci_lo=v[1], ci_hi=v[2], units=int(len(w)), seeds=v[3], grid="fw_live_n1000 / fw_live_n1000_verified"))
    mv, orc = mean_ci(w["midian_v"]), mean_ci(w["oracle"])
    ax.axhline(mv[0], color=COLOR["midian_v"], ls="--", lw=0.8, label=f"V {mv[0]:.3f}"); ax.axhline(orc[0], color=COLOR["oracle"], ls=":", lw=0.8, label=f"oracle {orc[0]:.3f}")
    recs += [dict(framework="", arm="midian_v", mean=mv[0], ci_lo=mv[1], ci_hi=mv[2], units=int(len(w)), seeds=mv[3], grid="fw_live_n1000"), dict(framework="", arm="oracle", mean=orc[0], ci_lo=orc[1], ci_hi=orc[2], units=int(len(w)), seeds=orc[3], grid="fw_live_n1000")]
    ax.set_xticks(x); ax.set_xticklabels([ABBR[f] for f in FWS], rotation=45, ha="right", fontsize=6.5); ax.set_ylim(0.40, 0.85); ax.set_ylabel("success", fontsize=7); ax.tick_params(axis="y", labelsize=6, pad=1); ax.tick_params(axis="x", pad=1)
    ax.grid(axis="y", alpha=.25, lw=0.3); ax.legend(loc="upper left", ncol=2, fontsize=5.5, frameon=True, framealpha=0.7, edgecolor="none", handlelength=1.0, labelspacing=0.15, columnspacing=0.5, borderpad=0.15, handletextpad=0.4); csv("F1_shortlist_lift_n1000", recs); save(fig, "F1_shortlist_lift_n1000")


FIGS = {f.__name__: f for f in (M1, M1_full, M2, M3, F1_energy, F1_shortlist)}
if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGS): FIGS[name]()
