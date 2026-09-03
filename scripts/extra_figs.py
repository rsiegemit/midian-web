"""v2 figures H1-H9 + appendix (RESULTS_rte_v2.md).  python scripts/extra_figs.py [H1 H3 ...]
Every figure renders from whatever rows exist and skips (with a note) what is missing. Labels come from rte.analyze
(one name per arm). No wall-clock anywhere except H5's supervisor-latency panel (framework calls are never memoised)."""
import ast, json, os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rte.analyze import RTE_DATA, load as _load, FLAT, FLAT_ON

O = f"{RTE_DATA}/results/extra_figs"; os.makedirs(O, exist_ok=True)
RED, ORG, YEL, BLU, GRN, PUR, GRY, DRK = "#c0392b", "#e67e22", "#f1c40f", "#3498db", "#27ae60", "#8e44ad", "#999999", "#2c3e50"
HAL, HALP = "sequential_halving", "sequential_halving_peer"
HAL_RB, HAL_ST = "sequential_halving[churn_mode=rebuild,peer_reported=True]", "sequential_halving[churn_mode=stale,peer_reported=True]"
MAG14 = "fw_magentic_one[supervisor=Qwen/Qwen2.5-14B-Instruct]"
COLOR = {"oracle": GRY, "midian": RED, "midian_v": ORG, "midian_v_r5": YEL, "midian_sh": "#d35400", "midian_a": "#7b241c", "midian_sha": "#e74c3c",
         FLAT: "#7f8c8d", FLAT_ON: BLU, HAL: DRK, HALP: PUR, HAL_RB: PUR, HAL_ST: "#bb8fce", "warm_start_bandit": GRN, "linucb_honest": "#16a085",
         "declared_argmax": "#5d6d7e", "llm_supervisor": "#34495e", "fw_autogen": "#2980b9", "fw_magentic_one": "#1f618d", MAG14: "#5dade2", "random": "#ccc"}
CLASS_COLOR = {"framework": "#2980b9", "midian": RED, "ceiling": GRY, "floor": "#ccc", "declared": "#5d6d7e", "verified_central": GRN, "verified_decentral": PUR}
STYLE = dict(marker="o", ms=4)
fw = lambda m: m.replace("fw_", "").replace("_", " ")
col = lambda m: COLOR.get(m, None)


def rows(*grids):
    """Rows of the grids that exist (labelled by rte.analyze); empty frame if none."""
    have = [g for g in grids if os.path.isdir(f"{RTE_DATA}/results/{g}/rows.d") and os.listdir(f"{RTE_DATA}/results/{g}/rows.d")]
    return _load(have) if have else pd.DataFrame(columns=["label", "success"])


def piv(df, index=("dist", "beta", "seed"), val="success"):
    return df.pivot_table(index=list(index), columns="label", values=val) if len(df) else pd.DataFrame()


def stat(df, key):
    """One number per row from the method_stats JSON column (NaN when absent)."""
    return df.get("method_stats", pd.Series(index=df.index, dtype=object)).map(lambda s: (json.loads(s) if isinstance(s, str) and s.startswith("{") else {}).get(key, np.nan))


def ci(x, B=2000):
    rng = np.random.default_rng(0); x = np.asarray(x, float); x = x[np.isfinite(x)]
    return np.percentile([rng.choice(x, len(x)).mean() for _ in range(B)], [2.5, 97.5]) if len(x) else (np.nan, np.nan)


def line(ax, s, label, **kw):
    ax.errorbar(s.mean().index, s.mean().values, yerr=1.96 * s.sem().fillna(0).values, label=label, color=col(label) if "color" not in kw else kw.pop("color"), **STYLE, **kw)


def need(w, labels, fig):
    miss = [l for l in labels if l not in w]
    if miss: print(f"[{fig}] waiting on data: {', '.join(miss)}")
    return [l for l in labels if l in w]


def save(fig, name):
    fig.savefig(f"{O}/{name}.png", dpi=300, bbox_inches="tight"); plt.close(fig); print(f"[{name}] written")


def selfdesc(df):
    return df[df.declared_source == "self_described"] if "declared_source" in df else df


# ------------------------------------------------------------------ headline
def H1():
    """By shape, n=1000, self-described: oracle / midian_v / midian / midian_sha / flat online bars; frameworks as a min-max band
    with fallback % annotated. midian_sha and flat online come from variants_f1 on the same (dist, beta, seed) cells."""
    fwr = selfdesc(rows("fw_live_n1000")); var = selfdesc(rows("variants_f1"))
    var = var[var.liar_select == "random"] if "liar_select" in var else var
    w = pd.concat([piv(fwr), piv(var)[[c for c in piv(var).columns if c in ("midian_sha", FLAT_ON)]]] if len(var) else [piv(fwr)], axis=1)
    bars = need(w, ["oracle", "midian_v", "midian", "midian_sha", FLAT_ON], "H1"); fws = sorted(c for c in w.columns if c.startswith("fw_") and c != MAG14)
    core = w[bars + fws].dropna() if bars and fws else pd.DataFrame(); shapes = [d for d in ["specialist", "heavy_tail", "bimodal"] if len(core) and d in core.index.get_level_values("dist")]
    if not shapes: return print("[H1] no complete cells yet")
    fb = fwr[fwr.label.isin(fws)].assign(fbr=stat(fwr, "fallback_rate")).groupby("label").fbr.mean()
    fig, axes = plt.subplots(1, len(shapes), figsize=(6 * len(shapes), 6), sharey=True)
    for ax, d in zip(np.atleast_1d(axes), shapes):
        g = core.xs(d, level="dist"); x = np.arange(len(bars))
        ax.bar(x, [g[b].mean() for b in bars], color=[col(b) for b in bars], yerr=[[g[b].mean() - ci(g[b])[0] for b in bars], [ci(g[b])[1] - g[b].mean() for b in bars]], capsize=3)
        fm = g[fws].mean(); ax.axhspan(fm.min(), fm.max(), color="#2980b9", alpha=.18, label=f"10 frameworks, min-max ({fm.min():.2f}-{fm.max():.2f})")
        ax.axhline(fm.mean(), color="#2980b9", ls="--", lw=1)
        for i, f in enumerate(fm.sort_values().index):
            ax.text(len(bars) - .4, fm[f], f"{fw(f)} {100 * fb.get(f, np.nan):.0f}%", fontsize=6, va="center", ha="left", color="#1f618d")
        ax.set_xticks(x); ax.set_xticklabels(bars, rotation=25, ha="right"); ax.set_title(d.replace("_", " ")); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=7, loc="lower left")
    sp = core.xs("specialist", level="dist") if "specialist" in shapes else None
    ttl = f"specialist: frameworks {sp[fws].mean().mean():.2f} vs MIDIAN {sp['midian'].mean():.2f}" if sp is not None else ""
    fig.suptitle(f"H1  n=1000, self-described channel, {len(core)} paired cells; {ttl} (framework labels: fallback %)"); axes[0].set_ylabel("success")
    save(fig, "H1_headline_by_shape")


def H2():
    """Legibility: x = Spearman(D_self_described, S) per shape (scripts/legibility.py -> legibility.json) and x = declared_argmax
    success; y = framework success - MIDIAN success; one point per (shape, framework, n)."""
    p = f"{O}/legibility.json"
    if not os.path.exists(p): print("[H2] waiting on data: legibility.json (run scripts/legibility.py on a compute node)"); return
    rho = pd.DataFrame(json.load(open(p))).groupby(["n", "dist"]).spearman.mean()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for n in (100, 1000):
        w = piv(selfdesc(rows(f"fw_live_n{n}")))
        if "midian" not in w or "declared_argmax" not in w: print(f"[H2] waiting on data: fw_live_n{n}"); continue
        g = w.groupby(level="dist").mean(); fws = [c for c in w.columns if c.startswith("fw_") and c != MAG14]
        for d in g.index:
            for f in fws:
                y = g.loc[d, f] - g.loc[d, "midian"]
                if (n, d) in rho: axes[0].scatter(rho[(n, d)], y, color=CLASS_COLOR["framework"], marker="o" if n == 1000 else "^", alpha=.7)
                axes[1].scatter(g.loc[d, "declared_argmax"], y, color=CLASS_COLOR["framework"], marker="o" if n == 1000 else "^", alpha=.7)
            if (n, d) in rho: axes[0].annotate(f"{d} n={n}", (rho[(n, d)], (g.loc[d, fws] - g.loc[d, 'midian']).mean()), fontsize=7)
            axes[1].annotate(f"{d} n={n}", (g.loc[d, "declared_argmax"], (g.loc[d, fws] - g.loc[d, 'midian']).mean()), fontsize=7)
    for ax, xl in zip(axes, ["Spearman(self-described D, true S), per shape", "declared_argmax success on the self-described channel"]):
        ax.axhline(0, color=GRY, ls=":"); ax.set_xlabel(xl); ax.set_ylabel("framework success - MIDIAN success"); ax.grid(alpha=.3)
    fig.suptitle("H2  legibility of skill from self-descriptions vs the frameworks' gap to MIDIAN (o n=1000, ^ n=100)")
    save(fig, "H2_legibility")


def H3():
    """x = success at beta=0, y = success at beta=0.5 collude low-skill-first; self-described; every method a point by class."""
    df = selfdesc(rows("live_f1_n1000", "variants_f1", "fw_live_n1000"))
    if not len(df): return print("[H3] waiting on data")
    lo = df[np.isclose(df.beta, 0)]; hi = df[np.isclose(df.beta, 0.5) & (df.collude == True) & (df.liar_select == "low_skill_first")]   # noqa: E712
    fwhi = df[np.isclose(df.beta, 0.5) & df.label.str.startswith("fw_")]                   # fw grids: random liars only
    x, y = lo.groupby("label").success.mean(), pd.concat([hi, fwhi]).groupby("label").success.mean()
    both = x.index.intersection(y.index); grp = df.drop_duplicates("label").set_index("label").group
    fig, ax = plt.subplots(figsize=(8.5, 8))
    for l in both:
        c = CLASS_COLOR.get(grp[l], "#000"); ax.scatter(x[l], y[l], color=c, s=70 if grp[l] == "midian" else 35, marker="*" if l == "oracle" else "o", zorder=3)
        if grp[l] in ("midian", "ceiling") or l in (HALP, HAL, FLAT_ON): ax.annotate(l, (x[l], y[l]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.plot([0.3, 0.9], [0.3, 0.9], color=GRY, ls=":", label="no loss from lying"); ax.set_xlim(0.3, 0.9); ax.set_ylim(0.3, 0.9)
    for g_, c in CLASS_COLOR.items(): ax.scatter([], [], color=c, label=g_)
    ax.set_xlabel("success at β=0"); ax.set_ylabel("success at β=0.5, colluding, low-skill-first liars (frameworks: random liars)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("H3  consistency vs robustness, n=1000, self-described channel"); save(fig, "H3_consistency_robustness")


def H4():
    """Cost-quality Pareto: x = build + Q*per-task for messages / comparisons / LLM calls at Q in {1e2..1e5}; y = success (n=1000,
    self-described); break-even Q of midian_v against the one-call framework marked."""
    df = selfdesc(rows("fw_live_n1000", "variants_f1")); df = df[df.liar_select == "random"] if "liar_select" in df else df
    if not len(df): return print("[H4] waiting on data")
    arms = need(df.groupby("label").success.mean(), ["midian", "midian_v", FLAT, HALP, "fw_autogen", "fw_magentic_one"], "H4")
    g = df[df.label.isin(arms)].groupby("label")[["success", "build_messages", "build_comparisons", "build_probes", "messages_per_task", "comparisons_per_task", "hops_per_task"]].mean()
    g["build_calls"], g["calls_per_task"] = g.build_probes, [1.0 if l.startswith("fw_") else 0.0 for l in g.index]       # frameworks: 1 supervisor call/task (Magentic-One: several; not measured -> lower bound)
    Qs = [1e2, 1e3, 1e4, 1e5]; fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (b, per, ttl) in zip(axes, [("build_messages", "messages_per_task", "messages"), ("build_comparisons", "comparisons_per_task", "comparisons"), ("build_calls", "calls_per_task", "LLM calls (probes are calls; frameworks >= 1/task)")]):
        ax.set_title(ttl)
        for l in g.index:
            xs = [g.loc[l, b] + Q * g.loc[l, per] for Q in Qs]; ax.plot(xs, [g.loc[l, "success"]] * 4, marker="o", color=col(l), label=l)
            for Q, x_ in zip(Qs, xs): ax.annotate(f"{Q:.0e}", (x_, g.loc[l, "success"]), fontsize=5, xytext=(0, 4), textcoords="offset points", ha="center")
        if {"midian_v", "fw_autogen"} <= set(g.index):
            d = g.loc["fw_autogen", per] - g.loc["midian_v", per]; be = (g.loc["midian_v", b] - g.loc["fw_autogen", b]) / d if d > 0 else np.inf
            ax.axvline(g.loc["midian_v", b] + be * g.loc["midian_v", per], color=ORG, ls="--", lw=1) if np.isfinite(be) and be > 0 else None
            ax.set_title(f"{ttl}\nbreak-even midian_v vs one-call framework: Q = {be:,.0f}" if np.isfinite(be) and be > 0 else f"{ttl}\nmidian_v cheaper from the first task")
        ax.set_xscale("symlog"); ax.set_xlabel("build + Q x per-task (points at Q = 1e2, 1e3, 1e4, 1e5)"); ax.grid(alpha=.3, which="both")
    axes[0].set_ylabel("success (n=1000, self-described)"); axes[0].legend(fontsize=8); fig.suptitle("H4  cost-quality with break-even"); save(fig, "H4_cost_quality_breakeven")


def H5():
    """Cost scaling on bernoulli_scale (K=16; b=3 to 1e5, b=1 above) + supervisor latency from the fw rows only."""
    d = rows("bernoulli_scale"); d = d[d.beta.isin([0.0, 0.25])] if len(d) else d
    keep = {"midian": "-", "midian_v": "-", FLAT: "-", "declared_argmax": "--", "cnp_self_bid": ":", HAL: "-"}
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, (c, ttl) in zip(axes[:3], [("comparisons_per_task", "comparisons per task"), ("messages_per_task", "messages per task"), ("build_probes", "build probes (b=1 above 1e5)")]):
        for l, ls in keep.items():
            s = d[d.label == l].groupby("n")[c].mean() if len(d) else pd.Series(dtype=float); s = s[s > 0]
            if len(s): ax.plot(s.index, s.values, marker="o", ls=ls, color=col(l), label=l, lw=2.5 if l.startswith("midian") else 1.3)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("n (agents)"); ax.set_title(ttl + " (0 omitted)"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    ax = axes[3]; f = rows("fw_live_n100", "fw_live_n1000"); f = f[f.label.str.startswith("fw_")] if len(f) else f
    if len(f) and "wall_clock_per_task" in f:
        q = f.groupby(["label", "n"]).wall_clock_per_task.median().unstack("n"); q.index = [fw(i) for i in q.index]
        q.plot.barh(ax=ax, color=["#aed6f1", "#2980b9"]); ax.set_xlabel("median supervisor seconds per task"); ax.set_title("framework supervisor latency\n(never memoised; shared-fleet load)"); ax.grid(axis="x", alpha=.3)
    fig.suptitle("H5  cost vs n (calibrated bernoulli, 10^2-10^7): MIDIAN per task = r*ceil(log_r n) (~n^0.11), midian_v 1, flat/declared/CNP = n"); save(fig, "H5_cost_scaling")


def H6():
    """MIDIAN vs halving by beta x liar selection, self-described, with the Phase-1 variants; second row = replay twin."""
    live = selfdesc(rows("live_f1_n1000", "variants_f1")); rep = rows("replay_mirror_live_f1_n1000")
    arms = ["oracle", HAL, HALP, "midian_v", "midian", "midian_sh", "midian_a", "midian_sha", FLAT_ON]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey="row")
    for r_, (df, name) in enumerate([(live, "live LLM population, self-described channel"), (rep, "RouterBench replay twin")]):
        w = piv(df, ("dist", "beta", "liar_select", "seed")) if len(df) else pd.DataFrame()
        for ax, ls in zip(axes[r_], ["random", "low_skill_first"]):
            for l in need(w, arms, "H6"):
                sub = w[l].xs(ls, level="liar_select"); line(ax, sub.groupby(level="beta"), l, lw=2.5 if l == "midian" else 1.2)
            ax.set_title(f"{name}, liars = {ls}"); ax.set_xlabel("β (liar fraction)"); ax.grid(alpha=.3)
        axes[r_][0].set_ylabel("success (n=1000)")
    axes[0][1].legend(fontsize=7); fig.suptitle("H6  MIDIAN and its variants vs sequential halving"); save(fig, "H6_midian_vs_halving")


def H7():
    """Shortlist lift: each framework with its own selection vs with midian_v's cohort (r=10, r=5); midian_v alone as reference."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True); done = False
    for ax, n in zip(axes, (100, 1000)):
        df = selfdesc(rows(f"fw_live_n{n}", f"fw_live_n{n}_verified")); w = piv(df)
        fws = sorted(c for c in w.columns if c.startswith("fw_") and "[" not in c)
        cols = [f + s for f in fws for s in ("", "[r=10,retrieval=midian]", "[r=5,retrieval=midian]")] + ["midian_v", "oracle"]
        if not fws or need(w, cols, "H7") != cols: continue
        c = w[cols].dropna(); x = np.arange(len(fws)); done = True
        for off, suf, lab, cc in ((-.27, "", "own selection (TF-IDF top-10)", "#2980b9"), (0, "[r=10,retrieval=midian]", "+ midian_v cohort r=10", ORG), (.27, "[r=5,retrieval=midian]", "+ midian_v cohort r=5", YEL)):
            ax.bar(x + off, [c[f + suf].mean() for f in fws], .27, label=lab, color=cc)
        ax.axhline(c["midian_v"].mean(), color=RED, ls="--", label=f"midian_v alone ({c['midian_v'].mean():.2f})"); ax.axhline(c["oracle"].mean(), color=GRY, ls=":", label="oracle")
        ax.set_xticks(x); ax.set_xticklabels([fw(f) for f in fws], rotation=35, ha="right"); ax.set_ylim(0.4, 0.85); ax.grid(axis="y", alpha=.3); ax.set_title(f"n={n} ({len(c)} paired cells)")
    if done: axes[0].set_ylabel("success"); axes[1].legend(fontsize=8); fig.suptitle("H7  frameworks with MIDIAN's verified shortlist"); save(fig, "H7_shortlist_lift")
    else: plt.close(fig)


def H8():
    """Budget sweep split by declaration channel (budget_sweep: b=1,3,10 programmatic; budget_b10_shapes adds b=10 on both channels)."""
    df = rows("budget_sweep", "budget_b10_shapes")
    if not len(df): return print("[H8] waiting on data")
    arms = ["oracle", HAL, HALP, "midian_v", "midian", FLAT, FLAT_ON, "warm_start_bandit", "declared_argmax", "linucb_honest", "fw_langgraph", "fw_autogen"]
    chans = sorted(df.declared_source.unique()); fig, axes = plt.subplots(1, len(chans), figsize=(7 * len(chans), 5.5), sharey=True)
    for ax, ch in zip(np.atleast_1d(axes), chans):
        w = piv(df[df.declared_source == ch], ("dist", "seed", "b"))
        for l in need(w, arms, "H8"): line(ax, w[l].groupby(level="b"), l, lw=2.5 if l == "midian" else 1.2)
        ax.set_xscale("log"); ax.set_xticks([1, 3, 10]); ax.set_xticklabels(["1", "3", "10"]); ax.set_xlabel("probe budget b per (agent, family)")
        ax.set_title(f"{ch}" + (" = upper bound (S + N(0, 0.05))" if ch == "programmatic" else " = the live channel")); ax.grid(alpha=.3); ax.legend(fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("success (n=1000, β=0.25)"); fig.suptitle("H8  success vs build budget, by declaration channel"); save(fig, "H8_budget_by_channel")


def H9():
    """Churn: success per 100 tasks and cumulative probes across churn events; one panel per churn fraction."""
    df = rows("churn_n1000")
    if not len(df) or "churn" not in df: return print("[H9] waiting on data: churn_n1000")
    df = df.assign(frac=df.churn.map(lambda c: (c if isinstance(c, dict) else ast.literal_eval(str(c))).get("frac")), blocks=df.success_by_block.map(lambda b: b if isinstance(b, list) else ast.literal_eval(str(b))))
    arms = ["oracle", "midian", "midian_sh", "midian_a", "midian_v", HAL_RB, HAL_ST, FLAT_ON, "warm_start_bandit", "linucb_honest", "fw_langgraph", "fw_autogen"]
    fracs = sorted(df.frac.dropna().unique()); fig, axes = plt.subplots(2, len(fracs), figsize=(7 * len(fracs), 9), squeeze=False)
    for j, fr in enumerate(fracs):
        d = df[df.frac == fr]
        for l in [a for a in arms if a in set(d.label)]:
            g = d[d.label == l]; curve = np.mean(np.stack(g.blocks.values), axis=0); x = np.arange(len(curve)) * 100 + 50
            axes[0][j].plot(x, curve, marker="o", ms=3, color=col(l), label=l, lw=2.5 if l == "midian" else 1.2)
            rep = g.get("repair_probes_per_event", pd.Series(0, index=g.index)).fillna(0).mean(); ev = np.floor(np.arange(len(curve)) * 100 / 200)
            axes[1][j].plot(x, g.build_probes.mean() + ev * rep, color=col(l), label=l, lw=2.5 if l == "midian" else 1.2)
        for k in range(200, 1000, 200): axes[0][j].axvline(k, color=GRY, ls=":", lw=.8); axes[1][j].axvline(k, color=GRY, ls=":", lw=.8)
        axes[0][j].set_title(f"churn {int(fr * 100)}% of agents every 200 tasks"); axes[0][j].set_ylabel("success per 100 tasks"); axes[1][j].set_ylabel("cumulative probes (build + repairs)")
        axes[1][j].set_xlabel("task index"); axes[1][j].set_yscale("log"); [a.grid(alpha=.3) for a in axes[:, j]]
    axes[0][-1].legend(fontsize=7); fig.suptitle("H9  churn_n1000 (n=1000, self-described, specialist + heavy_tail, β ∈ {0, 0.25})"); save(fig, "H9_churn")


# ------------------------------------------------------------------ appendix
def A_internals():
    df = rows("internals_v2")
    if not len(df): return print("[A_internals] waiting on data: internals_v2")
    df = df.assign(r=df.params.map(lambda p: json.loads(p).get("r")), delta=df.params.map(lambda p: json.loads(p).get("delta")))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, m in zip(axes, ["midian", "midian_a"]):
        d = df[df.method == m]
        for (cl, ls), g in d.groupby(["collude", "liar_select"]):
            t = g.pivot_table(index="r", columns="delta", values="success")
            if t.shape[1] == 2: ax.plot(t.index, t.iloc[:, 1] - t.iloc[:, 0], marker="o", label=f"collude={cl}, liars={ls}")
        ax.axhline(0, color=GRY, ls=":"); ax.set_xscale("log"); ax.set_xticks([5, 10, 20]); ax.set_xticklabels(["5", "10", "20"]); ax.set_xlabel("cohort size r")
        ax.set_title(f"{m}: success(δ=1/3) − success(δ=0) at β=0.5"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    axes[0].set_ylabel("effect of trimming"); fig.suptitle("Appendix  internals_v2: does trimming help once reports are audited?"); save(fig, "A_internals_v2")


def A_learning():
    runs = [json.load(open(f)) for f in sorted(__import__("glob").glob(f"{RTE_DATA}/scratch/curve_*.json"))]
    if not runs: return print("[A_learning] waiting on data: scratch/curve_*.json")
    B = 100; fig, ax = plt.subplots(figsize=(9, 5)); orc = np.array([r["oracle"] for r in runs], float).reshape(len(runs), -1, B).mean(axis=(0, 2))
    for m, lab in [('midian{"cached":true,"verify":true}', "midian_v"), ("midian{}", "midian"), ('midian{"online":false}', "midian[online=False]"), ('flat_probe_argmax{"online":true}', FLAT_ON), ("warm_start_bandit{}", "warm_start_bandit"), ("llm_supervisor{}", "llm_supervisor")]:
        arr = np.array([r[m] for r in runs], float).reshape(len(runs), -1, B).mean(axis=(0, 2)) - orc
        ax.plot(np.arange(len(arr)) * B + B / 2, arr, marker="o", label=lab, color=col(lab), ls="--" if "False" in lab else "-", lw=2.5 if lab == "midian" else 1.2)
    ax.axhline(0, color=GRY, ls=":", label="oracle"); ax.set_xlabel("task index (blocks of 100)"); ax.set_ylabel(f"success minus oracle ({len(runs)} live cells, n=1000, β=0.25)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("Appendix  learning over the stream"); save(fig, "A_learning_curve")


def A_ksens():
    df = rows("fw_k_sensitivity")
    if not len(df): return print("[A_ksens] waiting on data")
    t = df.pivot_table(index="label", columns="beta", values="success"); fig, ax = plt.subplots(figsize=(8, 4.5)); t.plot.barh(ax=ax); ax.set_xlabel("success (specialist, n=1000, self-described)")
    ax.set_title("Appendix  framework shortlist size k ∈ {5, 10, 20}"); ax.grid(axis="x", alpha=.3); save(fig, "A_k_sensitivity")


def A_replay():
    w = piv(rows("replay_mirror_live_f1_n1000"), ("dist", "beta", "liar_select", "declared_source", "seed"))
    if not len(w): return print("[A_replay] waiting on data")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for l in need(w, ["oracle", HAL, HALP, "declared_argmax", "midian_v", "midian", FLAT_ON, FLAT, "random"], "A_replay"): line(ax, w[l].groupby(level="beta"), l, lw=2.5 if l == "midian" else 1.2)
    ax.set_xlabel("β"); ax.set_ylabel("success (RouterBench replay, n=1000, K=64, 10 seeds)"); ax.grid(alpha=.3); ax.legend(fontsize=7); ax.set_title("Appendix  replay twin of the F1 sweep"); save(fig, "A_replay_mirror")


def A_fallback():
    df = pd.concat([rows(f"fw_live_n{n}").assign(n=n) for n in (100, 1000)]); df = df[df.label.str.startswith("fw_")] if len(df) else df
    if not len(df): return print("[A_fallback] waiting on data")
    t = pd.DataFrame({"fallback_rate": stat(df, "fallback_rate"), "success": df.success, "success_strict": stat(df, "success_strict"), "label": df.label, "n": df.n}).groupby(["label", "n"]).mean()
    fig, ax = plt.subplots(figsize=(10, 6)); ax.axis("off"); tbl = ax.table(cellText=np.round(t.values, 3), rowLabels=[f"{fw(a)} n={b}" for a, b in t.index], colLabels=t.columns, loc="center"); tbl.set_fontsize(7); tbl.scale(1, 1.2)
    ax.set_title("Appendix  framework fallback rate, lenient success, strict success (failure = 0)"); save(fig, "A_fallback_table")


def A_bandits():
    w = piv(selfdesc(rows("live_f1_n1000")), ("dist", "beta", "liar_select", "declared_source", "seed"))
    if not len(w): return print("[A_bandits] waiting on data")
    fig, ax = plt.subplots(figsize=(8, 5))
    for l in need(w, ["midian", FLAT_ON, "ucb_per_family", "thompson_per_family", "warm_start_bandit", "linucb_honest"], "A_bandits"): line(ax, w[l].groupby(level="beta"), l, lw=2.5 if l == "midian" else 1.2)
    ax.set_xlabel("β"); ax.set_ylabel("success (n=1000, self-described)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("Appendix  UCB / Thompson: 16k arms, Q=1,000, under-explored by construction"); save(fig, "A_bandits")


FIGS = {f.__name__: f for f in (H1, H2, H3, H4, H5, H6, H7, H8, H9, A_internals, A_learning, A_ksens, A_replay, A_fallback, A_bandits)}
if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGS):
        try: FIGS[name]()
        except Exception as e: print(f"[{name}] FAILED: {type(e).__name__}: {e}")
