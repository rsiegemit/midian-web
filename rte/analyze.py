"""Aggregate rows -> paired statistics -> figures F1-F7 -> pre-registered target check (SPEC §7-8).

    python -m rte.analyze --grid smoke [--grids replay_scale,live_f1_n1000] [--out DIR]

Paired by seed: a cell is one point of the CELL axes and its seeds share a task stream, so
MIDIAN-vs-rival deltas are taken seed by seed, never as a difference of two means. A delta is
WITHIN_FLOOR when it does not exceed MIDIAN's own seed envelope (max-min of its per-seed success in
that cell). Each method's `group` (framework | midian | declared | verified_central |
verified_decentral | floor | ceiling) comes from its declared `needs` and the fw_ prefix.
"""
import argparse, json, os, sys, warnings
import numpy as np, pandas as pd
from . import run as _run

RTE_DATA, consolidate = _run.RTE_DATA, _run.consolidate
CELL_COLS = tuple(getattr(_run, "CELL_FIELDS", None) or getattr(_run, "CELL", None) or (
    "backend n K dist beta liar_select collude declared_source lie_mode demand b Q".split()))
REF, FLOOR, B_BOOT = "midian", "WITHIN_FLOOR", 2000
MARK = {"llm": "o", "replay": "s", "bernoulli": "^"}       # marker shape = backend, on every figure
COST = ["comparisons_per_task", "hops_per_task", "messages_per_task", "total_comm_per_task",
        "wall_clock_per_task"]
BUILD = ["build_probes", "build_reports", "build_messages", "build_total_comm"]
log = lambda m: print(m, file=sys.stderr, flush=True)
cells = lambda df: [c for c in CELL_COLS if c in df.columns]
fmt = lambda d: ", ".join(f"{k} {v:+.3f}" for k, v in sorted(d.items()))

# ------------------------------------------------------------------ load + group
def group_of(name):
    """framework | midian | ceiling | floor | declared | verified_decentral | verified_central."""
    if name.startswith("fw_"): return "framework"
    if name.startswith(REF): return "midian"
    if name in ("oracle", "random"): return "ceiling" if name == "oracle" else "floor"
    try:
        from .methods import load_method
        needs = frozenset(load_method(name).needs)
    except Exception: return "unknown"                     # optional dep or LLM-only file
    if not needs & {"probe", "reports"}: return "declared"
    return "verified_decentral" if needs & {"reports", "bus"} else "verified_central"
def load(grids):
    """Rows of every grid, plus label, group, and the total-communication columns when absent."""
    frames = []
    for g in grids:
        d = f"{RTE_DATA}/results/{g}"
        if os.path.isdir(f"{d}/rows.d"): consolidate(d)    # refresh rows.csv from the per-row files
        if os.path.exists(f"{d}/rows.csv"): frames.append(pd.read_csv(f"{d}/rows.csv"))
        else: log(f"[analyze] no rows for grid {g!r} at {d}")
    if not frames: raise SystemExit(f"no rows found for grids {grids}")
    df = pd.concat(frames, ignore_index=True)
    df["params"] = df.params.fillna("{}")
    short = lambda p: ",".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in sorted(json.loads(p).items()))
    df["label"] = [m if p == "{}" else f"{m}[{short(p)}]" for m, p in zip(df.method, df.params)]
    df["group"] = df.method.map(group_of)
    per = ["probes_per_task", "reports_per_task", "messages_per_task", "tasks_per_task"]
    for c in per + BUILD[:3]:                              # pre-rewrite CSVs lack some counters
        if c not in df: df[c] = 0.0
    if "total_comm_per_task" not in df: df["total_comm_per_task"] = df[per].sum(axis=1)
    if "build_total_comm" not in df: df["build_total_comm"] = df[BUILD[:3]].sum(axis=1)
    return df

# ------------------------------------------------------------------ statistics
def boot(x, B=B_BOOT, seed=12345):
    """(mean, lo, hi) percentile bootstrap of the mean."""
    x = np.asarray([v for v in np.asarray(x, float) if np.isfinite(v)])
    if x.size < 2: return (float(x[0]),) * 3 if x.size else (np.nan,) * 3
    m = x[np.random.default_rng(seed).integers(0, x.size, (B, x.size))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
def sign_test(d):
    """(n_positive, n_nonzero, two-sided p) of the paired sign test."""
    nz = np.asarray([v for v in np.asarray(d, float) if np.isfinite(v) and v != 0])
    if nz.size == 0: return 0, 0, 1.0
    from scipy.stats import binomtest
    return int((nz > 0).sum()), nz.size, float(binomtest(int((nz > 0).sum()), nz.size, 0.5).pvalue)
def aggregate(df, metric="success"):
    """Per (cell, group, label): seeds, mean, 95% bootstrap CI, seed envelope."""
    keys = cells(df) + ["group", "label"]
    return pd.DataFrame([
        {**dict(zip(keys, k)), "metric": metric, "n_seeds": int(np.isfinite(v).sum()),
         **dict(zip(("mean", "ci_lo", "ci_hi"), boot(v))),
         "envelope": float(v.max() - v.min()) if v.size > 1 else 0.0}
        for k, v in ((k, g[metric].to_numpy(float)) for k, g in df.groupby(keys, dropna=False))])
def paired(df, metric="success", ref=REF):
    """Paired-by-seed delta of `ref` against every other label, per cell."""
    keys, rows = cells(df), []
    for k, g in df.groupby(keys, dropna=False):
        piv = g.pivot_table(index="seed", columns="label", values=metric, aggfunc="mean")
        if ref not in piv: continue
        env = float(piv[ref].max() - piv[ref].min()) if piv[ref].notna().sum() > 1 else 0.0
        for lab in piv.columns.drop(ref):
            both = piv[[ref, lab]].dropna()
            if both.empty: continue
            d = (both[ref] - both[lab]).to_numpy(float)
            m, lo, hi = boot(d); pos, nz, p = sign_test(d)
            rows.append({**dict(zip(keys, k)), "ref": ref, "rival": lab,
                         "group": g.loc[g.label == lab, "group"].iloc[0], "n_pairs": d.size,
                         "delta_mean": m, "delta_lo": lo, "delta_hi": hi, "sign_pos": pos,
                         "sign_nonzero": nz, "sign_p": p, "seed_envelope": env,
                         "verdict": FLOOR if abs(m) <= env else ("midian_better" if m > 0 else "rival_better")})
    return pd.DataFrame(rows)
def fit(n, y, seeds, B=500, seed=7):
    """log10(y) = a + k log10(n), exponent k bootstrapped over seeds. None when unfittable."""
    n, y, seeds = map(np.asarray, (n, y, seeds))
    ok = np.isfinite(n) & np.isfinite(y.astype(float)) & (n > 0) & (y > 0)
    n, y, seeds = n[ok].astype(float), y[ok].astype(float), seeds[ok]
    if np.unique(n).size < 2: return None
    k, a = np.polyfit(np.log10(n), np.log10(y), 1)
    uniq, rng, ks = np.unique(seeds), np.random.default_rng(seed), []
    for _ in range(B if uniq.size > 1 else 0):
        m = np.concatenate([np.flatnonzero(seeds == s) for s in rng.choice(uniq, uniq.size, True)])
        if np.unique(n[m]).size >= 2: ks.append(np.polyfit(np.log10(n[m]), np.log10(y[m]), 1)[0])
    lo, hi = np.percentile(ks, [2.5, 97.5]) if ks else (np.nan, np.nan)
    return float(k), float(lo), float(hi), float(a)
def exponents(df, metrics):
    """Fitted cost/n exponents per (metric, label), plus rows flagging identically-zero costs."""
    out = []
    for metric in [m for m in metrics if m in df.columns]:
        for lab, g in df.groupby("label"):
            zero = not (g[metric] > 0).any()
            f = (0.0, 0.0, 0.0) if zero else (fit(g.n, g[metric], g.seed) or (None,))
            if f[0] is not None: out.append(
                {"metric": metric, "label": lab, "exponent": f[0], "exp_lo": f[1], "exp_hi": f[2],
                 "note": "identically zero at every n" if zero else f"{g.n.nunique()} values of n"})
    return pd.DataFrame(out)

# ------------------------------------------------------------------ plotting (one generic helper)
def _plt():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt
def _write(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight"); _plt().close(fig); log(f"  wrote {path}"); return path
def panel_plot(df, x, y, hue, panel, title, path, logx=False, logy=False, marker_by="backend",
               ylabel=None, err=True):
    """The only line-figure function: y vs x, one line per `hue`, one axes per `panel` value, error
    bars = 95% bootstrap over seeds, marker shape = `marker_by`. Returns the path, or None."""
    df = df.dropna(subset=[x, y])
    if df.empty or df[x].nunique() < 2: return None
    plt = _plt()
    vals = sorted(df[panel].dropna().unique()) if panel else [None]
    fig, axes = plt.subplots(1, len(vals), figsize=(4.8 * len(vals) + 2, 4.2), sharey=True, squeeze=False)
    cmap = plt.get_cmap("tab20")
    col = {h: cmap(i % 20) for i, h in enumerate(sorted(df[hue].astype(str).unique()))}
    for ax, pv in zip(axes[0], vals):
        for h, g in (df if pv is None else df[df[panel] == pv]).groupby(df[hue].astype(str)):
            a = g.groupby(x)[y].apply(list).reset_index().sort_values(x)
            st = [boot(v) for v in a[y]]
            ax.errorbar(a[x], [s[0] for s in st], color=col[h], marker="", label=h, capsize=2,
                        lw=2.4 if h.startswith(REF) else 1.1,
                        yerr=[[s[0] - s[1] for s in st], [s[2] - s[0] for s in st]] if err else None)
            for be, gg in g.groupby(marker_by):             # marker shape says which backend
                b = gg.groupby(x)[y].mean().reset_index().sort_values(x)
                ax.plot(b[x], b[y], MARK.get(be, "d"), color=col[h], ms=5)
        if logx: ax.set_xscale("log")
        if logy: ax.set_yscale("log")
        ax.set_xlabel(x); ax.grid(alpha=.3, which="both"); ax.set_title("" if pv is None else str(pv))
    axes[0][0].set_ylabel(ylabel or y)
    axes[0][-1].legend(fontsize=6, ncol=2, loc="center left", bbox_to_anchor=(1.02, .5))
    axes[0][0].legend(handles=[plt.Line2D([], [], marker=m, ls="", color="k", label=b)
                               for b, m in MARK.items() if b in set(df[marker_by])],
                      fontsize=7, title=marker_by, loc="best")
    fig.suptitle(title)
    return _write(fig, path)
def heat(piv, path, title):
    """F5 is the one figure that is not a line plot: method x distribution, annotated."""
    if piv.empty or piv.shape[1] < 2: return None
    plt = _plt()
    piv = piv.reindex(piv.mean(axis=1).sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(1.7 * piv.shape[1] + 3.5, .34 * piv.shape[0] + 2))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=7)
    for (i, j), v in np.ndenumerate(piv.to_numpy()):
        if np.isfinite(v): ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                                   color="w" if v < .6 else "k")
    fig.colorbar(im, ax=ax, label="success"); ax.set_title(title)
    return _write(fig, path)
def at_n(df, want=1000):
    ns = sorted(df.n.dropna().unique())
    return None if not ns else (want if want in ns else max(ns))
def figures(df, fd):
    """F1-F7 (+F3b), each one call to `panel_plot`/`heat`. Returns {name: path} and the exponents."""
    os.makedirs(fd, exist_ok=True)
    P, rivals = {}, df[df.method != "oracle"]
    n0, b25 = at_n(df), df[np.isclose(df.beta, 0.25)]
    n25 = at_n(b25)
    def long(ms):                          # cost panels; zero costs are dropped (log) but still fitted
        m = rivals.melt(id_vars=["n", "backend", "seed", "label"], var_name="metric",
                        value_name="value", value_vars=[c for c in ms if c in rivals.columns])
        return m[m.value > 0]
    P["F1"] = panel_plot(df[df.n == n0], "beta", "success", "label", "dist",
                         f"F1  success vs liar fraction (n={n0})", f"{fd}/F1_success_vs_beta.png")
    P["F2"] = panel_plot(b25, "n", "success", "label", "dist", "F2  success vs n at beta=0.25",
                         f"{fd}/F2_success_vs_n.png", logx=True)
    P["F3"] = panel_plot(long(COST), "n", "value", "label", "metric", "F3  per-task cost vs n (log-log)",
                         f"{fd}/F3_cost_vs_n.png", logx=True, logy=True, ylabel="per task", err=False)
    P["F3b"] = panel_plot(long(BUILD), "n", "value", "label", "metric", "F3b  build cost vs n (log-log)",
                          f"{fd}/F3b_build_cost_vs_n.png", logx=True, logy=True, ylabel="build", err=False)
    P["F4"] = panel_plot(b25[b25.n == n25], "b", "success", "label", "dist",
                         "F4  success vs build budget b", f"{fd}/F4_success_vs_budget.png", logx=True)
    P["F5"] = heat(b25[b25.n == n25].pivot_table(index="label", columns="dist", values="success"),
                   f"{fd}/F5_method_x_dist.png", "F5  method x distribution, beta=0.25")
    P["F6"] = panel_plot(rivals[rivals.n == n0], "beta", "misroute_to_liar", "label", "dist",
                         f"F6  misroute to liar vs beta (n={n0})", f"{fd}/F6_misroute_vs_beta.png",
                         ylabel="fraction routed to a liar")
    mid = df[df.method == REF].copy()                       # F7: MIDIAN internals
    par = mid.params.map(json.loads)
    mid["r"] = [p.get("r", np.nan) for p in par]
    mid["trim"] = [f"delta={p.get('delta', np.nan):.2g}, collude={c}" for p, c in zip(par, mid.collude)]
    P["F7"] = panel_plot(mid.dropna(subset=["r"]), "r", "success", "trim", "beta",
                         "F7  MIDIAN internals: trimming vs cohort size", f"{fd}/F7_midian_internals.png")
    return {k: v for k, v in P.items() if v}, exponents(rivals, COST + BUILD)

# ------------------------------------------------------------------ pre-registered targets
def pair(df, a, b, metric="success"):
    """Methods `a` and `b` paired on (cell, seed); empty when either is missing from `df`."""
    if df.empty: return pd.DataFrame()
    piv = df.pivot_table(index=cells(df) + ["seed"], columns="method", values=metric)
    return piv[[a, b]].dropna() if {a, b} <= set(piv.columns) else pd.DataFrame()
def _t1(df, fits):
    """declared/framework lose >=0.25 from beta 0->0.5, probe-only move <=0.03"""
    b0, b5 = df[np.isclose(df.beta, 0)], df[np.isclose(df.beta, 0.5)]
    if b0.empty or b5.empty: return None, "needs beta=0 and beta=0.5"
    drop = (b0.groupby("method").success.mean() - b5.groupby("method").success.mean()).dropna()
    grp = df.drop_duplicates("method").set_index("method").group
    dec = {m: v for m, v in drop.items() if grp.get(m) in ("declared", "framework")}
    prb = {m: v for m, v in drop.items() if grp.get(m) in ("verified_central", "verified_decentral", "midian")}
    return (bool(dec) and bool(prb) and all(v >= 0.25 for v in dec.values())
            and all(abs(v) <= 0.03 for v in prb.values()),
            f"declared+framework: {fmt(dec)} | probe-based: {fmt(prb)}")
def _t2(df, fits):
    """MIDIAN == flat_probe_argmax within 0.02 at beta=0; comparisons ~ r log_r n vs flat ~ n"""
    parts, ok = [], []
    d = pair(df[np.isclose(df.beta, 0)], REF, "flat_probe_argmax")
    if not d.empty:
        x = float((d[REF] - d.flat_probe_argmax).mean()); ok.append(abs(x) <= 0.02)
        parts.append(f"paired mean delta {x:+.4f} over {len(d)} (cell, seed) pairs. NOTE: MIDIAN's default is "
                     f"online=True and flat_probe_argmax is frozen after build, so this compares an online method "
                     f"with an offline one -- rerun with midian params {{online: false}} to test the max-tree claim.")
    f = fits[fits.metric == "comparisons_per_task"].set_index("label") if not fits.empty else pd.DataFrame()
    if {REF, "flat_probe_argmax"} <= set(f.index):
        m, s = f.loc[REF], f.loc["flat_probe_argmax"]
        ok.append(bool(m.exponent < 0.4 and s.exponent > 0.8))
        parts.append(f"comparisons: MIDIAN k={m.exponent:.2f} [{m.exp_lo:.2f},{m.exp_hi:.2f}]; "
                     f"flat_probe_argmax k={s.exponent:.2f} [{s.exp_lo:.2f},{s.exp_hi:.2f}]")
    return (all(ok) if ok else None), "; ".join(parts) or "needs beta=0 rows and >=2 values of n"
def _t3(df, fits):
    """trimming helps only where beta*r exceeds the trim"""
    mid = df[df.method == REF].copy()
    mid["delta"] = [json.loads(p).get("delta", np.nan) for p in mid.params]
    c = mid[(mid.collude == True) & mid.delta.notna()]      # noqa: E712
    if c.delta.nunique() < 2: return None, "needs MIDIAN at two deltas with collude=True (grid midian_internals)"
    tab = c.pivot_table(index="beta", columns="delta", values="success", aggfunc="mean")
    sep = tab[tab.columns.max()] - tab[tab.columns.min()]
    below, above = [b for b in sep.index if b <= 0.3], [b for b in sep.index if b > 0.3]
    return (bool(above) and all(abs(sep[b]) <= 0.02 for b in below) and any(sep[b] > 0.02 for b in above),
            f"success(delta={tab.columns.max():.2g}) - success(delta={tab.columns.min():.2g}) by beta: "
            + ", ".join(f"{b}: {sep[b]:+.3f}" for b in sep.index))
def _t4(df, fits):
    """verify_on_claim: <=0.03 from oracle at beta<=0.1, loses >=0.10 by beta=0.5"""
    v = df[df.method == "verify_on_claim"]
    if v.empty: return None, "verify_on_claim rows missing"
    at_lo, at_hi = v.beta <= 0.1, np.isclose(v.beta, 0.5)
    gap = float((v.oracle_success - v.success)[at_lo].mean()) if at_lo.any() else np.nan
    lo = float(v[at_lo].success.mean()) if at_lo.any() else np.nan
    hi = float(v[at_hi].success.mean()) if at_hi.any() else np.nan
    return (None if not np.isfinite(gap - hi) else bool(gap <= 0.03 and lo - hi >= 0.10),
            f"oracle gap at beta<=0.1 = {gap:.3f}; success {lo:.3f} -> {hi:.3f} (loss {lo-hi:+.3f})")
def _t5(df, fits):
    """sequential_halving ~= flat_probe_argmax; bandits learn past MIDIAN at b=1"""
    parts, ok, miss = [], [], []
    d = pair(df, "sequential_halving", "flat_probe_argmax")
    if d.empty: miss.append("sequential_halving/flat_probe_argmax rows")
    else:
        x = float((d.sequential_halving - d.flat_probe_argmax).mean()); ok.append(abs(x) <= 0.03)
        parts.append(f"sequential_halving - flat_probe_argmax = {x:+.3f}")
    for bandit in ("ucb_per_family", "thompson_per_family"):
        d = pair(df[df.b == 1], bandit, REF, "success_late")
        if d.empty: continue
        x = float((d[bandit] - d[REF]).mean()); ok.append(x >= 0)
        parts.append(f"b=1 success_late {bandit} - MIDIAN = {x:+.3f}")
    if not any("b=1" in p for p in parts): miss.append("a b=1 block with bandit success_late")
    return ((all(ok) if ok else None),
            "; ".join(parts) + (f"  [not evaluated: {', '.join(miss)}]" if miss else ""))
def _t6(df, fits):
    """argmax-vs-floor gap largest under heavy_tail, smallest under iid_uniform"""
    finders = df[df.group.isin(["midian", "verified_central", "declared"])]
    floors = df[df.method.isin(["random", "route_to_k_majority"])]
    if finders.empty or floors.empty or df.dist.nunique() < 2: return None, "needs >=2 distributions"
    g = (finders.groupby("dist").success.mean() - floors.groupby("dist").success.mean()).dropna()
    return (bool(g.idxmax() == "heavy_tail" and g.idxmin() == "iid_uniform")
            if {"heavy_tail", "iid_uniform"} <= set(g.index) else None,
            "gap by dist: " + ", ".join(f"{k} {v:.3f}" for k, v in g.sort_values(ascending=False).items()))
def targets(df, fits):
    """The six expectations of TARGETS_rte.md. PASS / MISS / NO DATA with the numbers. Never a fix."""
    out = []
    for i, fn in enumerate((_t1, _t2, _t3, _t4, _t5, _t6), 1):
        try: ok, detail = fn(df, fits)
        except (KeyError, ValueError, IndexError) as e: ok, detail = None, f"no data ({type(e).__name__}: {e})"
        out.append({"target": i, "verdict": "NO DATA" if ok is None else ("PASS" if ok else "MISS"),
                    "name": fn.__doc__, "detail": detail})
    return out

# ------------------------------------------------------------------ report
def md(df, f="{:.4f}"):
    if df is None or df.empty: return "_(no rows)_\n"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]): d[c] = d[c].map(lambda v: "" if pd.isna(v) else f.format(v))
    head = "| " + " | ".join(map(str, d.columns)) + " |\n| " + " | ".join("---" for _ in d.columns) + " |\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)) + "\n"
def by_method(df, extra=()):
    """Mean +- bootstrap CI of success per method, grouped by method class, costs beside it."""
    g = df.groupby(["group", "label"])
    cols = [c for c in ("success_late", "regret", "misroute_to_liar", *extra) if c in df.columns]
    return (g.success.apply(lambda s: pd.Series(dict(zip(("success", "lo", "hi"), boot(s))))).unstack()
            .join(g[cols].mean()).sort_values("success", ascending=False).reset_index())
def roll_up(cmp_):
    """Per rival, across cells: mean paired delta, its CI, sign-test p and how often MIDIAN wins."""
    n = lambda tag: ("verdict", lambda s: int((s == tag).sum()))
    return cmp_.groupby(["group", "rival"]).agg(
        cells=("delta_mean", "size"), delta_mean=("delta_mean", "mean"), delta_lo=("delta_lo", "mean"),
        delta_hi=("delta_hi", "mean"), delta_min=("delta_mean", "min"), delta_max=("delta_mean", "max"),
        midian_better=n("midian_better"), rival_better=n("rival_better"), within_floor=n(FLOOR),
        min_sign_p=("sign_p", "min")).sort_values("delta_mean").reset_index()
def summary(out, grids, df, agg, cmp_, fits, tgts, figs):
    sec = lambda t, table, *prose: ["", f"## {t}", "", *prose, *([""] if prose else []), table]
    L = [f"# RTE results -- {', '.join(grids)}", "",
         f"{len(df)} rows | {df.label.nunique()} methods | "
         f"{df.groupby(cells(df), dropna=False).ngroups} cells | seeds {sorted(map(int, df.seed.unique()))}",
         f"Backends: {', '.join(sorted(df.backend.astype(str).unique()))}.", "",
         "Intervals are 95% percentile bootstrap over seeds; MIDIAN-vs-rival deltas are paired by",
         f"seed. `{FLOOR}` means the delta does not exceed MIDIAN's own seed envelope in that cell.",
         *sec("Pre-registered targets (TARGETS_rte.md)",
              md(pd.DataFrame(tgts)[["target", "verdict", "name", "detail"]])),
         *sec("HEADLINE: frameworks vs MIDIAN, by method class (SPEC 6A)",
              md(by_method(df, COST + BUILD), "{:.4g}"),
              "Frameworks are the systems practitioners deploy, run through their own libraries; every",
              "one reads names and self-descriptions only. The other groups are SPEC 6 mechanism controls."),
         *sec("Success by method x beta",
              md(df.pivot_table(index=["group", "label"], columns="beta", values="success").reset_index()))]
    if not cmp_.empty:
        L += sec("MIDIAN vs every rival, paired by seed", md(roll_up(cmp_)))
        L += ["", "### Per-cell detail", "", md(cmp_.drop(columns="ref"))]
    if not fits.empty:
        L += sec("Cost exponents (cost ~ n^k)", md(fits.sort_values(["metric", "exponent"]), "{:.3f}"),
                 "MIDIAN's per-task messages are 2*ceil(log_r n): logarithmic, so its fitted exponent sits",
                 "near 0. Flat table lookups send 0 messages per task and CNP 2n: exponents 0 and 1.")
    if figs: L += ["", "## Figures", ""] + [f"- {k}: `{os.path.relpath(v, out)}`" for k, v in sorted(figs.items())]
    L += sec("Per-cell aggregate (success)", md(agg))
    path, tmp = f"{out}/summary.md", f"{out}/summary.md.tmp{os.getpid()}"
    open(tmp, "w").write("\n".join(L) + "\n"); os.replace(tmp, path)
    return path
def main(argv=None):
    p = argparse.ArgumentParser("rte.analyze")
    p.add_argument("--grid", required=True); p.add_argument("--grids")
    p.add_argument("--metric", default="success"); p.add_argument("--out")
    a = p.parse_args(argv)
    grids = [a.grid] + [g.strip() for g in (a.grids or "").split(",") if g.strip() and g.strip() != a.grid]
    df = load(grids)
    out = a.out or f"{RTE_DATA}/results/{a.grid}"
    os.makedirs(out, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        figs, fits = figures(df, f"{out}/figs")
    agg, cmp_ = aggregate(df, a.metric), paired(df[df.method != "oracle"], a.metric)
    tgts = targets(df, fits)
    log("\n=== pre-registered targets ===")
    for t in tgts: log(f"  [{t['verdict']:>7}] {t['target']}. {t['name']}\n            {t['detail']}")
    agg.to_csv(f"{out}/aggregate.csv", index=False)
    cmp_.to_csv(f"{out}/paired_vs_midian.csv", index=False)
    if not fits.empty: fits.to_csv(f"{out}/cost_exponents.csv", index=False)
    log(f"\n[analyze] wrote {summary(out, grids, df, agg, cmp_, fits, tgts, figs)}")

if __name__ == "__main__":
    main()
