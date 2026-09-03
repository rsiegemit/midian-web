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
FLAT, FLAT_ON = "flat_probe_argmax_frozen", "flat_probe_argmax_online"
ALIAS = {"flat_probe_argmax": FLAT, "flat_probe_argmax[online=True]": FLAT_ON,        # one name per arm everywhere
         "midian[cached=True,verify=True]": "midian_v", "midian[cached=True,r=5,verify=True]": "midian_v_r5",
         "sequential_halving[peer_reported=True]": "sequential_halving_peer", "midian[stratify=True]": "midian_stratified",
         "sequential_halving[churn_mode=rebuild,peer_reported=True]": "sequential_halving_peer_rebuild",
         "sequential_halving[churn_mode=stale,peer_reported=True]": "sequential_halving_peer_stale"}
STATS = ("success_strict", "fallback_rate")               # framework accountings carried inside method_stats (0.2)
MARK = {"llm": "o", "replay": "s", "bernoulli": "^"}       # marker shape = backend, on every figure
COST = ["comparisons_per_task", "hops_per_task", "messages_per_task", "total_comm_per_task"]   # no wall-clock: memo-mixed
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
def reads_declared(name):
    """True for every method whose `needs` include the declared channel (frameworks read self-descriptions)."""
    if name.startswith("fw_"): return True
    try:
        from .methods import load_method
        return "declared" in load_method(name).needs
    except Exception: return False
def load(grids):
    """Rows of every grid, plus label, group, and the total-communication columns when absent."""
    frames = []
    for g in grids:
        d = f"{RTE_DATA}/results/{g}"
        if os.path.isdir(f"{d}/rows.d"): consolidate(d)    # refresh rows.csv from the per-row files
        if os.path.exists(f"{d}/rows.csv"): frames.append(pd.read_csv(f"{d}/rows.csv"))
        else: log(f"[analyze] no rows for grid {g!r} at {d}")
    if not frames: raise SystemExit(f"no rows found for grids {grids}")
    return prepare(pd.concat(frames, ignore_index=True))
def prepare(df):
    """Label, group, method_stats -> columns, churn fraction, total-communication columns when absent."""
    df["params"] = df.params.fillna("{}")
    stats = [json.loads(x) if isinstance(x, str) and x.startswith("{") else {} for x in df.get("method_stats", pd.Series([""] * len(df)))]
    for c in STATS: df[c] = [d.get(c, np.nan) for d in stats]
    ch = df.get("churn", pd.Series([""] * len(df))).fillna("")
    df["churn_frac"] = [json.loads(str(x).replace("'", '"'))["frac"] if str(x).startswith("{") else 0.0 for x in ch]
    short = lambda p: ",".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in sorted(json.loads(p).items()))
    df["label"] = [ALIAS.get(l, l) for l in (m if p == "{}" else f"{m}[{short(p)}]" for m, p in zip(df.method, df.params))]
    df["group"] = df.method.map(group_of)
    legacy = (df.method.str.startswith("midian") & ~df.method.isin(["midian_llm_descent"]) & np.array([d.get("observe_charged") is None for d in stats])
              & ~df.params.str.contains('"online": ?false', regex=True)).to_numpy()
    if legacy.any():                                          # observe-time path recompute was uncharged before 2026-09-03 15:20: r*depth comparisons + depth messages per task
        r = np.array([json.loads(p).get("r", 10) for p in df.params]); depth = np.ceil(np.log(df.n.to_numpy()) / np.log(r)).astype(int)
        df.loc[legacy, "comparisons_per_task"] = df.loc[legacy, "comparisons_per_task"].to_numpy() + (r * depth)[legacy]
        df.loc[legacy, "messages_per_task"] = df.loc[legacy, "messages_per_task"].to_numpy() + depth[legacy]
        if "total_comm_per_task" in df: df.loc[legacy, "total_comm_per_task"] = df.loc[legacy, "total_comm_per_task"].to_numpy() + depth[legacy]
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
    piv = df.pivot_table(index=cells(df) + ["seed"], columns="label", values=metric)   # label = method + params: plain MIDIAN only
    return piv[[a, b]].dropna() if {a, b} <= set(piv.columns) else pd.DataFrame()
def _t1(df, fits):
    """declared/framework lose >=0.25 from beta 0->0.5, probe-only move <=0.03"""
    b0, b5 = df[np.isclose(df.beta, 0)], df[np.isclose(df.beta, 0.5)]
    if b0.empty or b5.empty: return None, "needs beta=0 and beta=0.5"
    drop = (b0.groupby("label").success.mean() - b5.groupby("label").success.mean()).dropna()
    grp = df.drop_duplicates("label").set_index("label").group
    dec = {m: v for m, v in drop.items() if grp.get(m) in ("declared", "framework")}
    prb = {m: v for m, v in drop.items() if grp.get(m) in ("verified_central", "verified_decentral", "midian")}
    return (bool(dec) and bool(prb) and all(v >= 0.25 for v in dec.values())
            and all(abs(v) <= 0.03 for v in prb.values()),
            f"declared+framework: {fmt(dec)} | probe-based: {fmt(prb)}")
def _t2(df, fits):
    """MIDIAN == flat_probe_argmax within 0.02 at beta=0; comparisons ~ r log_r n vs flat ~ n"""
    parts, ok = [], []
    b0 = df[np.isclose(df.beta, 0)]
    for flat in (FLAT, FLAT_ON):
        d = pair(b0, REF, flat)
        if d.empty: continue
        x = float((d[REF] - d[flat]).mean())
        if flat == FLAT: ok.append(abs(x) <= 0.02)          # the pre-registered comparison is against the frozen scan
        parts.append(f"midian - {flat} = {x:+.4f} over {len(d)} (cell, seed) pairs")
    d = pair(b0, "midian[online=False]", FLAT)
    if not d.empty:
        parts.append(f"midian[online=False] - {FLAT} = {float((d['midian[online=False]'] - d[FLAT]).mean()):+.4f} "
                     f"(the max-tree alone, both frozen after build)")
    f = fits[fits.metric == "comparisons_per_task"].set_index("label") if not fits.empty else pd.DataFrame()
    if {REF, FLAT} <= set(f.index):
        m, s = f.loc[REF], f.loc[FLAT]
        ok.append(bool(m.exponent < 0.4 and s.exponent > 0.8))
        parts.append(f"comparisons: MIDIAN k={m.exponent:.2f} [{m.exp_lo:.2f},{m.exp_hi:.2f}]; "
                     f"{FLAT} k={s.exponent:.2f} [{s.exp_lo:.2f},{s.exp_hi:.2f}]")
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
    d = pair(df, "sequential_halving", FLAT)
    if d.empty: miss.append(f"sequential_halving/{FLAT} rows")
    else:
        x = float((d.sequential_halving - d[FLAT]).mean()); ok.append(abs(x) <= 0.03)
        parts.append(f"sequential_halving - {FLAT} = {x:+.3f}")
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

# ------------------------------------------------------------------ pre-registered targets, v2 (TARGETS_rte_v2.md)
SELF = lambda df: df[df.declared_source == "self_described"]
def delta(df, a, b, metric="success"):
    """Mean paired (a - b) and pair count over (cell, seed); (nan, 0) without overlap."""
    d = pair(df, a, b, metric)
    return (float((d[a] - d[b]).mean()), len(d)) if not d.empty else (np.nan, 0)
def envelope(df):
    """Mean over cells of plain MIDIAN's seed envelope (max-min success across seeds)."""
    m = df[df.label == REF]
    return float(m.groupby(cells(m)).success.agg(lambda v: v.max() - v.min()).mean()) if len(m) else np.nan
def _v1(df, fits):
    """V2-1 MIDIAN-SH: within 0.02 of halving_peer at beta<=0.25 (self-described, every shape); >= MIDIAN-0.02 at beta=0.5 collude low-skill; per-task cost = MIDIAN's"""
    s = SELF(df); lo = s[s.beta <= 0.25]
    by = {d: delta(g, "midian_sh", "sequential_halving_peer")[0] for d, g in lo.groupby("dist")}
    hi = s[np.isclose(s.beta, .5) & (s.collude == True) & (s.liar_select == "low_skill_first")]   # noqa: E712
    x, n = delta(hi, "midian_sh", REF)
    cost = {c: (float(df[df.label == "midian_sh"][c].mean()), float(df[df.label == REF][c].mean())) for c in ("comparisons_per_task", "messages_per_task")}
    if not by or not n: return None, "needs variants_f1 rows for midian_sh and sequential_halving_peer"
    ok = all(abs(v) <= 0.02 for v in by.values()) and x >= -0.02 and all(abs(a - b) < 1e-9 for a, b in cost.values())
    return ok, f"vs halving_peer at beta<=0.25 by shape: {fmt(by)}; vs MIDIAN at beta=0.5/collude/low-skill: {x:+.3f} ({n} pairs); per-task (SH, MIDIAN): " + ", ".join(f"{c} {a:.3g}/{b:.3g}" for c, (a, b) in cost.items()), max(by.values(), key=abs)
def _v2(df, fits):
    """V2-2 MIDIAN-A: beta=0.5-collude loss vs beta=0 <= 0.02 on specialist; unchanged within 0.01 at beta<=0.25; build probes <= 1.05x"""
    a = df[(df.label == "midian_a") & (df.dist == "specialist") & (df.collude == True)]   # noqa: E712
    if a.empty: return None, "needs midian_a rows on specialist"
    loss = float(a[np.isclose(a.beta, 0)].success.mean() - a[np.isclose(a.beta, .5)].success.mean())
    x, n = delta(df[df.beta <= 0.25], "midian_a", REF)
    d = pair(df, "midian_a", REF, "build_probes"); ratio = float((d["midian_a"] / d[REF]).mean()) if not d.empty else float("nan")   # same cells only
    if not np.isfinite(loss) or not n: return None, "needs midian_a at beta=0 and beta=0.5 (collude) on specialist and at beta<=0.25 paired with midian"
    return bool(loss <= 0.02 and abs(x) <= 0.01 and ratio <= 1.05 + 1e-9), f"specialist loss beta 0->0.5 = {loss:+.3f}; vs MIDIAN at beta<=0.25 {x:+.3f} ({n} pairs); build probes {ratio:.3f}x", x
def _v3(df, fits):
    """V2-3 MIDIAN-SH+A >= max(MIDIAN-SH, MIDIAN-A) - 0.01 at every beta"""
    piv = df[df.label.isin(["midian_sha", "midian_sh", "midian_a"])].pivot_table(index="beta", columns="label", values="success")
    if piv.shape[1] < 3: return None, "needs midian_sh, midian_a and midian_sha rows"
    gap = piv.midian_sha - piv[["midian_sh", "midian_a"]].max(axis=1)
    return bool((gap >= -0.01).all()), "SH+A - max(SH, A) by beta: " + ", ".join(f"{b}: {v:+.3f}" for b, v in gap.items())
def _v4(df, fits):
    """V2-4 stratified cohorts vs random (no directional expectation; reported as measured)"""
    d = pair(df, "midian_stratified", REF)
    if d.empty: return None, "needs grid stratify"
    by = (d.midian_stratified - d[REF]).groupby(level=["dist", "beta"]).mean()
    return "REPORTED", "stratified - random by (shape, beta): " + ", ".join(f"{k} {v:+.3f}" for k, v in by.items())
def _v5(df, fits):
    """V2-5 LinUCB-honest between flat_online and warm_start_bandit at beta=0; flat in beta (|beta=0.5 - beta=0| <= 0.03)"""
    b0 = df[np.isclose(df.beta, 0)]
    lo, hi = delta(b0, "linucb_honest", FLAT_ON)[0], delta(b0, "warm_start_bandit", "linucb_honest")[0]
    l = df[df.label == "linucb_honest"]
    if l.empty or not np.isfinite(lo): return None, "needs linucb_honest rows at beta=0 with flat_probe_argmax_online"
    flat = float(l[np.isclose(l.beta, .5)].success.mean() - l[np.isclose(l.beta, 0)].success.mean())
    return bool(lo >= 0 and hi >= 0 and abs(flat) <= 0.03), f"linucb - flat_online {lo:+.3f}; warm_start - linucb {hi:+.3f}; beta 0.5 - 0: {flat:+.3f}"
def _v6(df, fits):
    """V2-6 churn: MIDIAN within 0.03 of no-churn at 10% with repair <= 3% of build; halving-stale loses >= 0.05 at 30%; halving-rebuild matches at >= 10x MIDIAN's repair"""
    c, base = df[df.churn_frac > 0], df[(df.churn_frac == 0) & (df.declared_source == "self_described")]
    if c.empty or base.empty: return None, "needs churn_n1000 rows and a no-churn baseline (live_f1_n1000 / variants_f1, self-described)"
    keys = ["dist", "beta", "seed"]
    def drop(label, frac):                                  # no-churn success - churned success, paired on (shape, beta, seed)
        a = c[(c.label == label) & np.isclose(c.churn_frac, frac)].groupby(keys).success.mean()
        b = base[base.label == label].groupby(keys).success.mean()
        j = pd.concat([a, b], axis=1, join="inner"); return float((j.iloc[:, 1] - j.iloc[:, 0]).mean()) if len(j) else np.nan
    rep = lambda label: float((c[c.label == label].repair_probes_per_event / c[c.label == label].build_probes).mean())
    m10, stale30, reb30 = drop(REF, .1), drop("sequential_halving_peer_stale", .3), drop("sequential_halving_peer_rebuild", .3)
    rm, rh = rep(REF), rep("sequential_halving_peer_rebuild")
    ok = m10 <= 0.03 and rm <= 0.03 and stale30 >= 0.05 and reb30 <= 0.03 and rh >= 10 * rm
    return (bool(ok) if np.isfinite([m10, rm, stale30, reb30, rh]).all() else None,
            f"MIDIAN loss at 10% {m10:+.3f}, repair {100*rm:.2f}% of build/event; halving-stale loss at 30% {stale30:+.3f}; halving-rebuild loss at 30% {reb30:+.3f}, repair {rh/rm if rm else np.nan:.1f}x MIDIAN's")
def _v7(df, fits):
    """V2-7 n=10k, b=3, self-described: MIDIAN >= flat_online - 0.02; frameworks below flat_online on specialist by >= 0.10"""
    d = df[(df.n == 10000) & (df.b == 3)]
    x, n = delta(d, REF, FLAT_ON)
    fw = d[d.group == "framework"].success.mean() - d[d.label == FLAT_ON].success.mean() if (d.group == "framework").any() else np.nan
    if not n: return None, "needs live_n10k_v2 rows"
    return bool(x >= -0.02 and fw <= -0.10), f"MIDIAN - flat_online {x:+.3f} ({n} pairs); frameworks - flat_online {fw:+.3f}", x
def _v8(df, fits):
    """V2-8 replication (seeds 11-20): MIDIAN-V - MIDIAN = +0.02 +- 0.02 at beta<=0.25; beta=0.5 collude exposure reported"""
    r = df[df.seed >= 11]
    x, n = delta(r[r.beta <= 0.25], "midian_v", REF); y, _ = delta(r[np.isclose(r.beta, .5)], "midian_v", REF)
    if not n: return None, "needs midian_v_replication rows"
    return bool(0.0 <= x <= 0.04), f"midian_v - midian at beta<=0.25 {x:+.3f} ({n} pairs); at beta=0.5 {y:+.3f} (as measured)", x - 0.02
def _v9(df, fits):
    """V2-9 b=10: bimodal framework gap within +-0.02 of MIDIAN; heavy_tail MIDIAN >= frameworks + 0.03"""
    d = df[df.b == 10]
    if d.empty or not (d.group == "framework").any(): return None, "needs budget_b10_shapes rows"
    g = {s: float(x[x.group == "framework"].success.mean() - x[x.label == REF].success.mean()) for s, x in d.groupby("dist")}
    ok = abs(g.get("bimodal", np.nan)) <= 0.02 and g.get("heavy_tail", np.nan) <= -0.03
    return (bool(ok) if all(k in g for k in ("bimodal", "heavy_tail")) else None), "frameworks - MIDIAN at b=10: " + fmt(g)
def _v10(df, fits):
    """V2-10 internals at beta=0.5 collude: trimming (delta=1/3 vs 0) hurts plain MIDIAN by >= 0.02, not MIDIAN-A (|d| <= 0.02)"""
    d = df[np.isclose(df.beta, .5) & (df.collude == True) & df.method.isin([REF, "midian_a"])].copy()   # noqa: E712
    par = d.params.map(json.loads); d["r"], d["delta"] = [p.get("r") for p in par], [p.get("delta") for p in par]
    d = d[(d.r == 10) & d.delta.notna()]
    if d.method.nunique() < 2 or d.delta.nunique() < 2: return None, "needs internals_v2 rows (r=10, both deltas, midian and midian_a)"
    t = d.pivot_table(index="method", columns="delta", values="success"); sep = t[t.columns.max()] - t[t.columns.min()]
    return bool(sep[REF] <= -0.02 and abs(sep["midian_a"]) <= 0.02), f"delta=1/3 - delta=0 at r=10: MIDIAN {sep[REF]:+.3f}, MIDIAN-A {sep['midian_a']:+.3f}"
def targets_v2(df, fits):
    """The ten expectations of TARGETS_rte_v2.md: HIT / MISS / WITHIN_FLOOR / REPORTED / NO DATA, with the numbers."""
    out, env = [], envelope(df)
    for i, fn in enumerate((_v1, _v2, _v3, _v4, _v5, _v6, _v7, _v8, _v9, _v10), 1):
        try: ok, detail, *key = fn(df, fits)                # key = the decisive paired delta, when the target is one
        except (KeyError, ValueError, IndexError, TypeError) as e: ok, detail, key = None, f"no data ({type(e).__name__}: {e})", []
        v = ("NO DATA" if ok is None else ok if isinstance(ok, str) else "HIT" if ok else
             FLOOR if key and np.isfinite(env) and abs(key[0]) <= env else "MISS")
        out.append({"target": f"V2-{i}", "verdict": v, "name": fn.__doc__, "detail": detail + (f"  [MIDIAN seed envelope {env:.3f}]" if np.isfinite(env) else "")})
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
    """Mean +- bootstrap CI of success per method, grouped by method class, framework accountings and costs beside it."""
    g = df.groupby(["group", "label"])
    cols = [c for c in ("success_late", *STATS, "regret", "misroute_to_liar", *extra) if c in df.columns]
    return (g.success.apply(lambda s: pd.Series(dict(zip(("success", "lo", "hi"), boot(s))))).unstack()
            .join(g.success.std().rename("sd_units")).join(g.size().rename("units"))       # variance across (cell, seed) units
            .join(g[cols].mean()).sort_values("success", ascending=False).reset_index())
def roll_up(cmp_):
    """Per rival, across cells: mean paired delta, its CI, sign-test p and how often MIDIAN wins."""
    n = lambda tag: ("verdict", lambda s: int((s == tag).sum()))
    return cmp_.groupby(["group", "rival"]).agg(
        cells=("delta_mean", "size"), delta_mean=("delta_mean", "mean"), delta_lo=("delta_lo", "mean"),
        delta_hi=("delta_hi", "mean"), delta_min=("delta_mean", "min"), delta_max=("delta_mean", "max"),
        midian_better=n("midian_better"), rival_better=n("rival_better"), within_floor=n(FLOOR),
        min_sign_p=("sign_p", "min")).sort_values("delta_mean").reset_index()
def strict(df):
    """(0.2) Frameworks under the strict accounting (a task the framework did not delegate scores 0), paired vs MIDIAN."""
    f = df[(df.group == "framework") & df.success_strict.notna()]
    if f.empty: return []
    d = pd.concat([f.assign(success=f.success_strict), df[df.label == REF]]); c = paired(d)
    if c.empty: return []
    return sec("Frameworks, STRICT accounting (no fallback credit), paired vs MIDIAN", md(roll_up(c)),
               "Lenient (headline) routes the fallback pick and scores it; strict scores every non-delegated task 0. "
               "`fallback_rate` = 1 - picks/tasks under either accounting.")
UPPER = "programmatic = upper bound (S + N(0,0.05)): an honest declaration no live agent produces"
def sec(t, table, *prose, level=2):
    return ["", f"{'#' * level} {t}", "", *prose, *([""] if prose else []), table]
def by_channel(df, cmp_):
    """(0.1) Success tables per declaration channel: declared-channel readers are never pooled across channels
    (x beta, x dist and the paired-vs-MIDIAN roll-up per channel); probe-only methods once, identical by construction."""
    piv = lambda d, col: md(d.pivot_table(index=["group", "label"], columns=col, values="success").reset_index())
    dec = df.method.map(reads_declared)
    chans = sorted(df.declared_source.unique(), reverse=True)          # self_described first
    L = []
    for ch in chans if chans[1:] else []:
        d = df[dec & (df.declared_source == ch)]
        cap = f"declaration = {ch}; " + (UPPER if ch == "programmatic" else "the live channel: agents' own self-descriptions")
        for col in ("beta", "dist"): L += sec(f"Declared-channel readers, success x {col} [{ch}]", piv(d, col), cap, level=3)
        c = cmp_[cmp_.rival.isin(d.label.unique()) & (cmp_.declared_source == ch)] if not cmp_.empty else cmp_
        if not c.empty: L += sec(f"MIDIAN vs declared-channel readers, paired by seed [{ch}]", md(roll_up(c)), cap, level=3)
    rest = df if not chans[1:] else df[~dec]
    note = f"single declaration channel: {chans[0]}" + (f"; {UPPER}" if chans == ["programmatic"] else "") if not chans[1:] else "probe-only methods: identical across channels"
    return L + sec("Success x beta", piv(rest, "beta"), note, level=3)
def latency(df):
    """(0.6) The one wall-clock table: frameworks' supervisor call per task. Their clients call vLLM directly and are
    never memoised, so these are cache-consistent; they are latencies under shared-fleet load, not compute costs."""
    f = df[(df.group == "framework") & (df.wall_clock_per_task > 0)] if "wall_clock_per_task" in df else pd.DataFrame()
    if f.empty: return []
    q = f.groupby(["label", "n"]).wall_clock_per_task.quantile([.25, .5, .75]).unstack()
    q.columns = ["q25_s", "median_s", "q75_s"]
    return sec("Frameworks' supervisor latency per task (seconds; cache-consistent, under shared-fleet load)", md(q.reset_index(), "{:.2f}"),
               "Every other wall-clock column is omitted: memo hits and misses are mixed and say nothing about cost.")
def summary(out, grids, df, agg, cmp_, fits, tgts, tgts2, figs):
    L = [f"# RTE results -- {', '.join(grids)}", "",
         f"{len(df)} rows | {df.label.nunique()} methods | "
         f"{df.groupby(cells(df), dropna=False).ngroups} cells | seeds {sorted(map(int, df.seed.unique()))}",
         f"Backends: {', '.join(sorted(df.backend.astype(str).unique()))}.", "",
         "Intervals are 95% percentile bootstrap over seeds; MIDIAN-vs-rival deltas are paired by",
         f"seed. `{FLOOR}` means the delta does not exceed MIDIAN's own seed envelope in that cell.",
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
    tgts, tgts2 = targets(df, fits), targets_v2(df, fits)
    log("\n=== pre-registered targets (v1, v2) ===")
    for t in tgts + tgts2: log(f"  [{t['verdict']:>12}] {t['target']}. {t['name']}\n                 {t['detail']}")
    agg.to_csv(f"{out}/aggregate.csv", index=False)
    cmp_.to_csv(f"{out}/paired_vs_midian.csv", index=False)
    if not fits.empty: fits.to_csv(f"{out}/cost_exponents.csv", index=False)
    log(f"\n[analyze] wrote {summary(out, grids, df, agg, cmp_, fits, tgts, tgts2, figs)}")

if __name__ == "__main__":
    main()
