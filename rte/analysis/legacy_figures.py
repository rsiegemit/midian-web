"""The analyzer's own diagnostic figures F1-F7 (+F3b), written next to each grid's summary. The paper figures are
drawn by scripts/, not here."""
import json, os
import numpy as np
from .load import BUILD, COST, PLAIN, log
from .stats import boot, exponents

MARK = {"llm": "o", "replay": "s", "bernoulli": "^"}       # marker shape = backend, on every figure


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
                        lw=2.4 if h.startswith("midian") else 1.1,
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
    mid = PLAIN(df).copy()                                  # F7: MIDIAN w/o defenses internals
    par = mid.params.map(json.loads)
    mid["r"] = [p.get("r", np.nan) for p in par]
    mid["trim"] = [f"delta={p.get('delta', np.nan):.2g}, collude={c}" for p, c in zip(par, mid.collude)]
    P["F7"] = panel_plot(mid.dropna(subset=["r"]), "r", "success", "trim", "beta",
                         "F7  MIDIAN w/o defenses internals: trimming vs cohort size", f"{fd}/F7_midian_internals.png")
    return {k: v for k, v in P.items() if v}, exponents(rivals, COST + BUILD)
