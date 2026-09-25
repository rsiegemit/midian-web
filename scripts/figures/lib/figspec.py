"""The figure specification for the ICLR 2027 submission, in one place (docs: the spec this implements is quoted in the
module that owns each figure). Every figure script imports its canvas, fonts, colours, names, legend order and drawing
helpers from here and nothing else: a figure carries data, axes and a legend; titles, letters, regime notes, file names
and incompleteness marks belong in the caption.

    from scripts.figures.lib import figspec as S
    S.apply()                                  # rcParams: serif (Times / STIX), 8 / 9 / 8 pt, fonttype 42, spines, grid
    fig, ax = S.figure("body")                 # 5.5 x 1.9 in ("d" 5.5 x 1.8, "appendix" 5.5 x 2.4, "pair" 5.5 x 2.0)
    ... S.bar(ax, x, m, w, "midian", cartel=False) ... S.whisker(ax, x, m, lo, hi) ... S.oracle(ax, x0, x1, y)
    S.legend(ax, keys)                         # fixed order (ORDER), frameless, inside if room else one row above
    S.save(fig, name)                          # <OUT>/<name>.pdf (vector) + .png (300 dpi)

Also here: the exploratory figures' names, palette and rcParams (figures/bars, figures/shortlist), the framework names,
and LEGACY_RC, the rcParams the retired paper_figs.py set at import, which the A / B figures inherited (see LEGACY_RC).
"""

from __future__ import annotations
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

cm = lambda x: x / 2.54  # noqa: E731  (cm -> inches)

# ---- canvas (inches) -------------------------------------------------------------------------------------------------
CANVAS = {"body": (5.5, 1.9), "d": (5.5, 1.8), "appendix": (5.5, 2.4), "pair": (5.5, 2.0)}
DPI_PNG = 300

# ---- colours (unchanged palette) and regime encoding -----------------------------------------------------------------
COLOR = {
    "midian": "#2ecc71",
    "midian_wo_defenses": "#c0392b",
    "flat_probe_argmax_online": "#3498db",
    "best_learned": "#ff7f0e",
    "best_bandit": "#9467bd",
    "declared_argmax": "#5d6d7e",
    "random": "#bbbbbb",
    "best_framework": "#8e6c3a",
    "oracle": "#7f8c8d",
    "flat_scan": "#3498db",
    "fw_band": "#e5e5e5",
    "grey": "#8c8c8c",
}  # fw_band: D's framework band
HATCH = "////"  # beta = 0.5 low-skill-first cartel; solid = honest (beta = 0)
B_SHADE = {1: 0.5, 3: 0.0, 5: -0.4}  # b = 1 light, 3 as is, 5 dark (lighten > 0, darken < 0)
B_LINESTYLE = {1: ":", 3: "-", 5: "--"}  # line figures (D): b = 1 dotted, 3 solid, 5 dashed

# ---- names: legend strings and tick labels (spec section 2)
# -----------------------------------------------------------
NAME = {
    "oracle": "oracle",
    "midian": "MIDIAN",
    "midian_wo_defenses": "MIDIAN w/o defenses",
    "flat_probe_argmax_online": "flat probe argmax",
    "best_learned": "best learned/declared router",
    "best_bandit": "best bandit",
    "declared_argmax": "declared argmax",
    "random": "random",
    # "best text shortlist" in the caption: the long form cannot fit a two-row legend
    "best_framework": "best framework",
    "midian_ref": "MIDIAN, no framework",
    "best_fw_dot": "best framework",
    "flat_scan": "any flat scan",
    "random_line": "random",
}
ORDER = [
    "oracle",
    "midian",
    "midian_wo_defenses",
    "flat_probe_argmax_online",
    "best_learned",
    "best_bandit",
    "declared_argmax",
    "best_framework",
    "random",
    # best_framework beside declared argmax: both route on descriptions                                   # the
    # same in every figure that shares arms
]
SHORTLIST_BODY = {
    "declared": "declared top-k",
    "va_cohort": "MIDIAN cohort",
    "dense": "dense (Qwen3-8B)",
    "embed": "MiniLM",
    "sota": "fusion + reranker",
    "bm25": "BM25",
    "tfidf": "hashed TF-IDF",
}
SHORTLIST_APPENDIX = {
    **SHORTLIST_BODY,
    "dense_icomp": "dense, competence instr.",
    "dense_idemo": "dense, demonstration instr.",
    "sota_icomp": "fusion + reranker, competence instr.",
    "sota_idemo": "fusion + reranker, demonstration instr.",
    "dense": "dense (Qwen3-8B), no instr.",
    "sota": "fusion + reranker, no instr.",
}
SHORTLIST_ORDER = [
    "tfidf",
    "bm25",
    "embed",
    "dense",
    "dense_icomp",
    "dense_idemo",
    "sota",
    "sota_icomp",
    "sota_idemo",
    "declared",
    "va_cohort",
]  # left to right wherever a figure shows several shortlists
SHORTLIST_COLOR = {
    "tfidf": "#b0b0b0",
    "bm25": "#17becf",
    "embed": "#1f77b4",
    "dense": "#c5b0d5",
    "dense_icomp": "#9467bd",
    "dense_idemo": "#5b2c83",
    "sota": "#ff9896",
    "sota_icomp": "#d62728",
    "sota_idemo": "#8b0000",
    "declared": "#e7ba52",
    "va_cohort": "#117a3d",
}
BACKEND = {
    "live": "live",
    "bernoulli": "Bernoulli",
    "replay": "RouterBench replay",
    "routereval": "RouterEval",
    "llmrouterbench": "LLMRouterBench",
}
FWS = [
    "fw_autogen",
    "fw_camel_workforce",
    "fw_crewai",
    "fw_google_adk",
    "fw_langgraph",
    "fw_llamaindex",
    "fw_maf",
    "fw_magentic_one",
    "fw_openai_agents",
    "fw_smolagents",
]
FW_NAME = {
    "fw_google_adk": "Google ADK",
    "fw_crewai": "CrewAI",
    "fw_magentic_one": "Magentic-One",
    "fw_openai_agents": "OpenAI Agents",
    "fw_llamaindex": "LlamaIndex",
    "fw_langgraph": "LangGraph",
    "fw_autogen": "AutoGen",
    "fw_camel_workforce": "CAMEL",
    "fw_maf": "MAF",
    "fw_smolagents": "smolagents",
}
ABBR = {
    "fw_autogen": "autogen",
    "fw_camel_workforce": "camel",
    "fw_crewai": "crewai",
    "fw_google_adk": "adk",
    "fw_langgraph": "langgraph",
    "fw_llamaindex": "llama",
    "fw_maf": "maf",
    "fw_magentic_one": "magentic",
    "fw_openai_agents": "openai",
    "fw_smolagents": "smol",
}
AXIS = {
    "success": "task success",
    "rel": "success relative to oracle",
    "n": "population size n",
    "T": "queries served T",
    "work": "messages + comparisons per query",
    "energy": "energy per query (J)",
    "lift": "gain in task success over hashed TF-IDF",
}


def pow10(v: float, var: str = "") -> str:
    """$10^k$ (exact powers above 10) or $10$, $5{,}000$; `var` prefixes "n = " style labels (e.g. var="m")."""
    import math

    k = math.log10(v) if v > 0 else 0
    s = f"10^{{{int(round(k))}}}" if abs(k - round(k)) < 1e-9 and round(k) > 1 else f"{int(v):,}".replace(",", "{,}")
    return f"${var + ' = ' if var else ''}{s}$"


def pow10_ticks(values, var: str = "") -> list[str]:
    """Tick labels for a population axis: the first carries the variable ($n = 10^2$, $10^3$, ...)."""
    return [pow10(v, var if i == 0 else "") for i, v in enumerate(values)]


def apply():
    """rcParams for every submission figure. Serif = Times where installed, else STIX (TrueType, bundled): never an
    OpenType-CFF face such as Nimbus Roman, which fonttype 42 embeds with a font-type mismatch."""
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.titlesize": 9,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.linewidth": 0.3,
            "grid.alpha": 0.4,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
            "savefig.dpi": DPI_PNG,
            "axes.formatter.use_mathtext": True,
        }
    )


def figure(kind: str = "body", ncols: int = 1):
    apply()
    return plt.subplots(1, ncols, figsize=CANVAS[kind])


def shade(color: str, b: int) -> tuple:
    """The budget encoding: b = 1 lighter, b = 5 darker than the arm's colour."""
    import numpy as np, matplotlib.colors as mc

    c, t = np.array(mc.to_rgb(color)), B_SHADE.get(b, 0.0)
    return tuple(c + (1 - c) * t) if t > 0 else tuple(c * (1 + t))


def color(key: str) -> str:
    """An arm's or a shortlist's colour."""
    return COLOR[key] if key in COLOR else SHORTLIST_COLOR[key]


def bar(ax, x, h, w, key: str, cartel: bool = False, b: int | None = None, bottom: float = 0.0, z: float = 2, **kw):
    """One bar in the arm's (or shortlist's) colour (budget-shaded when b is given), hatched under the cartel."""
    col = shade(color(key), b) if b else color(key)
    return ax.bar(
        x, h, w, bottom=bottom, color=col, edgecolor="black", lw=0.3, hatch=HATCH if cartel else None, zorder=z, **kw
    )


def whisker(ax, x, m, lo, hi):
    """+/- 1 s.e.: black, lw 0.6, cap 1.5 pt."""
    ax.errorbar(
        x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="black", elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=6
    )


def oracle(ax, x0, x1, y):
    ax.hlines(y, x0, x1, colors=COLOR["oracle"], linestyles=":", lw=1.0, zorder=7)


def ref(ax, x0, x1, y, key: str = "midian_ref"):
    """Whole-population reference line: MIDIAN green solid (E, F, H); random grey dotted."""
    style = {
        "midian_ref": dict(colors=COLOR["midian"], linestyles="-"),
        "random": dict(colors=COLOR["random"], linestyles=":"),
    }[key]
    ax.hlines(y, x0, x1, lw=1.0, zorder=7, **style)


def handle(key: str):
    """Legend proxy for an arm key, a reference line or the best-framework dot."""
    if key == "oracle":
        return Line2D([], [], color=COLOR["oracle"], ls=":", lw=1.0)
    if key == "midian_ref":
        return Line2D([], [], color=COLOR["midian"], ls="-", lw=1.0)
    if key == "random_line":
        return Line2D([], [], color=COLOR["random"], ls=":", lw=1.0)
    if key == "best_fw_dot":
        return Line2D([], [], color="black", marker="o", ms=3, ls="none")
    return Patch(facecolor=color(key), edgecolor="black", lw=0.3)


def budget_handles(pad: int = 0):
    """The three-swatch budget legend (grey); `pad` blank entries first, so the swatches fill a legend column of their
    own."""
    return (
        [Patch(visible=False)] * pad
        + [Patch(facecolor=shade(COLOR["grey"], b), edgecolor="black", lw=0.3) for b in (1, 3, 5)],
        [""] * pad + ["b = 1", "b = 3", "b = 5"],
    )


def legend(ax, keys, labels=None, where: str = "above", ncol: int | None = None, extra=None, handles=None, **kw):
    """One frameless legend, entries in the given (fixed) order, never re-ranked (matplotlib's Legend is built directly,
    so the value-ranking wrapper of legend_rank.install() does not apply). where: "above" (rows over the axes) or a
    matplotlib loc inside the axes. handles: proxies in place of handle(key) (e.g. lines instead of patches)."""
    from matplotlib.legend import Legend

    hs = list(handles) if handles is not None else [handle(k) for k in keys]
    ls = list(labels or [NAME.get(k, k) for k in keys])
    if extra:
        hs += extra[0]
        ls += extra[1]
    opt = dict(frameon=False, handlelength=1.2, columnspacing=0.8)
    opt.update(
        dict(ncol=ncol or len(hs), loc="lower center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2)
        if where == "above"
        else dict(ncol=ncol or 1, loc=where)
    )
    opt.update(kw)
    leg = Legend(ax, hs, ls, **opt)
    ax.legend_ = leg
    leg._remove_method = ax._remove_legend
    return leg


def finish(ax, ylabel: str | None = None, xlabel: str | None = None, ylim=None, decimals: int = 2):
    """Axis labels (sentence case), value-axis limits, two-decimal ticks, no title."""
    from matplotlib.ticker import FormatStrFormatter

    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylim:
        ax.set_ylim(*ylim)
    if decimals is not None and ax.get_yscale() == "linear":
        ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))
    ax.set_title("")


def save(fig, name: str, out: str):
    """<out>/<name>.pdf (vector, text as text) + <out>/<name>.png (300 dpi)."""
    fig.savefig(f"{out}/{name}.pdf")
    fig.savefig(f"{out}/{name}.png", dpi=DPI_PNG)
    plt.close(fig)
    print(f"[{name}] written")


# ---- exploratory figures (figures/bars, figures/shortlist) and legacy rcParams
# --------------------------------------------
LONG_NAME = {
    "oracle": "oracle",
    "sequential_halving_peer": "seq. halving (peer)",
    "sequential_halving": "seq. halving (trusted)",
    "midian": "MIDIAN",
    "midian_wo_audit": "MIDIAN w/o audits",
    "midian_wo_verify": "MIDIAN w/o verification",
    "midian_wo_defenses": "MIDIAN w/o defenses",
    "flat_probe_argmax_online": "flat probe argmax (online)",
    "knn_router": "RouterBench KNN router",
    "mlp_router": "RouterBench MLP router",
    "declared_argmax": "declared argmax",
    "warm_start_bandit": "warm-start bandit",
    "random": "random",
}
EXPLORE_COLOR = {
    "oracle": "#999999",
    "midian_wo_defenses": "#c0392b",
    "midian_wo_audit": "#e67e22",
    "midian_wo_audit_r5": "#f1c40f",
    "midian_wo_verify": "#7b241c",
    "midian": "#2ecc71",
    "flat_probe_argmax_frozen": "#7f8c8d",
    "flat_probe_argmax_online": "#3498db",
    "sequential_halving": "#2c3e50",
    "sequential_halving_peer": "#8e44ad",
    "sequential_halving[churn_mode=rebuild,peer_reported=True]": "#8e44ad",
    "sequential_halving[churn_mode=stale,peer_reported=True]": "#bb8fce",
    "warm_start_bandit": "#27ae60",
    "linucb_honest": "#16a085",
    "declared_argmax": "#5d6d7e",
    "llm_supervisor": "#34495e",
    "fw_autogen": "#2980b9",
    "fw_magentic_one": "#1f618d",
    "fw_magentic_one[supervisor=Qwen/Qwen2.5-14B-Instruct]": "#5dade2",
    "random": "#ccc",
}
FW_RAMP = [
    "#08306b",
    "#08519c",
    "#2171b5",
    "#4292c6",
    "#6baed6",
    "#9ecae1",
    "#c6dbef",
    "#3f007d",
    "#54278f",
    "#6a51a3",
    "#807dba",
    "#9e9ac8",
]
DECL_RAMP = [
    "#3b3b3b",
    "#5c5c5c",
    "#7f7f7f",
    "#a3a3a3",
    "#8c510a",
    "#bf812d",
    "#dfc27d",
    "#543005",
    "#c7c7c7",
    "#e0e0e0",
    "#4d4d4d",
]
EXPLORE_RC = {
    "pdf.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "figure.constrained_layout.use": True,
}
# LEGACY_RC: set at import by the retired scripts/paper_figs.py. Every script that imported it (bar_figs,
# shortlist_figs,
# and condensed_figs through them) drew under it, so A / B carried its tick widths, line defaults and layout pads under
# apply(). Those scripts now set it explicitly, so their figures render as before; no other figure gets it.
LEGACY_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "grid.linewidth": 0.4,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.02,
    "figure.constrained_layout.w_pad": 0.02,
}


def explore_colours(labels, decl):
    """One colour per label, the same in every exploratory figure and every run: EXPLORE_COLOR for the arms it names, a
    blue/purple ramp for frameworks, greys/browns for the declaration-channel arms (`decl`), tab20 ramps for the rest,
    handed out in sorted-label order."""
    import matplotlib.colors as mc

    other = (
        [mc.to_hex(plt.cm.tab20(i)) for i in range(20)]
        + [mc.to_hex(plt.cm.tab20b(i)) for i in range(20)]
        + [mc.to_hex(plt.cm.tab20c(i)) for i in range(20)]
    )
    out = {lab: mc.to_hex(EXPLORE_COLOR[lab]) for lab in labels if lab in EXPLORE_COLOR}
    used = set(out.values())
    ramps = {
        "fw": iter([c for c in FW_RAMP if c not in used]),
        "decl": iter([c for c in DECL_RAMP if c not in used]),
        "other": iter([c for c in other if c not in used]),
    }
    for lab in sorted(labels):
        if lab in out:
            continue
        kind = "fw" if lab.startswith("fw_") else "decl" if lab in decl else "other"
        out[lab] = next(ramps[kind], None) or next(ramps["other"])
    return out
