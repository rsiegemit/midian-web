"""Bar figures of success for EVERY arm, grouped by n, one figure per (experiment family, liar regime, grouping); the oracle
is a dotted line across the top; error bars are the 95% seed bootstrap.
    python scripts/bar_figs.py [live bernoulli replay routereval llmrouterbench]   -> figures/bars/<family>__<regime>__<group>.{png,pdf}
                                                                                    + figures/bars/<family>.csv + figures/bars/INDEX.md
One panel and one row per figure, always. Families: live (RTE live backend, 10^2-10^5, grouped by population shape; self-described channel; the ten frameworks are the
pre-registered TF-IDF adapter -- at 10^5 that row is the clone artefact of erratum 25, see M6 for the other shortlists),
bernoulli (calibrated synthetic, 10..10^7, 1000 seeds, b = 3), replay (RouterBench outcomes, 10..10^6, shapes pooled),
routereval (real LLM pools 10 / 100 / 1,000 per pool config and the 5,000-LLM leaderboard pool), llmrouterbench (20 models).
The trusted-observer halving arm is never drawn (erratum 26)."""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rte.analyze import RTE_DATA as _RD, load as _load, FLAT_ON
from extra_figs import COLOR as _COLOR, HALP, ci as _ci
from paper_figs import NAME as _NAME, ABBR, cm

R = f"{_RD}/results"; OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "bars"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6, "axes.linewidth": 0.6, "figure.constrained_layout.use": True})
NEVER = {"sequential_halving", "oracle"}                        # oracle is the dotted line, trusted halving is never reported
REGIMES = [("beta0", 0.0, "random", "honest (β = 0)"), ("beta01_random", 0.1, "random", "β = 0.1, random liars"),
           ("beta025_random", 0.25, "random", "β = 0.25, random liars"), ("beta025_cartel", 0.25, "low_skill_first", "β = 0.25, low-skill cartel"),
           ("beta05_random", 0.5, "random", "β = 0.5, random liars"), ("cartel", 0.5, "low_skill_first", "β = 0.5, low-skill cartel")]
MATRIX_REGIME = {"beta0": "beta=0 (no liars)", "beta025_random": "beta=0.25 random liars", "beta025_cartel": "beta=0.25 CARTEL (low-skill-first)",
                 "beta05_random": "beta=0.5 random liars", "cartel": "beta=0.5 CARTEL (low-skill-first)"}
LIVE_GRIDS = {100: ["fw_live_n100", "learned_n100", "live_core_n100", "fw_live_n100_lowskill"],
              1000: ["fw_live_n1000", "live_f1_n1000", "variants_f1", "learned_f1", "fw_live_n1000_lowskill"],
              10000: ["learned_n10k", "live_n10k_v2", "fw_live_n10k_cartel"], 100000: ["live_n100k"]}
_mem = {}; JOBS = []


def rows(g):
    if g not in _mem:
        try: _mem[g] = _load([g])
        except SystemExit: _mem[g] = pd.DataFrame()
    return _mem[g]


def name(label):
    if label in _NAME: return _NAME[label]
    if label.startswith("fw_") and "[" not in label: return ABBR.get(label, label[3:])
    return label.replace("_", " ")


COLOURS_JSON = f"{OUT}/COLOURS.json"; CMAP = {}
FW_RAMP = ["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6", "#9ecae1", "#c6dbef", "#3f007d", "#54278f", "#6a51a3", "#807dba", "#9e9ac8"]
DECL_RAMP = ["#3b3b3b", "#5c5c5c", "#7f7f7f", "#a3a3a3", "#8c510a", "#bf812d", "#dfc27d", "#543005", "#c7c7c7", "#e0e0e0", "#4d4d4d"]
OTHER_RAMP = [plt.cm.tab20(i) for i in range(20)] + [plt.cm.tab20b(i) for i in range(20)] + [plt.cm.tab20c(i) for i in range(20)]


def assign_colours(all_labels):
    """One colour per label, identical in every figure: the paper palette (figures/COLOURS.md) for the arms it names, a blue/purple
    ramp for the ten frameworks, greys/browns for declaration-channel arms, tab20 ramps for the rest; persisted to COLOURS.json."""
    import json, matplotlib.colors as mc
    if os.path.exists(COLOURS_JSON): CMAP.update(json.load(open(COLOURS_JSON)))
    used = set(CMAP.values()); ramps = {"fw": iter([c for c in FW_RAMP if c not in used]), "decl": iter([c for c in DECL_RAMP if c not in used]),
                                        "other": iter([mc.to_hex(c) for c in OTHER_RAMP if mc.to_hex(c) not in used])}
    for l in sorted(all_labels):
        if l in CMAP: continue
        if l in _COLOR: CMAP[l] = mc.to_hex(_COLOR[l]); continue
        kind = "fw" if l.startswith("fw_") else "decl" if l in DECL else "other"
        try: CMAP[l] = next(ramps[kind])
        except StopIteration: CMAP[l] = next(ramps["other"])
    json.dump(CMAP, open(COLOURS_JSON, "w"), indent=1, sort_keys=True)


def color(label, i=0):
    return CMAP.get(label, "#999999")


def stats_from_rows(df, beta, liar):
    """label -> (mean, lo, hi, seeds) over per-seed means of (dist, seed) units in one regime."""
    q = df[np.isclose(df.beta, beta) & (df.liar_select == liar)]
    if beta == 0 and q.empty: q = df[np.isclose(df.beta, 0)]                     # liar-free: some grids only carry the random-liar cell
    out = {}
    for lab, g in q.groupby("label"):
        per = g.groupby("seed").success.mean()
        if per.empty: continue
        lo, hi = _ci(per); out[lab] = (float(per.mean()), float(lo), float(hi), int(len(per)))
    return out


DECL = {"declared_argmax", "declared_softmax", "cnp_self_bid", "cluster_head_router", "route_to_k_majority", "disrouter_cascade", "llm_supervisor",
        "random", "referral_network", "gossip_reputation_greedy", "linucb_honest"}


def _panel(ax, labels, series, oracle, ns, ylim, legend=True):
    """Bars packed contiguously per n group (only the arms present there), consistent colour and order across groups."""
    x = np.arange(len(ns)); ci = {l: i for i, l in enumerate(labels)}
    for j, n in enumerate(ns):
        present = [l for l in labels if n in series[l]]; w = 0.84 / max(len(present), 1)
        for i, l in enumerate(present):
            m, lo, hi, _ = series[l][n]
            ax.bar(x[j] - 0.42 + w * (i + 0.5), m, w, color=color(l, ci[l]), label=name(l) if j == ns.index(max(n_ for n_ in ns if n_ in series[l])) else None,
                   yerr=[[m - lo], [hi - m]], capsize=0.6, error_kw=dict(elinewidth=0.35, ecolor="#333"))
    ox = [x[j] for j, n in enumerate(ns) if n in oracle]; oy = [oracle[n] for n in ns if n in oracle]
    if ox: ax.plot(ox, oy, ls=":", color=_COLOR.get("oracle", "#7f8c8d"), marker="o", ms=3, lw=1.2, label="oracle", zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([f"n = {n:,}" for n in ns]); ax.set_ylim(*ylim); ax.set_ylabel("success"); ax.grid(axis="y", alpha=.3, lw=0.4)
    h, l = ax.get_legend_handles_labels(); seen = {}; [seen.setdefault(b, a) for a, b in zip(h, l)]
    order = ["oracle"] + [name(x) for x in labels if name(x) in seen]                      # legend in bar order (best at n_max first)
    if legend: ax.legend([seen[k] for k in order if k in seen], [k for k in order if k in seen], loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(12, len(seen)), frameon=False, handlelength=1.0, columnspacing=0.8, labelspacing=0.15)


def draw(series, oracle, title, fname, ns, ylim=(0.2, 1.0), _jobs=None):
    """series: label -> {n: (mean, lo, hi, seeds)}; oracle: {n: mean}. ONE panel, ONE row, always (every n group side by side,
    every arm as a bar); the figure widens with the number of bars."""
    if _jobs is not None: _jobs.append((series, oracle, title, fname, ns, ylim)); return fname      # deferred: colours are assigned once all labels are known
    labels = [l for l in series if l not in NEVER]
    nmax = max(ns); labels.sort(key=lambda l: -series[l].get(nmax, series[l][max(series[l])])[0])
    nbars = sum(1 for l in labels for n in ns if n in series[l])
    fig, ax = plt.subplots(figsize=(cm(max(14.0, 0.28 * nbars + 5)), cm(7.0)))
    _panel(ax, labels, series, oracle, ns, ylim); ax.set_title(title)
    fig.savefig(f"{OUT}/{fname}.png", dpi=300); fig.savefig(f"{OUT}/{fname}.pdf"); plt.close(fig); return fname


def family_live(index, recs):
    for reg, beta, liar, rtitle in REGIMES:
        for dist in ("specialist", "heavy_tail", "bimodal"):
            series, oracle, ns = {}, {}, []
            for n, grids in LIVE_GRIDS.items():
                df = pd.concat([rows(g) for g in grids], ignore_index=True)
                if df.empty: continue
                if "declared_source" in df: df = df[df.declared_source == "self_described"]
                df = df[df.dist == dist]
                st = stats_from_rows(df, beta, liar)
                if not st: continue
                ns.append(n)
                for lab, v in st.items():
                    series.setdefault(lab, {})[n] = v; recs.append(dict(family="live", regime=reg, group=dist, n=n, label=lab, mean=v[0], ci_lo=v[1], ci_hi=v[2], seeds=v[3]))
                if "oracle" in st: oracle[n] = st["oracle"][0]
            if ns: index.append(draw(series, oracle, f"live RTE, {dist}, {rtitle}", f"live__{reg}__{dist}", ns, _jobs=JOBS))


def family_matrix(fam, grid, index, recs, group="all shapes pooled"):
    m = pd.read_csv(f"{R}/{grid}/matrix_success.csv"); m = m[(m.b == 3) & (m.metric == "success")]
    for reg, _, _, rtitle in REGIMES:
        if reg not in MATRIX_REGIME: continue
        q = m[m.regime == MATRIX_REGIME[reg]]
        if q.empty: continue
        ns = sorted(int(n) for n in q.n.unique()); series, oracle = {}, {}
        for _, r in q.iterrows():
            v = (float(r["mean"]), float(r.ci_lo), float(r.ci_hi), int(r.seeds)); series.setdefault(r.label, {})[int(r.n)] = v
            recs.append(dict(family=fam, regime=reg, group=group, n=int(r.n), label=r.label, mean=v[0], ci_lo=v[1], ci_hi=v[2], seeds=v[3]))
            if r.label == "oracle": oracle[int(r.n)] = v[0]
        index.append(draw(series, oracle, f"{fam} ({grid}), {group}, {rtitle}", f"{fam}__{reg}__{group.replace(' ', '_')}", ns, _jobs=JOBS))


def family_routereval(index, recs):
    small, big = rows("routereval_mmlu"), rows("routereval_mmlu5k")
    for reg, beta, liar, rtitle in REGIMES:
        if reg == "beta01_random": continue
        for pool in ("strong_to_weak", "all_strong", "all_weak"):
            series, oracle, ns = {}, {}, []
            for n in (10, 100, 1000):
                st = stats_from_rows(small[(small.dist == pool) & (small.n == n)], beta, liar)
                if not st: continue
                ns.append(n)
                for lab, v in st.items(): series.setdefault(lab, {})[n] = v; recs.append(dict(family="routereval", regime=reg, group=pool, n=n, label=lab, mean=v[0], ci_lo=v[1], ci_hi=v[2], seeds=v[3]))
                if "oracle" in st: oracle[n] = st["oracle"][0]
            st = stats_from_rows(big, beta, liar)
            if st:
                ns.append(5000)
                for lab, v in st.items(): series.setdefault(lab, {})[5000] = v; recs.append(dict(family="routereval", regime=reg, group=pool, n=5000, label=lab, mean=v[0], ci_lo=v[1], ci_hi=v[2], seeds=v[3]))
                if "oracle" in st: oracle[5000] = st["oracle"][0]
            if ns: index.append(draw(series, oracle, f"RouterEval real LLM pools, {pool} (10-1,000) + leaderboard (5,000), {rtitle}", f"routereval__{reg}__{pool}", ns, ylim=(0.3, 1.0), _jobs=JOBS))


def family_llmrouterbench(index, recs):
    df = rows("llmrouterbench_pool")
    for reg, beta, liar, rtitle in REGIMES:
        st = stats_from_rows(df, beta, liar)
        if not st: continue
        series = {lab: {20: v} for lab, v in st.items()}
        for lab, v in st.items(): recs.append(dict(family="llmrouterbench", regime=reg, group="20 models", n=20, label=lab, mean=v[0], ci_lo=v[1], ci_hi=v[2], seeds=v[3]))
        index.append(draw(series, {20: st["oracle"][0]} if "oracle" in st else {}, f"LLMRouterBench 20-model pool, {rtitle}", f"llmrouterbench__{reg}__20", [20], ylim=(0.3, 1.0), _jobs=JOBS))


FAMILIES = {"live": family_live, "bernoulli": lambda i, r: family_matrix("bernoulli", "bernoulli_scale_v5", i, r, "specialist"),
            "replay": lambda i, r: family_matrix("replay", "replay_scale_v5", i, r), "routereval": family_routereval, "llmrouterbench": family_llmrouterbench}

if __name__ == "__main__":
    fams = sys.argv[1:] or list(FAMILIES); index_all = []
    for fam in fams:
        index, recs = [], []; FAMILIES[fam](index, recs)
        pd.DataFrame(recs).to_csv(f"{OUT}/{fam}.csv", index=False); index_all += [(fam, f) for f in index]; print(f"[{fam}] {len(index)} figures, {len(recs)} bars")
    assign_colours({l for series, *_ in JOBS for l in series})
    for series, oracle, title, fname, ns, ylim in JOBS: draw(series, oracle, title, fname, ns, ylim)
    with open(f"{OUT}/INDEX.md", "a" if len(fams) < len(FAMILIES) else "w") as f:
        if len(fams) == len(FAMILIES): f.write("# Bar figures: every arm by n, one per (family, regime, grouping)\n\nOracle = dotted line; error bars = 95% seed bootstrap; the trusted-observer halving arm is never drawn (erratum 26). Data: `<family>.csv` beside each set.\n\n")
        for fam, fn in index_all: f.write(f"- `{fn}.png` ({fam})\n")
