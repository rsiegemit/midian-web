"""Bar figures of success for EVERY arm, grouped by n, one figure per (experiment family, liar regime, grouping); the
oracle is a dotted line across the top; error bars are the 95% seed bootstrap.
    python scripts/figures/bar_figs.py [live bernoulli replay routereval llmrouterbench]
      -> results/aggregates/bars/<family>.csv (the input of the condensed A / B figures)
         figures/bars/<family>__<regime>__<group>.{png,pdf} + figures/bars/INDEX.md (exploratory, not tracked)
One panel and one row per figure, always. No framework arms: they are drawn per shortlist in figures/shortlist
(shortlist_figs.py). Families: live (RTE live backend, 10^2-10^5, grouped by population shape; self-described channel),
bernoulli (calibrated synthetic, 10..10^7, 1000 seeds, b = 3), replay (RouterBench outcomes, 10..10^6, shapes pooled),
routereval (real LLM pools 10 / 100 / 1,000 per pool config and the 5,000-LLM leaderboard pool), llmrouterbench (20
models). The do-not-add list (lib/exclusions.py) applies. route_to_k_majority executes THREE agents per task and
majority-votes, so it can sit above the single-agent oracle: it is drawn hatched and named so. Rendering:
figspec.LEGACY_RC + EXPLORE_RC, legend_rank.install(), figspec.explore_colours."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from scripts.figures.lib import AGG, ROOT, legend_rank  # noqa: E402
from scripts.figures.lib import figspec as S  # noqa: E402
from scripts.figures.lib.exclusions import excluded  # noqa: E402
from scripts.figures.lib.grids import LIVE_GRIDS, MATRICES  # noqa: E402
from scripts.figures.lib.regimes import MATRIX_REGIME, REGIMES  # noqa: E402
from scripts.figures.lib.rows import RESULTS as R  # noqa: E402
from scripts.figures.lib.stats import ci as _ci  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

CSV_OUT, OUT = os.path.join(AGG, "bars"), os.path.join(ROOT, "figures", "bars")
NEVER = {"oracle"}  # oracle is the dotted line; the do-not-add list drops the rest
_mem = {}
JOBS = []


def rows(g):
    from rte.analyze import load as _load

    if g not in _mem:
        try:
            _mem[g] = _load([g])
        except SystemExit:
            _mem[g] = pd.DataFrame()
    return _mem[g]


MULTI = {
    "route_to_k_majority": "route-to-3 majority (3 executions per task)"
}  # route-to-many arms: not comparable at equal per-task cost, hatched


def name(label):
    if label in MULTI:
        return MULTI[label]
    if label in S.LONG_NAME:
        return S.LONG_NAME[label]
    if label.startswith("fw_") and "[" not in label:
        return S.ABBR.get(label, label[3:])
    return label.replace("_", " ")


CMAP = {}


def color(label, i=0):
    return CMAP.get(label, "#999999")


def stats_from_rows(df, beta, liar):
    """label -> (mean, lo, hi, seeds) over per-seed means of (dist, seed) units in one regime."""
    q = df[np.isclose(df.beta, beta) & (df.liar_select == liar)]
    if beta == 0 and q.empty:
        q = df[np.isclose(df.beta, 0)]  # liar-free: some grids only carry the random-liar cell
    out = {}
    for lab, g in q.groupby("label"):
        per = g.groupby("seed").success.mean()
        if per.empty:
            continue
        lo, hi = _ci(per)
        out[lab] = (float(per.mean()), float(lo), float(hi), int(len(per)))
    return out


DECL = {
    "declared_argmax",
    "declared_softmax",
    "cnp_self_bid",
    "cluster_head_router",
    "route_to_k_majority",
    "disrouter_cascade",
    "llm_supervisor",
    "random",
    "referral_network",
    "gossip_reputation_greedy",
    "linucb_honest",
}


def _panel(ax, labels, series, oracle, ns, ylim, legend=True):
    """Bars packed contiguously per n group (only the arms present there), consistent colour and order across groups."""
    x = np.arange(len(ns))
    ci = {l: i for i, l in enumerate(labels)}
    for j, n in enumerate(ns):
        present = [l for l in labels if n in series[l]]
        w = 0.84 / max(len(present), 1)
        for i, l in enumerate(present):
            m, lo, hi, _ = series[l][n]
            ax.bar(
                x[j] - 0.42 + w * (i + 0.5),
                m,
                w,
                color=color(l, ci[l]),
                label=name(l) if j == ns.index(max(n_ for n_ in ns if n_ in series[l])) else None,
                yerr=[[m - lo], [hi - m]],
                capsize=0.6,
                error_kw=dict(elinewidth=0.35, ecolor="#333"),
                hatch="////" if l in MULTI else None,
                edgecolor="white" if l in MULTI else None,
                lw=0.3,
            )
    # oracle: one continuous dotted piecewise line -- flat across each n cluster at that cluster's oracle, joined
    # between clusters
    pts = [(x[j], oracle[n]) for j, n in enumerate(ns) if n in oracle]
    if pts:
        px, py = [], []
        for xj, y in pts:
            px += [xj - 0.42, xj + 0.42]
            py += [y, y]
        ax.plot(px, py, ls=":", color=S.EXPLORE_COLOR["oracle"], lw=1.3, label="oracle", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"n = {n:,}" for n in ns])
    ax.set_ylim(*ylim)
    ax.set_ylabel("success")
    ax.grid(axis="y", alpha=0.3, lw=0.4)
    h, l = ax.get_legend_handles_labels()
    seen = {}
    [seen.setdefault(b, a) for a, b in zip(h, l)]
    order = ["oracle"] + [name(x) for x in labels if name(x) in seen]  # legend in bar order (best at n_max first)
    if legend:
        ax.legend(
            [seen[k] for k in order if k in seen],
            [k for k in order if k in seen],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=max(1, min(12, len(seen), int(ax.figure.get_figwidth() / 1.35))),
            frameon=False,
            handlelength=1.0,
            columnspacing=0.8,
            labelspacing=0.15,
        )


def draw(series, oracle, title, fname, ns, ylim=(0.2, 1.0), _jobs=None):
    """series: label -> {n: (mean, lo, hi, seeds)}; oracle: {n: mean}. ONE panel, ONE row, always (every n group side by
    side, every arm as a bar); the figure widens with the number of bars."""
    if _jobs is not None:
        _jobs.append((series, oracle, title, fname, ns, ylim))
        return fname  # deferred: colours are assigned once all labels are known
    labels = [
        l for l in series if l not in NEVER and not excluded(l) and not l.startswith("fw_")
    ]  # frameworks: figures/shortlist
    nmax = max(ns)
    labels.sort(key=lambda l: -series[l].get(nmax, series[l][max(series[l])])[0])
    nbars = sum(1 for l in labels for n in ns if n in series[l])
    fig, ax = plt.subplots(figsize=(S.cm(max(14.0, 0.28 * nbars + 5)), S.cm(7.0)))
    _panel(ax, labels, series, oracle, ns, ylim)
    ax.set_title(title)
    fig.savefig(f"{OUT}/{fname}.png", dpi=300)
    fig.savefig(f"{OUT}/{fname}.pdf")
    plt.close(fig)
    return fname


def family_live(index, recs):
    for reg, beta, liar, rtitle in REGIMES:
        for dist in ("specialist", "heavy_tail", "bimodal"):
            series, oracle, ns = {}, {}, []
            for n, grids in LIVE_GRIDS.items():
                df = pd.concat([rows(g) for g in grids], ignore_index=True)
                if df.empty:
                    continue
                if "declared_source" in df:
                    df = df[df.declared_source == "self_described"]
                df = df[df.dist == dist]
                st = stats_from_rows(df, beta, liar)
                if not st:
                    continue
                ns.append(n)
                for lab, v in st.items():
                    series.setdefault(lab, {})[n] = v
                    recs.append(
                        dict(
                            family="live",
                            regime=reg,
                            group=dist,
                            n=n,
                            label=lab,
                            mean=v[0],
                            ci_lo=v[1],
                            ci_hi=v[2],
                            seeds=v[3],
                        )
                    )
                if "oracle" in st:
                    oracle[n] = st["oracle"][0]
            if ns:
                index.append(
                    draw(series, oracle, f"live RTE, {dist}, {rtitle}", f"live__{reg}__{dist}", ns, _jobs=JOBS)
                )


def family_matrix(fam, grid, index, recs, group="all shapes pooled"):
    m = pd.read_csv(f"{R}/{grid}/matrix_success.csv")
    m = m[(m.b == 3) & (m.metric == "success")]
    for reg, _, _, rtitle in REGIMES:
        if reg not in MATRIX_REGIME:
            continue
        q = m[m.regime == MATRIX_REGIME[reg]]
        if q.empty:
            continue
        ns = sorted(int(n) for n in q.n.unique())
        series, oracle = {}, {}
        for _, r in q.iterrows():
            v = (float(r["mean"]), float(r.ci_lo), float(r.ci_hi), int(r.seeds))
            series.setdefault(r.label, {})[int(r.n)] = v
            recs.append(
                dict(
                    family=fam,
                    regime=reg,
                    group=group,
                    n=int(r.n),
                    label=r.label,
                    mean=v[0],
                    ci_lo=v[1],
                    ci_hi=v[2],
                    seeds=v[3],
                )
            )
            if r.label == "oracle":
                oracle[int(r.n)] = v[0]
        index.append(
            draw(
                series,
                oracle,
                f"{fam} ({grid}), {group}, {rtitle}",
                f"{fam}__{reg}__{group.replace(' ', '_')}",
                ns,
                _jobs=JOBS,
            )
        )


def family_routereval(index, recs):
    small = pd.concat(
        [rows("routereval_mmlu"), rows("fw_routereval_1k")], ignore_index=True
    )  # the nine frameworks ran on the m = 1,000 pools (fw_routereval_1k)
    big = pd.concat(
        [rows("routereval_mmlu5k"), rows("fw_routereval_5k")], ignore_index=True
    )  # ... and on the 5,000-LLM pool (fw_routereval_5k)
    for reg, beta, liar, rtitle in REGIMES:
        if reg == "beta01_random":
            continue
        for pool in ("strong_to_weak", "all_strong", "all_weak"):
            series, oracle, ns = {}, {}, []
            for n in (10, 100, 1000):
                st = stats_from_rows(small[(small.dist == pool) & (small.n == n)], beta, liar)
                if not st:
                    continue
                ns.append(n)
                for lab, v in st.items():
                    series.setdefault(lab, {})[n] = v
                    recs.append(
                        dict(
                            family="routereval",
                            regime=reg,
                            group=pool,
                            n=n,
                            label=lab,
                            mean=v[0],
                            ci_lo=v[1],
                            ci_hi=v[2],
                            seeds=v[3],
                        )
                    )
                if "oracle" in st:
                    oracle[n] = st["oracle"][0]
            st = stats_from_rows(big, beta, liar)
            if st:
                ns.append(5000)
                for lab, v in st.items():
                    series.setdefault(lab, {})[5000] = v
                    recs.append(
                        dict(
                            family="routereval",
                            regime=reg,
                            group=pool,
                            n=5000,
                            label=lab,
                            mean=v[0],
                            ci_lo=v[1],
                            ci_hi=v[2],
                            seeds=v[3],
                        )
                    )
                if "oracle" in st:
                    oracle[5000] = st["oracle"][0]
            if ns:
                index.append(
                    draw(
                        series,
                        oracle,
                        f"RouterEval real LLM pools, {pool} (10-1,000) + leaderboard (5,000), {rtitle}",
                        f"routereval__{reg}__{pool}",
                        ns,
                        ylim=(0.3, 1.0),
                        _jobs=JOBS,
                    )
                )


def family_llmrouterbench(index, recs):
    df = rows("llmrouterbench_pool")
    for reg, beta, liar, rtitle in REGIMES:
        st = stats_from_rows(df, beta, liar)
        if not st:
            continue
        series = {lab: {20: v} for lab, v in st.items()}
        for lab, v in st.items():
            recs.append(
                dict(
                    family="llmrouterbench",
                    regime=reg,
                    group="20 models",
                    n=20,
                    label=lab,
                    mean=v[0],
                    ci_lo=v[1],
                    ci_hi=v[2],
                    seeds=v[3],
                )
            )
        index.append(
            draw(
                series,
                {20: st["oracle"][0]} if "oracle" in st else {},
                f"LLMRouterBench 20-model pool, {rtitle}",
                f"llmrouterbench__{reg}__20",
                [20],
                ylim=(0.3, 1.0),
                _jobs=JOBS,
            )
        )


FAMILIES = {
    "live": family_live,
    "bernoulli": lambda i, r: family_matrix("bernoulli", MATRICES["bernoulli"], i, r, "specialist"),
    "replay": lambda i, r: family_matrix("replay", MATRICES["replay"], i, r),
    "routereval": family_routereval,
    "llmrouterbench": family_llmrouterbench,
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("families", nargs="*", help=f"any of {', '.join(FAMILIES)} (default: all)")
    fams = p.parse_args(argv).families or list(FAMILIES)
    plt.rcParams.update({**S.LEGACY_RC, **S.EXPLORE_RC})
    legend_rank.install()
    os.makedirs(CSV_OUT, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    index_all = []
    for fam in fams:
        index, recs = [], []
        FAMILIES[fam](index, recs)
        pd.DataFrame(recs).to_csv(f"{CSV_OUT}/{fam}.csv", index=False)
        index_all += [(fam, f) for f in index]
        print(f"[{fam}] {len(index)} figures, {len(recs)} bars")
    CMAP.update(S.explore_colours({lab for series, *_ in JOBS for lab in series}, DECL))
    for series, oracle, title, fname, ns, ylim in JOBS:
        draw(series, oracle, title, fname, ns, ylim)
    with open(f"{OUT}/INDEX.md", "a" if len(fams) < len(FAMILIES) else "w") as f:
        if len(fams) == len(FAMILIES):
            f.write(
                "# Bar figures: every arm by n, one per (family, regime, grouping)\n\nOracle = dotted line; error bars "
                "= 95% seed "
                "bootstrap; the do-not-add arms are never drawn. Data: results/aggregates/bars/<family>.csv.\n\n"
            )
        for fam, fn in index_all:
            f.write(f"- `{fn}.png` ({fam})\n")


if __name__ == "__main__":
    main()
