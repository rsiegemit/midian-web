"""Figures 3 / 4 of the submission (F_shortlists_1e5, H_routereval_shortlists) and the appendix shortlist figures
(E_shortlists_by_n, F_shortlists_1e5_appendix, G_shortlist_lift_1e5); style, names, colours and saving from
scripts/figures/lib/figspec.py (docs/figures.md).
    python scripts/figures/shortlist_condensed.py [--out DIR]   -> <out>/{E,F,G,H}_*.{png,pdf,csv}
      (<out> = $RTE_FIG_OUT or figures/paper; --from-csv is accepted and changes nothing: these figures read CSVs only)
Reads results/aggregates/shortlist/{live,routereval}.csv (per framework x shortlist x cell: seed mean;
shortlist_figs.py).
  E  (appendix) live specialist, n = 10^2 .. 10^5: per n, one bar per shortlist (instruction variants apart) = the MEAN
     over frameworks (solid honest, hatched beta = 0.5 low-skill cartel); oracle and random dotted, MIDIAN with no
     framework solid, as lines over each group.
  F  n = 10^5 (every shortlist ran there), the body shortlists sorted by the honest mean; dense (Qwen3-8B) and fusion +
     reranker are each their best instruction variant at that n by honest mean (body(); the csv's `variant` names it);
     a black dot = the best single framework. F_shortlists_1e5_appendix: every variant.
  G  (appendix) n = 10^5: each shortlist's gain over hashed TF-IDF, paired WITHIN each framework, averaged over
     frameworks (whisker = +/- 1 s.e. across frameworks). At 10^5 hashed TF-IDF is the clone shortlist (0.379 for every
     framework, erratum 25), so G is the gain over that floor.
  H  RouterEval (strong-to-weak pools, the 5,000-LLM leaderboard at m = 5,000): the body shortlists per m, lines as F.
No titles and no incompleteness marks in the figures: a cell with too few full-seed frameworks is an empty slot, and the
script prints INCOMPLETE for it; framework numbers with an erratum-28 rerun outstanding are flagged in the csv (star).
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from scripts.figures.lib import AGG, fig_out  # noqa: E402
from scripts.figures.lib import figspec as S  # noqa: E402

REG = ("beta0", "cartel")
MAIN = [
    k for k in S.SHORTLIST_ORDER if k not in ("bm25", "dense")
]  # E: the shortlists run at every n; BM25 / plain dense are 10^3 only
BODY_H = ["tfidf", "embed", "dense", "sota", "declared", "va_cohort"]  # H, in the spec's order
VARIANTS = {"dense": ("dense", "dense_icomp", "dense_idemo"), "sota": ("sota", "sota_icomp", "sota_idemo")}
LINES = ["oracle", "random_line", "midian_ref"]  # the reference lines' legend keys
INCOMPLETE = [False]  # set while drawing a figure: a slot left empty or a rerun outstanding


def load(family="live"):
    d = pd.read_csv(f"{AGG}/shortlist/{family}.csv")
    keep = (
        (d.dist == "specialist") if family == "live" else ((d.dist == "strong_to_weak") | (d.n == 5000))
    )  # RouterEval: the mixed pool; 5,000 = the leaderboard
    return d[keep & d.regime.isin(list(REG))]


MIN_FW = 6  # a bar needs >= 6 frameworks, each with the full seed count


def summarise(d):
    """(n, regime, shortlist) -> mean over frameworks, best framework, any rerun outstanding; plus the reference lines.
    Only frameworks with the full seed count of that (n, shortlist) count -- every seed its grid ran; the 10^2 / 10^3
    backfill shortlists ran 3 seeds by design, the rest 10 -- and a bar needs MIN_FW of them; a thinner cell is left out
    (an empty slot) instead of averaging a different, smaller framework set."""
    fw = d[d.shortlist != "-"]
    fw = fw[fw.seeds == fw.groupby(["n", "shortlist"]).seeds.transform("max")]
    fw = fw[fw.groupby(["n", "regime", "shortlist"]).arm.transform("nunique") >= MIN_FW]
    s = (
        fw.groupby(["n", "regime", "shortlist"])
        .agg(mean=("mean", "mean"), best=("mean", "max"), k=("arm", "nunique"), star=("rerun_outstanding", "any"))
        .reset_index()
    )
    ref = d[d.shortlist == "-"].pivot_table(index=["n", "regime"], columns="arm", values="mean")
    return s.assign(name=s.shortlist.map(S.SHORTLIST_APPENDIX)), ref


def body(s):
    """The body shortlists: dense and fusion + reranker as their best variant at each n (honest mean over frameworks),
    under the family key; `variant` names the variant drawn."""
    out = []
    for n, q in s.groupby("n"):
        h = q[q.regime == "beta0"].set_index("shortlist")["mean"]
        pick = {
            k: max((v for v in vs if v in h.index), key=h.get)
            for k, vs in VARIANTS.items()
            if any(v in h.index for v in vs)
        }
        out += [
            q[q.shortlist.isin([k for k in S.SHORTLIST_BODY if k not in VARIANTS])].assign(
                variant=lambda x: x.shortlist
            )
        ]
        out += [q[q.shortlist == v].assign(variant=v, shortlist=k) for k, v in pick.items()]
    b = pd.concat(out, ignore_index=True)
    return b.assign(name=b.shortlist.map(S.SHORTLIST_BODY))


def lines(ax, ref, n, x0, x1):
    """Oracle and random (dotted grey) and MIDIAN with no framework (green solid) at population n."""
    if (n, "beta0") not in ref.index:
        return
    o = ref.loc[(n, "beta0")]
    S.oracle(ax, x0, x1, o["oracle"])
    if pd.notna(o.get("random")):
        S.ref(ax, x0, x1, o["random"], "random")
    S.ref(ax, x0, x1, o[S.NAME["midian_ref"]])


def pair(ax, x, w, q, key, dots=False):
    """Solid honest + hatched cartel bar for one (n, shortlist); `dots`: a black dot on each = the best framework."""
    for h, r in enumerate(REG):
        if r not in q.index:
            INCOMPLETE[0] = True
            continue  # not run yet: an empty slot, no marker
        v = q.loc[r]
        INCOMPLETE[0] |= bool(v["star"])
        xx = x + (h - 0.5) * w
        S.bar(ax, xx, v["mean"], w * 0.95, key, cartel=bool(h))
        if dots:
            ax.plot(xx, v["best"], "o", ms=3, color="black", zorder=8)


def finish(fig, ax, name, data, out, keys=(), labels=None, ylabel=S.AXIS["success"], ylim=(0.2, 0.95), **kw):
    S.finish(ax, ylabel=ylabel, ylim=ylim)
    if keys:
        S.legend(ax, list(keys), labels, **kw)
    if INCOMPLETE[0]:
        print(f"[{name}] INCOMPLETE: an empty slot or a framework rerun outstanding (see the csv)")
    INCOMPLETE[0] = False
    S.save(fig, name, out)
    data.to_csv(f"{out}/{name}.csv", index=False)


def by_n(s, ref, name, srcs, names, kind, var, ncol, out, ylim=(0.2, 0.95), lines_first=False):
    """x = n (E, H): fixed slots per n, one colour per shortlist; the reference lines over each group (first in the
    legend when `lines_first`)."""
    fig, ax = S.figure(kind)
    ns = sorted(s.n.unique())
    w = 0.86 / (2 * len(srcs))
    for i, n in enumerate(ns):
        for j, src in enumerate(srcs):
            pair(ax, i + (2 * j + 1 - len(srcs)) * w, w, s[(s.n == n) & (s.shortlist == src)].set_index("regime"), src)
        lines(ax, ref, n, i - 0.46, i + 0.46)
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels(S.pow10_ticks(ns, var))
    ax.set_xlim(-0.5, len(ns) - 0.5)
    keys, labels = list(srcs) + LINES, [names[k] for k in srcs] + [S.NAME[k] for k in LINES]
    if lines_first:
        keys, labels = keys[-len(LINES) :] + keys[: -len(LINES)], labels[-len(LINES) :] + labels[: -len(LINES)]
    finish(fig, ax, name, s, out, keys, labels, ylim=ylim, ncol=ncol)


def fig_F(s, ref, name, names, kind, out, n=100000, rotate=False):
    """x = shortlist at one n, sorted by the honest mean; black dots = the best single framework."""
    q = s[s.n == n]
    order = q[q.regime == "beta0"].sort_values("mean", ascending=False).shortlist.tolist()
    fig, ax = S.figure(kind)
    w = 0.38
    for i, src in enumerate(order):
        pair(ax, i, w, q[q.shortlist == src].set_index("regime"), src, dots=True)
    lines(ax, ref, n, -0.5, len(order) - 0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_xticklabels(
        [names[x] if rotate else textwrap.fill(names[x], 14) for x in order],  # horizontal: long names on two lines
        **(dict(rotation=30, ha="right", rotation_mode="anchor") if rotate else dict(linespacing=0.95)),
    )
    finish(fig, ax, name, q, out, LINES + ["best_fw_dot"])  # one row above: inside, it meets the dots


def fig_G(d, out, n=100000):
    fw = d[(d.shortlist != "-") & (d.n == n)]
    fw = fw[fw.seeds == fw.groupby("shortlist").seeds.transform("max")]  # full-seed frameworks only, as in summarise
    base = fw[fw.shortlist == "tfidf"].set_index(["regime", "arm"])["mean"]
    rows = []
    for (r, src), q in fw[fw.shortlist != "tfidf"].groupby(["regime", "shortlist"]):
        diff = (q.set_index(["regime", "arm"])["mean"] - base).dropna()
        k = len(diff)
        if k < MIN_FW:
            INCOMPLETE[0] = True
            continue  # too few paired frameworks: left out
        half = diff.std(ddof=1) / np.sqrt(k) if k > 1 else np.nan  # +/- 1 s.e. across frameworks
        rows.append(
            dict(
                regime=r,
                shortlist=src,
                name=S.SHORTLIST_APPENDIX[src],
                lift=diff.mean(),
                half=half,
                frameworks=k,
                star=bool(q.rerun_outstanding.any()),
            )
        )
    t = pd.DataFrame(rows)
    order = t[t.regime == "beta0"].sort_values("lift", ascending=False).shortlist.tolist()
    fig, ax = S.figure("appendix")
    w = 0.38
    for i, src in enumerate(order):
        for h, r in enumerate(REG):
            q = t[(t.shortlist == src) & (t.regime == r)]
            if q.empty:
                INCOMPLETE[0] = True
                continue
            v = q.iloc[0]
            xx = i + (h - 0.5) * w
            INCOMPLETE[0] |= bool(v.star)
            S.bar(ax, xx, v.lift, w * 0.95, src, cartel=bool(h))
            S.whisker(ax, xx, v.lift, v.lift - v.half, v.lift + v.half)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([S.SHORTLIST_APPENDIX[x] for x in order], rotation=30, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.5, len(order) - 0.5)
    finish(
        fig, ax, "G_shortlist_lift_1e5", t, out, ylabel=S.AXIS["lift"].replace(" over", "\nover"), ylim=None
    )  # two lines: fits the axis


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", help="output directory (default: $RTE_FIG_OUT or figures/paper)")
    p.add_argument("--from-csv", action="store_true", help="accepted for make_all; these figures always read CSVs only")
    out = fig_out(p.parse_args(argv).out)
    d = load()
    s, ref = summarise(d)
    by_n(s, ref, "E_shortlists_by_n", MAIN, S.SHORTLIST_APPENDIX, "appendix", "n", 3, out)
    fig_F(body(s), ref, "F_shortlists_1e5", S.SHORTLIST_BODY, "body", out)
    fig_F(
        s, ref, "F_shortlists_1e5_appendix", S.SHORTLIST_APPENDIX, "appendix", out, rotate=True
    )  # the instruction variants: horizontal ones touch
    fig_G(d, out)
    s, ref = summarise(load("routereval"))
    by_n(
        body(s),
        ref,
        "H_routereval_shortlists",
        BODY_H,
        S.SHORTLIST_BODY,
        "body",
        "m",
        5,
        out,
        ylim=(0.4, 0.95),
        lines_first=True,
    )  # random is 0.527-0.544 on RouterEval: the floor at 0.4


if __name__ == "__main__":
    main()
