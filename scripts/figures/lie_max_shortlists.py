"""Appendix panel F_max_lie: Figure F's six body shortlists at live n = 10^5 under the max lie (lie_mode max: every
cartel member claims perfect skill in every family; beta = 0.5 low-skill-first cartel, specialist, b = 3, seeds 1-3,
ten frameworks) against the standard lie (true skill + 0.4, clipped) of Figure F's cartel bars.
Grids: lie_max_fw_n100k_va (MIDIAN shortlist), _tfidf (hashed TF-IDF), _em (MiniLM), _dense_ic / _sota_ic (dense and
fusion + reranker with the competence instruction, the variant Figure F draws at 10^5), lie_max_fw_n100k (declared
top-k). Bar (hatched: cartel) = the mean over frameworks of each framework's seed mean; black dot = the best framework;
black tick = the same shortlist's standard-lie cartel bar (<out>/F_shortlists_1e5.csv). Same order, palette, edge rule
and full-seed rule as F (a framework counts only with every seed its grid ran, a bar needs lib/pools.MIN_FW of
them); the reference lines are F's (oracle, random, MIDIAN with no framework; the claim-free arms, unchanged by the
lie). The csv adds the strict mean (method_stats.success_strict) and the framework count k.
    python scripts/figures/lie_max_shortlists.py [--out DIR] [--from-csv]
      -> <out>/F_max_lie.{png,pdf,csv} + results/aggregates/figures/F_max_lie.csv and its lines in refs.csv;
         --from-csv redraws from those two alone (no $RTE_DATA)."""

import argparse
import os
import sys
import textwrap

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from scripts.figures.lib import NEW, fig_out, publish, republish  # noqa: E402
from scripts.figures.lib import figspec as S  # noqa: E402

NAME, N = "F_max_lie" + NEW, 100000  # 10-seed build: F's reference lines at 10 seeds
GRIDS = {
    "va_cohort": "lie_max_fw_n100k_va",
    "tfidf": "lie_max_fw_n100k_tfidf",
    "embed": "lie_max_fw_n100k_em",
    "dense": "lie_max_fw_n100k_dense_ic",
    "sota": "lie_max_fw_n100k_sota_ic",
    "declared": "lie_max_fw_n100k",
}  # in SHORTLIST_ORDER


def shortlist(key, grid):
    """One shortlist under the max lie: the row of the figure csv, or None (printed) when too few frameworks."""
    from scripts.figures.lib.exclusions import excluded
    from scripts.figures.lib.pools import MIN_FW, source
    from scripts.figures.lib.rows import read_rows, stat

    x = read_rows(grid).drop_duplicates("rid")
    x = x[x.method.astype(str).str.startswith("fw_") & ~x.method.map(excluded)]
    assert set(x.lie_mode) == {"max"} and set(x.n) == {N} and set(x.b) == {3} and set(x.beta) == {0.5}, grid
    assert set(x.liar_select) == {"low_skill_first"} and set(x.dist) == {"specialist"}, grid
    variant = x.params.astype(str).map(source)
    assert variant.str.startswith(key).all() and variant.nunique() == 1, (grid, set(variant))
    x = x.assign(strict=stat(x, "success_strict"))
    per = x.groupby(["method", "seed"])[["success", "strict"]].mean().groupby("method")
    fw = per.mean().assign(seeds=per.size())
    fw = fw[fw.seeds == fw.seeds.max()]  # full-seed frameworks only
    if len(fw) < MIN_FW:
        print(f"[{NAME}] INCOMPLETE: {grid} has {len(fw)} full-seed frameworks (< {MIN_FW}); bar left out")
        return None
    return dict(
        n=N,
        shortlist=key,
        name=S.SHORTLIST_BODY[key],
        variant=variant.iloc[0],
        mean=fw.success.mean(),
        best=fw.success.max(),
        strict=fw.strict.mean(),
        k=len(fw),
        seeds=int(fw.seeds.max()),
    )


def compute(out):
    """The figure csv (one row per shortlist, standard-lie cartel values alongside) and F's reference lines."""
    from scripts.figures.shortlist_condensed import load, summarise

    rec = pd.DataFrame([r for k, g in GRIDS.items() if (r := shortlist(k, g))])
    f = pd.read_csv(f"{out}/F_shortlists_1e5{NEW}.csv")
    f = f[f.regime == "cartel"].set_index("shortlist")
    rec = rec.assign(
        standard_mean=rec.shortlist.map(f["mean"]),
        standard_best=rec.shortlist.map(f["best"]),
        standard_variant=rec.shortlist.map(f["variant"]),
    )
    for r in rec[rec.variant != rec.standard_variant].itertuples():
        print(f"[{NAME}] NOTE: {r.name}: max lie ran {r.variant}, Figure F draws {r.standard_variant}")
    ref = summarise(load())[1].loc[(N, "beta0")]
    orc = pd.DataFrame(
        [
            dict(figure=NAME, key=k, n=N, value=ref[c])
            for k, c in zip(S.REF_LINES, ["oracle", "random", S.NAME["midian_ref"]])
        ]
    )
    return rec, orc


def render(rec, orc, out):
    fig, ax = S.figure("appendix")
    order = [k for k in S.SHORTLIST_ORDER if k in set(rec.shortlist)]
    w = 0.6
    for i, key in enumerate(order):
        r = rec[rec.shortlist == key].iloc[0]
        S.bar(ax, i, r["mean"], w, key, cartel=True)
        S.best_dot(ax, i, r["best"])
        if pd.notna(r["standard_mean"]):
            S.std_tick(ax, i - w / 2, i + w / 2, r["standard_mean"])
    line = dict(zip(orc.key, orc.value))
    S.oracle(ax, -0.5, len(order) - 0.5, line["oracle"])
    S.ref(ax, -0.5, len(order) - 0.5, line["random_line"], "random")
    S.ref(ax, -0.5, len(order) - 0.5, line["midian_ref"])
    ax.set_xticks(range(len(order)))
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_xticklabels([textwrap.fill(S.SHORTLIST_BODY[k], 14) for k in order], linespacing=0.95)
    S.finish(ax, ylabel=S.AXIS["success"], ylim=(0.2, 0.95))
    S.legend(ax, S.REF_LINES + ["best_fw_dot", "standard_lie"])
    S.save(fig, NAME, out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--out", help="output directory (default: $RTE_FIG_OUT or figures/paper); must hold F_shortlists_1e5.csv"
    )
    p.add_argument("--from-csv", action="store_true", help="redraw from results/aggregates/figures only")
    a = p.parse_args(argv)
    out = fig_out(a.out)
    if a.from_csv:
        rec, orc = republish(NAME, out, refs=True)
    else:
        rec, orc = compute(out)
        publish(NAME, out, rec, refs=orc)
    render(rec, orc, out)


if __name__ == "__main__":
    main()
