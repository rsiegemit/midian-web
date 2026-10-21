"""Appendix panel I_max_lie: the strongest declared lie (lie_mode max: every cartel member claims perfect skill in every
family; grids lie_max_* / lie_max_fw_*, live specialist, beta = 0.5 low-skill-first cartel, b = 3, seeds 1-3) against
the standard lie (true skill + 0.4, clipped) of Figure 1. Bars = max lie (hatched: cartel), +/- 1 s.e. over seeds; a
black tick on each bar = the same arm under the standard lie (<out>/A_live_allb.csv, b = 3; frameworks:
results/aggregates/shortlist/live.csv, declared top-k). Pooled arms under the max lie are the best of the pool's arms
that ran there by mean (a choice that favours the rival): learned = online kNN, bandit = warm-start (n0 = 5 or 0.5),
framework = the best of ten on the declared top-k.
    python scripts/figures/lie_max_fig.py [--out DIR] [--from-csv]
      -> <out>/I_max_lie.{png,pdf,csv} + results/aggregates/figures/I_max_lie.csv and its oracle lines in refs.csv;
         --from-csv redraws from those two alone (no $RTE_DATA)."""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from scripts.figures.lib import AGG, NEW, SEEDS_4_10, fig_out, publish, republish  # noqa: E402
from scripts.figures.lib import figspec as S  # noqa: E402
from scripts.figures.lib.grids import SIZE  # noqa: E402
from scripts.figures.lib.rows import pooled, read_rows  # noqa: E402

NAME = "I_max_lie" + NEW
NS = [n for n in SIZE if n >= 1000]
KEYS = [k for k in S.ORDER if k != "oracle"]
POOL = {
    "midian": "midian",
    "midian_wo_defenses": "midian_wo_defenses",
    "flat_probe_argmax_online": "flat_probe_argmax_online",
    "knn_router_online": "best_learned",
    "warm_start_bandit": "best_bandit",
    "warm_start_bandit_n05": "best_bandit",
    "declared_argmax": "declared_argmax",
    "random": "random",
    "oracle": "oracle",
}
LABEL = {
    ("midian", "{}"): "midian",
    ("midian", '{"audit":false,"verify":false}'): "midian_wo_defenses",
    ("flat_probe_argmax", '{"online":true}'): "flat_probe_argmax_online",
    ("knn_router", '{"online":true}'): "knn_router_online",
    ("warm_start_bandit", "{}"): "warm_start_bandit",
    ("warm_start_bandit", '{"n0":0.5}'): "warm_start_bandit_n05",
}


def rows(g):
    x = read_rows(g, csv_first=True).drop_duplicates("rid")
    x["params"] = x.params.fillna("{}").astype(str)
    return x[x.b == 3]


def max_lie(n):
    # 10-seed build: + the seed mirrors, and lie_max_arms_<n> for the claim readers' seeds 4-10 (the same row ids:
    # configs/grids/seeds_to_10.yaml); a row in both is read once
    gs = [f"lie_max_{SIZE[n]}"] + ([f"lie_max_arms_{SIZE[n]}"] if SEEDS_4_10 else [])
    x = pooled(rows, gs, rid=True)
    x["label"] = [LABEL.get((m, p), m) for m, p in zip(x.method, x.params)]
    per = x.groupby(["label", "seed"]).success.mean().unstack("label")
    fw = rows(f"lie_max_fw_{SIZE[n]}")
    fw = fw[fw.method.str.startswith("fw_")].groupby(["method", "seed"]).success.mean().unstack("method")
    out = {}
    for lab, key in POOL.items():
        if lab in per:
            out.setdefault(key, []).append(per[lab].dropna())
    if len(fw.columns):
        out["best_framework"] = [fw[c].dropna() for c in fw.columns if fw[c].count() == fw.count().max()]
    best = {k: max(v, key=lambda s: s.mean()) for k, v in out.items()}  # pooled: best mean (favours the rival)
    return {k: (s.mean(), s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan, len(s)) for k, s in best.items()}


def standard(n, out):
    a = pd.read_csv(f"{out}/A_live_allb{NEW}.csv")
    a = a[a.group.str.replace(",", "").str.endswith(f"= {n}") & (a.regime != "honest") & a.b.isin(["3", "-"])]
    std = dict(zip(a.key, a.value))
    sl = pd.read_csv(f"{AGG}/shortlist{NEW}/live.csv")
    sl = sl[
        (sl.n == n)
        & (sl.shortlist == "declared")
        & (sl.regime == "cartel")
        & (sl.dist == "specialist")
        & sl.arm.str.startswith("fw_")
    ]
    if len(sl):
        std["best_framework"] = sl["mean"].max()
    return std


def compute(out):
    """The figure CSV (one row per bar) and the oracle per n."""
    rec, orc = [], []
    for n in NS:
        m, st = max_lie(n), standard(n, out)
        rec += [
            dict(n=n, arm=S.NAME[k], key=k, max_lie=m[k][0], se=m[k][1], seeds=m[k][2], standard_lie=st.get(k))
            for k in KEYS
            if k in m
        ]
        if "oracle" in m:
            orc.append(dict(figure=NAME, key="oracle", n=n, value=m["oracle"][0]))
    return pd.DataFrame(rec), pd.DataFrame(orc)


def render(rec, orc, out):
    """One group per n; no whisker where a single seed leaves the s.e. undefined."""
    rec = rec.assign(lo=rec.max_lie - rec.se, hi=rec.max_lie + rec.se)
    o = orc.groupby("n").value.first()
    groups = [(t, rec[rec.n == n], o.get(n)) for t, n in zip(S.pow10_ticks(NS, "n"), NS)]
    S.max_lie(groups, KEYS, NAME, out, S.AXIS["success"], (0.0, 0.95))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", help="output directory (default: $RTE_FIG_OUT or figures/paper); must hold A_live_allb.csv")
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
