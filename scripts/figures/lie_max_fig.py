"""Appendix panel I_max_lie: the strongest declared lie (lie_mode max: every cartel member claims perfect skill in every
family; grids lie_max_* / lie_max_fw_*, live specialist, beta = 0.5 low-skill-first cartel, b = 3, seeds 1-3) against the
standard lie (true skill + 0.4, clipped) of Figure 1. Bars = max lie (hatched: cartel), +/- 1 s.e. over seeds; a black tick
on each bar = the same arm under the standard lie (<out>/A_live_allb.csv, b = 3; frameworks:
results/aggregates/shortlist/live.csv, declared top-k). Pooled arms under the max lie are the best of the pool's arms that
ran there by mean (a choice that favours the rival): learned = online kNN, bandit = warm-start (n0 = 1 or 0.5),
framework = the best of ten on the declared top-k.
    python scripts/figures/lie_max_fig.py [--out DIR] [--from-csv]
      -> <out>/I_max_lie.{png,pdf,csv} + results/aggregates/figures/I_max_lie.csv and its oracle lines in refs.csv;
         --from-csv redraws from those two alone (no $RTE_DATA)."""
import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from scripts.figures.lib import AGG, fig_out                                                                   # noqa: E402
from scripts.figures.lib import figspec as S                                                                   # noqa: E402
from scripts.figures.lib.grids import SIZE                                                                     # noqa: E402
from scripts.figures.lib.rows import read_rows                                                                 # noqa: E402

NAME, DRAW = "I_max_lie", f"{AGG}/figures"
NS = [n for n in SIZE if n >= 1000]
KEYS = [k for k in S.ORDER if k != "oracle"]
POOL = {"midian": "midian", "midian_wo_defenses": "midian_wo_defenses", "flat_probe_argmax_online": "flat_probe_argmax_online",
        "knn_router_online": "best_learned", "warm_start_bandit": "best_bandit", "warm_start_bandit_n05": "best_bandit",
        "declared_argmax": "declared_argmax", "random": "random", "oracle": "oracle"}
LABEL = {("midian", "{}"): "midian", ("midian", '{"audit":false,"verify":false}'): "midian_wo_defenses",
         ("flat_probe_argmax", '{"online":true}'): "flat_probe_argmax_online", ("knn_router", '{"online":true}'): "knn_router_online",
         ("warm_start_bandit", "{}"): "warm_start_bandit", ("warm_start_bandit", '{"n0":0.5}'): "warm_start_bandit_n05"}


def rows(g):
    x = read_rows(g, csv_first=True).drop_duplicates("rid")
    x["params"] = x.params.fillna("{}").astype(str)
    return x[x.b == 3]


def max_lie(n):
    x = rows(f"lie_max_{SIZE[n]}")
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
    best = {k: max(v, key=lambda s: s.mean()) for k, v in out.items()}   # pooled: best mean (favours the rival)
    return {k: (s.mean(), s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan, len(s)) for k, s in best.items()}


def standard(n, out):
    a = pd.read_csv(f"{out}/A_live_allb.csv")
    a = a[a.group.str.replace(",", "").str.endswith(f"= {n}") & (a.regime != "honest") & a.b.isin(["3", "-"])]
    std = dict(zip(a.key, a.value))
    sl = pd.read_csv(f"{AGG}/shortlist/live.csv")
    sl = sl[(sl.n == n) & (sl.shortlist == "declared") & (sl.regime == "cartel") & (sl.dist == "specialist") & sl.arm.str.startswith("fw_")]
    if len(sl):
        std["best_framework"] = sl["mean"].max()
    return std


def compute(out):
    """The figure CSV (one row per bar) and the oracle per n."""
    rec, orc = [], []
    for n in NS:
        m, st = max_lie(n), standard(n, out)
        rec += [dict(n=n, arm=S.NAME[k], key=k, max_lie=m[k][0], se=m[k][1], seeds=m[k][2], standard_lie=st.get(k)) for k in KEYS if k in m]
        if "oracle" in m:
            orc.append(dict(figure=NAME, key="oracle", n=n, value=m["oracle"][0]))
    return pd.DataFrame(rec), pd.DataFrame(orc)


def render(rec, orc, out):
    fig, ax = S.figure("appendix")
    w = 0.86 / len(KEYS)
    for i, n in enumerate(NS):
        for r in rec[rec.n == n].itertuples():
            x = i + (KEYS.index(r.key) - (len(KEYS) - 1) / 2) * w
            S.bar(ax, x, r.max_lie, w * 0.95, r.key, cartel=True)
            if r.se == r.se:
                S.whisker(ax, x, r.max_lie, r.max_lie - r.se, r.max_lie + r.se)
            if pd.notna(r.standard_lie):
                ax.hlines(r.standard_lie, x - w * 0.45, x + w * 0.45, colors="black", lw=1.2, zorder=8)
        o = orc[orc.n == n]
        if len(o):
            S.oracle(ax, i - 0.46, i + 0.46, o.value.iloc[0])
    ax.set_xticks(range(len(NS))); ax.set_xticklabels(S.pow10_ticks(NS, "n")); ax.set_xlim(-0.5, len(NS) - 0.5)
    S.finish(ax, ylabel=S.AXIS["success"], ylim=(0.0, 0.95))
    from matplotlib.lines import Line2D
    S.legend(ax, ["oracle"] + KEYS, ncol=5, columnspacing=0.5, handlelength=1.0,
             extra=([Line2D([], [], color="black", lw=1.2)], ["standard lie"]))
    S.save(fig, NAME, out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", help="output directory (default: $RTE_FIG_OUT or figures/paper); must hold A_live_allb.csv")
    p.add_argument("--from-csv", action="store_true", help="redraw from results/aggregates/figures only")
    a = p.parse_args(argv)
    out, refs_csv = fig_out(a.out), f"{DRAW}/refs.csv"
    if a.from_csv:
        rec, orc = pd.read_csv(f"{DRAW}/{NAME}.csv"), pd.read_csv(refs_csv)
        orc = orc[orc.figure == NAME]
        shutil.copyfile(f"{DRAW}/{NAME}.csv", f"{out}/{NAME}.csv")
    else:
        rec, orc = compute(out)
        os.makedirs(DRAW, exist_ok=True)
        rec.to_csv(f"{out}/{NAME}.csv", index=False); rec.to_csv(f"{DRAW}/{NAME}.csv", index=False)
        old = pd.read_csv(refs_csv) if os.path.exists(refs_csv) else pd.DataFrame(columns=orc.columns)
        pd.concat([old[old.figure != NAME], orc], ignore_index=True).to_csv(refs_csv, index=False)
    render(rec, orc, out)


if __name__ == "__main__":
    main()
