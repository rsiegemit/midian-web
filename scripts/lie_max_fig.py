"""Appendix panel I_max_lie: the strongest declared lie (lie_mode max: every cartel member claims perfect skill in every
family; grids lie_max_* / lie_max_fw_*, live specialist, beta = 0.5 low-skill-first cartel, b = 3, seeds 1-3) against the
standard lie (true skill + 0.4, clipped) of Figure 1. Bars = max lie (hatched: cartel), +/- 1 s.e. over seeds; a black tick
on each bar = the same arm under the standard lie (A_live_allb, b = 3; frameworks: figures/shortlist/live.csv, declared
top-k). Pooled arms under the max lie are the best of the pool's arms that ran there by mean (a choice that favours the
rival): learned = online kNN, bandit = warm-start (n0 = 1 or 0.5), framework = the best of ten on the declared top-k.
    RTE_DATA=... PYTHONPATH=. python scripts/lie_max_fig.py"""
import glob, json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/scripts")
import figspec as S
from rte.methods import keys
R, OUT = os.environ["RTE_DATA"] + "/results/", f"{ROOT}/figures/condensed_sample"
NS = {1000: "n1000", 10000: "n10k", 100000: "n100k"}
POOL = {"midian": "midian", "midian_wo_defenses": "midian_wo_defenses", "flat_probe_argmax_online": "flat_probe_argmax_online",
        "knn_router_online": "best_learned", "warm_start_bandit": "best_bandit", "warm_start_bandit_n05": "best_bandit",
        "declared_argmax": "declared_argmax", "random": "random", "oracle": "oracle"}
LABEL = {("midian", "{}"): "midian", ("midian", '{"audit":false,"verify":false}'): "midian_wo_defenses",
         ("flat_probe_argmax", '{"online":true}'): "flat_probe_argmax_online", ("knn_router", '{"online":true}'): "knn_router_online",
         ("warm_start_bandit", "{}"): "warm_start_bandit", ("warm_start_bandit", '{"n0":0.5}'): "warm_start_bandit_n05"}


def rows(g):
    d = [pd.read_csv(R + g + "/rows.csv", low_memory=False)] if os.path.exists(R + g + "/rows.csv") else []
    fs = glob.glob(R + g + "/rows.d/*.json")
    if fs: d.append(pd.DataFrame([{**json.load(open(f)), "rid": os.path.basename(f)[:-5]} for f in fs]))
    x = keys.normalize(pd.concat(d, ignore_index=True), R + g).drop_duplicates("rid")
    x["params"] = x.params.fillna("{}").astype(str)
    return x[x.b == 3]


def max_lie(n):
    x = rows(f"lie_max_{NS[n]}"); x["label"] = [LABEL.get((m, p), m) for m, p in zip(x.method, x.params)]
    per = x.groupby(["label", "seed"]).success.mean().unstack("label")
    fw = rows(f"lie_max_fw_{NS[n]}"); fw = fw[fw.method.str.startswith("fw_")].groupby(["method", "seed"]).success.mean().unstack("method")
    out = {}
    for lab, key in POOL.items():
        if lab in per: out.setdefault(key, []).append(per[lab].dropna())
    if len(fw.columns): out["best_framework"] = [fw[c].dropna() for c in fw.columns if fw[c].count() == fw.count().max()]
    best = {k: max(v, key=lambda s: s.mean()) for k, v in out.items()}   # pooled: best mean (favours the rival)
    return {k: (s.mean(), s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan, len(s)) for k, s in best.items()}


def standard(n):
    a = pd.read_csv(f"{OUT}/A_live_allb.csv")
    a = a[a.group.str.replace(",", "").str.endswith(f"= {n}") & (a.regime != "honest") & a.b.isin(["3", "-"])]
    std = dict(zip(a.key, a.value))
    sl = pd.read_csv(f"{ROOT}/figures/shortlist/live.csv")
    sl = sl[(sl.n == n) & (sl.shortlist == "declared") & (sl.regime == "cartel") & (sl.dist == "specialist") & sl.arm.str.startswith("fw_")]
    if len(sl): std["best_framework"] = sl["mean"].max()
    return std


if __name__ == "__main__":
    keys_ = [k for k in S.ORDER if k != "oracle"]; ns = list(NS)
    fig, ax = S.figure("appendix"); w = 0.86 / len(keys_); rec = []
    for i, n in enumerate(ns):
        m, st = max_lie(n), standard(n)
        for j, k in enumerate(keys_):
            if k not in m: continue
            x = i + (j - (len(keys_) - 1) / 2) * w; mean, se, seeds = m[k]
            S.bar(ax, x, mean, w * 0.95, k, cartel=True)
            if se == se: S.whisker(ax, x, mean, mean - se, mean + se)
            if k in st: ax.hlines(st[k], x - w * 0.45, x + w * 0.45, colors="black", lw=1.2, zorder=8)
            rec.append(dict(n=n, arm=S.NAME[k], key=k, max_lie=mean, se=se, seeds=seeds, standard_lie=st.get(k)))
        if "oracle" in m: S.oracle(ax, i - 0.46, i + 0.46, m["oracle"][0])
    ax.set_xticks(range(len(ns))); ax.set_xticklabels(S.pow10_ticks(ns, "n")); ax.set_xlim(-0.5, len(ns) - 0.5)
    S.finish(ax, ylabel=S.AXIS["success"], ylim=(0.0, 0.95))
    from matplotlib.lines import Line2D
    S.legend(ax, ["oracle"] + keys_, ncol=5, columnspacing=0.5, handlelength=1.0,
             extra=([Line2D([], [], color="black", lw=1.2)], ["standard lie"]))
    S.save(fig, "I_max_lie", OUT); pd.DataFrame(rec).to_csv(f"{OUT}/I_max_lie.csv", index=False)
