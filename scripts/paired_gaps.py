"""Figure C (condensed sample): MIDIAN minus each of a FIXED set of rivals, seed-PAIRED, in every condition.
    python scripts/paired_gaps.py      -> figures/condensed_sample/C_paired_gaps.{png,pdf,csv}
Replaces the first C (best rival picked per condition, unpaired CI): picking the max of ~20 noisy arms per condition is
biased toward the rivals, and adding two independent CIs ignores that both arms ran on the same seeds. Here each dot is
one condition (family x n x shape x liar regime) and one rival; its bar is the 95% bootstrap CI of the per-seed difference
MIDIAN - rival over the seeds both arms share (per-seed values average the shapes only where the family pools them).
Rivals: the strongest arm of each class the paper compares against -- probing, bandits, learned routers, the declared
channel, claim verification -- plus MIDIAN w/o defenses. b = 3 only. Honest cells are hollow, liar cells filled.
Sequential halving is computed and written to the csv but not drawn while extra_figs.HIDE_HALVING is on."""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rte.analyze import RTE_DATA, prepare
from extra_figs import ci, excluded
from bar_figs import rows, REGIMES, LIVE_GRIDS

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "condensed_sample"); os.makedirs(OUT, exist_ok=True)
RIVALS = [("flat_probe_argmax_online", "flat probe argmax (online)"), ("warm_start_bandit", "warm-start bandit"), ("linucb_honest", "LinUCB"),
          ("knn_router_online", "KNN router (online)"), ("mlp_router", "MLP router"), ("declared_argmax", "declared argmax"),
          ("verify_on_claim", "verify on claim"), ("midian_wo_defenses", "MIDIAN w/o defenses"), ("sequential_halving_peer", "seq. halving (peer)")]
FAM_COL = {"live": "#c0392b", "bernoulli": "#2980b9", "replay": "#8e44ad", "routereval": "#16a085", "llmrouterbench": "#d35400"}
COLS = ["label", "n", "dist", "beta", "liar_select", "seed", "success", "b"]


def per_seed(df, family, group_of, pool_shapes):
    """long (family, group, n, regime, label, unit, success): one value per seed (and per shape unless the family pools shapes)."""
    df = df[[c for c in COLS if c in df]]
    if "b" in df: df = df[df.b.fillna(3).astype(int) == 3]
    out = []
    for reg, beta, liar, _ in REGIMES:
        q = df[np.isclose(df.beta, beta) & (df.liar_select == liar)]
        if beta == 0 and q.empty: q = df[np.isclose(df.beta, 0)]
        if q.empty: continue
        keys = ["n", "label", "seed"] + ([] if pool_shapes else ["dist"])
        g = q.groupby(keys).success.mean().reset_index()
        g["group"] = group_of(g); g["unit"] = g.seed.astype(str) + ("" if pool_shapes else "|" + g.dist.astype(str))
        g["family"], g["regime"] = family, reg; out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def gather():
    parts = []
    for n, grids in LIVE_GRIDS.items():
        df = pd.concat([rows(g) for g in grids], ignore_index=True)
        if "declared_source" in df: df = df[df.declared_source == "self_described"]
        parts.append(per_seed(df[df.n == n], "live", lambda g: g.dist, False))
    small = rows("routereval_mmlu"); big = rows("routereval_mmlu5k")
    parts.append(per_seed(small, "routereval", lambda g: g.dist, False))
    parts.append(per_seed(big, "routereval", lambda g: "leaderboard", True))
    parts.append(per_seed(rows("llmrouterbench_pool"), "llmrouterbench", lambda g: "20 models", True))
    for fam, grid in (("bernoulli", "bernoulli_scale_v5"), ("replay", "replay_scale_v5")):
        df = prepare(pd.read_csv(f"{RTE_DATA}/results/{grid}/rows.csv", low_memory=False))
        parts.append(per_seed(df, fam, lambda g: "shapes pooled", True))
    return pd.concat(parts, ignore_index=True)


def gaps(long):
    rec = []
    for (fam, grp, n, reg), q in long.groupby(["family", "group", "n", "regime"]):
        piv = q.pivot_table(index="unit", columns="label", values="success")
        if "midian" not in piv: continue
        for r, name in RIVALS:
            if r not in piv: continue
            d = (piv["midian"] - piv[r]).dropna()
            if len(d) < 2: continue
            lo, hi = ci(d.values)
            rec.append(dict(family=fam, group=grp, n=int(n), regime=reg, rival=r, rival_name=name, gap=float(d.mean()), ci_lo=lo, ci_hi=hi,
                            units=len(d), result="win" if lo > 0 else "loss" if hi < 0 else "tie"))
    return pd.DataFrame(rec)


def draw(t):
    shown = [(r, nm) for r, nm in RIVALS if r in set(t.rival) and not excluded(r)]
    fig, ax = plt.subplots(figsize=(7.2, 3.0)); rng = np.random.default_rng(0)
    for i, (r, nm) in enumerate(shown):
        q = t[t.rival == r]
        for fam, g in q.groupby("family"):
            x = i + rng.uniform(-0.32, 0.32, len(g)); honest = (g.regime == "beta0").values
            for sel, face in ((honest, "white"), (~honest, FAM_COL[fam])):
                if not sel.any(): continue
                ax.errorbar(x[sel], g.gap.values[sel], yerr=[g.gap.values[sel] - g.ci_lo.values[sel], g.ci_hi.values[sel] - g.gap.values[sel]],
                            fmt="o", ms=2.6, color=FAM_COL[fam], mfc=face, mec=FAM_COL[fam], elinewidth=0.35, alpha=0.85)
        w, l, tie = (q.result == "win").sum(), (q.result == "loss").sum(), (q.result == "tie").sum()
        ax.text(i, 1.02, f"{w}/{tie}/{l}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.5)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(len(shown))); ax.set_xticklabels([nm for _, nm in shown], rotation=20, ha="right", rotation_mode="anchor")
    ax.set_ylabel("MIDIAN − rival (paired, 95% CI)"); ax.grid(axis="y", lw=0.3, alpha=0.35); ax.set_axisbelow(True)
    for fam, c in FAM_COL.items(): ax.plot([], [], "o", color=c, ms=3, label=fam)
    ax.plot([], [], "o", mfc="white", mec="black", ms=3, label="hollow = honest"); ax.plot([], [], "o", color="black", ms=3, label="filled = liars")
    ax.set_title("C  MIDIAN vs each fixed rival, every condition, seed-paired (numbers: wins / ties / losses)", pad=12)
    ax.legend(ncol=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.08))
    fig.savefig(f"{OUT}/C_paired_gaps.png", dpi=250); fig.savefig(f"{OUT}/C_paired_gaps.pdf"); plt.close(fig)


if __name__ == "__main__":
    t = gaps(gather()); t.to_csv(f"{OUT}/C_paired_gaps.csv", index=False); draw(t)
    print(t.groupby(["rival", "result"]).size().unstack(fill_value=0).to_string())
