"""Per-condition bars of every framework under every shortlist source, against the oracle and MIDIAN-VA.
    python scripts/shortlist_figs.py [live routereval]   -> figures/shortlist/<family>__n<n>__<dist>__<regime>.{png,pdf}
                                                           + figures/shortlist/<family>.csv + figures/shortlist/INDEX.md
One figure per condition (family, n, population shape, liar regime); one panel and one row, always. Left of the gap:
the oracle (dotted bar) and MIDIAN-VA routing the whole population (solid bar). Right of it: one group per framework,
one bar per shortlist source -- the pre-registered hashed TF-IDF, dedup (one agent per distinct description), MiniLM
cosine, the MIDIAN-V leaf cohort and the MIDIAN-VA leaf cohort (the "MIDIAN-VA augmented" framework). Error bars are
the 95% seed bootstrap. Colours are shared with the other figure sets through extra_figs.COLOR and persisted here.
Do-not-add list (extra_figs.excluded) applies; the 14B Magentic-One supervisor arm is excluded (it answers M3, not this)."""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extra_figs import COLOR as _COLOR, ci as _ci, excluded
from fw_variant_numbers import VARIANTS, select, regime
from paper_figs import ABBR

RTE_DATA = os.environ.get("RTE_DATA", "/scratch/rte"); R = f"{RTE_DATA}/results"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "shortlist"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6, "axes.linewidth": 0.6, "figure.constrained_layout.use": True})

# shortlist source -> (display name, colour); drawn left to right in this order wherever the source has data
SOURCES = [("tfidf", "hashed TF-IDF (pre-registered)", "#b0b0b0"), ("dedup", "dedup (one per distinct text)", "#8c6d31"),
           ("embed", "MiniLM cosine", "#1f77b4"), ("v_cohort", "MIDIAN-V leaf cohort", "#ff7f0e"), ("va_cohort", "MIDIAN-VA leaf cohort", "#117a3d")]
SRC_NAME = {k: v for k, v, _ in SOURCES}; SRC_COLOR = {k: c for k, _, c in SOURCES}
# RouterEval carries three of the five (no dedup-only grid, no V-cohort grid)
RE_VARIANTS = {"tfidf": [("fw_routereval_small", "plain"), ("fw_routereval_1k", "plain"), ("fw_routereval_5k", "plain")],
               "embed": [("fw_routereval_small_em", "embed"), ("fw_routereval_1k_em", "embed"), ("fw_routereval_5k_em", "embed")],
               "va_cohort": [("fw_routereval_small_va", "midian_va"), ("fw_routereval_1k_va", "midian_va"), ("fw_routereval_5k_va", "midian_va")]}
# where the oracle and MIDIAN-VA for a condition come from: every grid at that n is pooled and matched on (dist, regime, seed)
REF_GRIDS = {"live": {100: ["fw_live_n100", "learned_n100", "live_core_n100", "fw_live_n100_lowskill"],
                      1000: ["fw_live_n1000", "live_f1_n1000", "variants_f1", "learned_f1", "fw_live_n1000_lowskill"],
                      10000: ["learned_n10k", "live_n10k_v2", "fw_live_n10k_cartel", "learned_n10k_beta01"],
                      100000: ["live_n100k", "live_n100k_fill", "live_n100k_beta01"]},
             "routereval": {10: ["routereval_mmlu"], 100: ["routereval_mmlu"], 1000: ["routereval_mmlu"], 5000: ["routereval_mmlu5k"]}}
REGIME_NAME = {"beta0": "honest (β = 0)", "beta01_random": "β = 0.1, random liars", "beta025_random": "β = 0.25, random liars",
               "beta025_cartel": "β = 0.25, low-skill cartel", "beta05_random": "β = 0.5, random liars", "cartel": "β = 0.5, low-skill cartel"}
_mem = {}


def rows(grid):
    """rows.csv only -- every framework grid writes one, and globbing rows.d over NFS costs minutes for no extra rows."""
    if grid not in _mem:
        p = f"{R}/{grid}/rows.csv"
        df = pd.read_csv(p, low_memory=False) if os.path.exists(p) else pd.DataFrame()
        if not df.empty:
            df["params"] = df.params.astype(str)
            if "rid" in df: df = df.drop_duplicates("rid")
        _mem[grid] = df
    return _mem[grid]


def tagged(df):
    return df.assign(regime=[regime(b, l) for b, l in zip(df.beta, df.liar_select)])


def collect(family):
    """(n, dist, regime) -> {framework -> {source -> per-seed Series}}, plus the oracle / MIDIAN-VA per-seed Series."""
    variants = VARIANTS if family == "live" else RE_VARIANTS
    cells: dict[tuple, dict] = {}
    for src, grids in variants.items():
        for grid, kind in grids:
            df = rows(grid)
            if df.empty: continue
            fw = select(df, kind)
            if fw.empty: continue
            for (n, dist, reg), q in tagged(fw).groupby(["n", "dist", "regime"]):
                if reg == "beta0" and q.liar_select.nunique() > 1: q = q[q.liar_select == "random"]   # liar-free: counted once
                per = q.groupby(["method", "seed"]).success.mean().unstack(0)                        # seeds x frameworks
                cell = cells.setdefault((int(n), str(dist), reg), {"fw": {}, "ref": {}})
                for m in per:
                    if excluded(m): continue
                    cell["fw"].setdefault(m, {})[src] = per[m].dropna()
    for (n, dist, reg), cell in cells.items():
        for grid in REF_GRIDS[family].get(n, []):
            df = rows(grid)
            if df.empty: continue
            q = tagged(df[df.method.isin(["oracle", "midian_va"]) & (df.n == n) & (df.dist.astype(str) == dist)])
            q = q[q.regime == reg]
            if reg == "beta0" and q.liar_select.nunique() > 1: q = q[q.liar_select == "random"]
            for m, g in q.groupby("method"):
                s = g.groupby("seed").success.mean().dropna()
                if len(s) > len(cell["ref"].get(m, [])): cell["ref"][m] = s      # the grid with the most seeds wins
    return cells


def draw(family, key, cell, csv_rows):
    n, dist, reg = key
    fws = sorted(cell["fw"], key=lambda m: ABBR.get(m, m))
    srcs = [s for s, _, _ in SOURCES if any(s in cell["fw"][m] for m in fws)]
    if not fws or not srcs: return None
    ref = [(m, lbl, tick) for m, lbl, tick in (("oracle", "oracle", "oracle"),
                                               ("midian_va", "MIDIAN-VA (whole population)", "MIDIAN-VA")) if m in cell["ref"]]

    w = 0.8 / len(srcs)                                              # one group per framework, one bar per source
    xref = list(range(len(ref))); x0 = len(ref) + 0.6                # a gap between the reference bars and the groups
    xfw = [x0 + i for i in range(len(fws))]
    fig, ax = plt.subplots(figsize=(max(6.4, 0.62 * (len(ref) + len(fws)) + 2.2), 2.9))

    for xi, (m, lbl, _) in zip(xref, ref):
        s = cell["ref"][m]; lo, hi = _ci(s); v = float(s.mean()); c = _COLOR.get(m, "#555")
        if m == "oracle":                                            # dotted bar
            ax.bar(xi, v, 0.62, facecolor=(*matplotlib.colors.to_rgb(c), 0.22), edgecolor=c, linewidth=1.1, linestyle=":", hatch="....", label=lbl, zorder=2)
        else:                                                        # solid bar
            ax.bar(xi, v, 0.62, color=c, edgecolor="black", linewidth=0.4, label=lbl, zorder=2)
        ax.errorbar(xi, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor="#222", elinewidth=0.6, capsize=1.6, zorder=3)
        csv_rows.append([family, reg, dist, n, lbl, "-", v, lo, hi, len(s)])

    for si, src in enumerate(srcs):
        off = (si - (len(srcs) - 1) / 2) * w
        xs, ys, el, eu = [], [], [], []
        for m, xm in zip(fws, xfw):
            s = cell["fw"][m].get(src)
            if s is None or not len(s): continue
            lo, hi = _ci(s); v = float(s.mean())
            xs.append(xm + off); ys.append(v); el.append(v - lo); eu.append(hi - v)
            csv_rows.append([family, reg, dist, n, ABBR.get(m, m), src, v, lo, hi, len(s)])
        if not xs: continue
        ax.bar(xs, ys, w * 0.92, color=SRC_COLOR[src], edgecolor="black", linewidth=0.3, label=SRC_NAME[src], zorder=2)
        ax.errorbar(xs, ys, yerr=[el, eu], fmt="none", ecolor="#222", elinewidth=0.5, capsize=1.2, zorder=3)

    if "oracle" in cell["ref"]:                                      # dotted guide at the oracle, so every group reads against it
        ax.axhline(float(cell["ref"]["oracle"].mean()), color=_COLOR.get("oracle", "#7f8c8d"), ls=":", lw=0.9, zorder=1)
    ax.set_xticks(xref + xfw); ax.set_xticklabels([t for _, _, t in ref] + [ABBR.get(m, m) for m in fws], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.7, xfw[-1] + 0.7); ax.set_ylabel("success"); ax.grid(axis="y", lw=0.3, alpha=0.35, zorder=0); ax.set_axisbelow(True)
    seeds = max((len(s) for m in fws for s in cell["fw"][m].values()), default=0)
    ax.set_title(f"{family} · n = {n:,} · {dist} · {REGIME_NAME.get(reg, reg)}   ({len(fws)} frameworks × {len(srcs)} shortlists, {seeds} seeds)")
    ax.legend(ncol=min(4, len(srcs) + len(ref)), frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.06))
    stem = f"{OUT}/{family}__n{n}__{dist}__{reg}"
    fig.savefig(f"{stem}.pdf"); fig.savefig(f"{stem}.png", dpi=200); plt.close(fig)
    return os.path.basename(stem)


def main(families):
    index = []
    for family in families:
        cells = collect(family); csv_rows = []
        for key in sorted(cells):
            got = draw(family, key, cells[key], csv_rows)
            if got: index.append((family, got)); print(f"[{family}] {got}")
        pd.DataFrame(csv_rows, columns=["family", "regime", "dist", "n", "arm", "shortlist", "mean", "ci_lo", "ci_hi", "seeds"]).to_csv(f"{OUT}/{family}.csv", index=False)
    with open(f"{OUT}/INDEX.md", "w") as f:
        f.write("# figures/shortlist -- every framework under every shortlist source, per condition\n\n"
                "One figure per (family, n, population shape, liar regime); one panel and one row. The oracle is the dotted bar (and a dotted\n"
                "guide across the panel); MIDIAN-VA routing the whole population is the solid bar; each framework group carries one bar per\n"
                "shortlist source. Error bars are the 95% seed bootstrap. Do-not-add arms (extra_figs.excluded) are never drawn.\n\n"
                "Shortlist sources: " + "; ".join(f"`{k}` = {v}" for k, v, _ in SOURCES) + ".\n\n")
        for family in families:
            f.write(f"\n## {family}\n")
            for fam, stem in index:
                if fam == family: f.write(f"- `{stem}.png`\n")
    print(f"\n{len(index)} figures -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["live", "routereval"])
