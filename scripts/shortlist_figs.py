"""Per-condition bars of every framework under every shortlist source, against the oracle and MIDIAN.
    python scripts/shortlist_figs.py [live routereval]   -> figures/shortlist/<family>__n<n>__<dist>__<regime>.{png,pdf}
                                                           + figures/shortlist/<family>.csv + figures/shortlist/INDEX.md
One figure per condition (family, n, population shape, liar regime); one panel and one row, always. One group per
framework, one bar per shortlist source that has data in the condition (SOURCES: the pre-registered hashed TF-IDF,
MiniLM, BM25, Qwen3-Embedding-8B dense with and without a task instruction, BM25+dense fusion + the Qwen3
cross-encoder with and without an instruction, the declared-claim top-k and the MIDIAN leaf cohort). Not drawn:
dedup TF-IDF, fusion without the reranker, the cohort of MIDIAN w/o audits; a condition with fewer than two shortlists is skipped).
The oracle (dotted) and MIDIAN routing the whole population (solid) are horizontal lines across the panel
with their 95% seed-bootstrap band. A source is recognised from each row's params, so a new shortlist grid needs only
an entry in SOURCES. Bars whose rows still have an erratum-28 rerun outstanding carry an asterisk.
Rows come from rows.csv AND rows.d (reruns write rows.d only). b = 3 only -- never pooled with b = 1.
Do-not-add list (extra_figs.excluded) applies; MIDIAN cohorts other than r = 10, the shuffled MIDIAN cohort (a position control), the lying-text condition and the 14B
Magentic-One supervisor arm are not shortlist sources and are never drawn."""
from __future__ import annotations
import glob, json, os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extra_figs import COLOR as _COLOR, se as _ci, excluded   # whiskers +/- 1 s.e. over seeds
from fw_variant_numbers import load, regime, pending_reruns
from paper_figs import ABBR
import figspec

RTE_DATA = os.environ.get("RTE_DATA", "/scratch/rte"); R = f"{RTE_DATA}/results"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "shortlist"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6, "axes.linewidth": 0.6, "figure.constrained_layout.use": True})

# shortlist source -> (display name, colour); drawn left to right in this order (figspec.SHORTLIST_ORDER) wherever it has data
_LONG = {"tfidf": "hashed TF-IDF (pre-registered)", "bm25": "BM25", "embed": "MiniLM", "dense": "Qwen3-8B dense",
         "dense_icomp": "Qwen3-8B dense, I-competent", "dense_idemo": "Qwen3-8B dense, I-demonstrated", "sota": "fusion + reranker",
         "sota_icomp": "fusion + reranker, I-competent", "sota_idemo": "fusion + reranker, I-demonstrated",
         "declared": "declared-claim top-k", "va_cohort": "MIDIAN leaf cohort"}
SOURCES = [(k, _LONG[k], figspec.SHORTLIST_COLOR[k]) for k in figspec.SHORTLIST_ORDER]
SRC_NAME = {k: v for k, v, _ in SOURCES}; SRC_COLOR = {k: c for k, _, c in SOURCES}
# Erratum 30: H moves to the no-repeat + calibrated-claims reruns (*_norep_cal) once ALL of them have landed; until then the
# old rows stand, never a mix. H30_FW / H30_REF are those grids.
H30_FW = [f"{g}_norep_cal" for g in ("fw_routereval_small", "fw_routereval_1k", "fw_routereval_5k", "fw_routereval_small_em",
          "fw_routereval_1k_em", "fw_routereval_5k_em", "fw_routereval_small_va", "fw_routereval_1k_va", "fw_routereval_5k_va",
          "re_sl_declared_small", "re_sl_declared_1k", "re_sl_declared_5k", "re_sl_embed_small", "re_sl_embed_1k", "re_sl_embed_5k")]
H30_REF = {10: ["routereval_mmlu_norep_cal"], 100: ["routereval_mmlu_norep_cal"], 1000: ["routereval_mmlu_norep_cal"], 5000: ["routereval5k_norep_cal"]}
_h30 = []


def h30():
    """True once every H30 grid is complete (seed_tables.complete, counting framework rows too)."""
    if not _h30:
        from seed_tables import complete
        _h30.append(all(complete(g, rows) for g in H30_FW) and all(complete(g) for v in H30_REF.values() for g in v))   # framework grids are all b = 3
    return _h30[0]


GRIDS = {"live": lambda g: g.startswith("fw_live_n") and "lietext" not in g or g in ("live_n10k_v2", "live_n100k"),
         "routereval": lambda g: g.startswith(("fw_routereval_", "re_sl_")) and g.endswith("_norep_cal") == h30()}


def source(params):
    """params JSON of a framework row -> shortlist source key, or None when the row is not drawn (not a shortlist arm, or
    dropped: dedup TF-IDF, fusion without the reranker, the cohort of MIDIAN w/o audits, the shuffled MIDIAN cohort, r != 10)."""
    p = json.loads(params) if isinstance(params, str) and params.startswith("{") else {}
    if "supervisor" in p or "lie_text" in p: return None
    ret = p.get("retrieval"); ins = str(p.get("embed_instruct", ""))
    tag = "_icomp" if "competent" in ins else "_idemo" if "demonstrated" in ins else ""
    if ret is None: return None if p.get("dedup") else "tfidf"
    if ret == "midian": return "va_cohort" if p.get("r") == 10 and not p.get("shuffle") else None
    if ret == "embed": return ("dense" + tag) if "Qwen" in str(p.get("embed_model", "")) else "embed"
    if ret == "sota": return "sota" + tag
    return ret if ret in ("bm25", "declared") else None
# where the oracle and MIDIAN for a condition come from: every grid at that n is pooled and matched on (dist, regime, seed)
REF_GRIDS = {"live": {100: ["fw_live_n100", "learned_n100", "live_core_n100", "fw_live_n100_lowskill"],
                      1000: ["fw_live_n1000", "live_f1_n1000", "variants_f1", "learned_f1", "fw_live_n1000_lowskill"],
                      10000: ["learned_n10k", "live_n10k_v2", "fw_live_n10k_cartel", "learned_n10k_beta01"],
                      100000: ["live_n100k", "live_n100k_fill", "live_n100k_beta01"]},
             "routereval": {10: ["routereval_mmlu"], 100: ["routereval_mmlu"], 1000: ["routereval_mmlu"], 5000: ["routereval_mmlu5k"]}}
REGIME_NAME = {"beta0": "honest (β = 0)", "beta01_random": "β = 0.1, random liars", "beta025_random": "β = 0.25, random liars",
               "beta025_cartel": "β = 0.25, low-skill cartel", "beta05_random": "β = 0.5, random liars", "cartel": "β = 0.5, low-skill cartel"}
_mem = {}


def rows(grid):
    """rows.csv + rows.d (reruns write rows.d only), b = 3 only."""
    if grid not in _mem:
        df = load(grid)
        if not df.empty and "b" in df: df = df[df.b.fillna(3).astype(int) == 3]
        _mem[grid] = df
    return _mem[grid]


def tagged(df):
    return df.assign(regime=[regime(b, l) for b, l in zip(df.beta, df.liar_select)])


def collect(family):
    """(n, dist, regime) -> {framework -> {source -> per-seed Series}}, the oracle / MIDIAN per-seed Series, and the
    (framework, source) pairs with an erratum-28 rerun outstanding."""
    cells: dict[tuple, dict] = {}; pending = pending_reruns()
    for grid in sorted(os.path.basename(g) for g in glob.glob(f"{R}/*") if GRIDS[family](os.path.basename(g))):
        df = rows(grid)
        if df.empty: continue
        fw = df[df.method.astype(str).str.startswith("fw_")]
        fw = tagged(fw.assign(src=fw.params.map(source)).dropna(subset=["src"]))
        for (n, dist, reg), q in fw.groupby(["n", "dist", "regime"]):         # a framework whose rows were ALL quarantined still
            for m in {pm for g, pm, pd_, pr in pending if (g, pd_, pr) == (grid, str(dist), reg)} - {x for x in q.method if excluded(x)}:
                c = cells.setdefault((int(n), str(dist), reg), {"fw": {}, "ref": {}, "pending": set()})   # gets its slots, starred
                c["fw"].setdefault(m, {}); c["pending"] |= {(m, src) for src in q.src.unique()}
        for (n, dist, reg, src), q in fw.groupby(["n", "dist", "regime", "src"]):
            if reg == "beta0" and q.liar_select.nunique() > 1: q = q[q.liar_select == "random"]   # liar-free: counted once
            per = q.groupby(["method", "seed"]).success.mean().unstack(0)                        # seeds x frameworks
            cell = cells.setdefault((int(n), str(dist), reg), {"fw": {}, "ref": {}, "pending": set()})
            for m in per:
                if excluded(m): continue
                s = per[m].dropna()
                if len(s) > len(cell["fw"].setdefault(m, {}).get(src, [])): cell["fw"][m][src] = s      # the grid with the most seeds wins
                if (grid, m, str(dist), reg) in pending: cell["pending"].add((m, src))
    for (n, dist, reg), cell in cells.items():
        for grid in (H30_REF if family == "routereval" and h30() else REF_GRIDS[family]).get(n, []):
            df = rows(grid)
            if df.empty: continue
            full = (df.method == "midian") & (df.params.fillna("{}").astype(str) == "{}")          # MIDIAN, the full method
            q = tagged(df[(df.method.isin(["oracle", "random"]) | full) & (df.n == n) & (df.dist.astype(str) == dist)])
            q = q[q.regime == reg]
            if reg == "beta0" and q.liar_select.nunique() > 1: q = q[q.liar_select == "random"]
            for m, g in q.groupby("method"):
                s = g.groupby("seed").success.mean().dropna()
                if len(s) > len(cell["ref"].get(m, [])): cell["ref"][m] = s      # the grid with the most seeds wins
    return cells


def draw(family, key, cell, csv_rows):
    n, dist, reg = key
    fws = sorted(cell["fw"], key=lambda m: ABBR.get(m, m))
    srcs = [s for s, _, _ in SOURCES if any(s in cell["fw"][m] or (m, s) in cell["pending"] for m in fws)]
    if not fws or len(srcs) < 2: return None                       # a single shortlist compares nothing
    w = 0.84 / len(srcs); xfw = list(range(len(fws)))              # one group per framework, one bar per source
    fig, ax = plt.subplots(figsize=(max(6.4, (0.35 + 0.075 * len(srcs)) * len(fws) + 2.0), 2.9))

    for m, lbl, ls in (("oracle", "oracle", ":"), ("midian", figspec.NAME["midian_ref"], "-"), ("random", "random", None)):   # reference lines
        if m not in cell["ref"]: continue
        s = cell["ref"][m]; lo, hi = _ci(s); v = float(s.mean()); c = _COLOR.get(m, "#555")
        csv_rows.append([family, reg, dist, n, lbl, "-", v, lo, hi, len(s), False])
        if ls is None: continue                                    # random: in the csv (the condensed figures' line), not drawn here
        ax.axhspan(lo, hi, color=c, alpha=0.12, lw=0, zorder=1)
        ax.axhline(v, color=c, ls=ls, lw=1.4, label=f"{lbl} ({v:.3f})", zorder=4)

    stars = []
    for si, src in enumerate(srcs):
        off = (si - (len(srcs) - 1) / 2) * w
        xs, ys, el, eu = [], [], [], []
        for m, xm in zip(fws, xfw):
            s = cell["fw"][m].get(src)
            if s is None or not len(s):
                if (m, src) in cell["pending"]: stars.append((xm + off, 0.0))   # every row quarantined: an empty, starred slot
                continue
            lo, hi = _ci(s); v = float(s.mean()); star = (m, src) in cell["pending"]
            xs.append(xm + off); ys.append(v); el.append(v - lo); eu.append(hi - v)
            if star: stars.append((xm + off, hi))
            csv_rows.append([family, reg, dist, n, ABBR.get(m, m), src, v, lo, hi, len(s), star])
        if not xs: continue
        ax.bar(xs, ys, w * 0.9, color=SRC_COLOR[src], edgecolor="black", linewidth=0.25, label=SRC_NAME[src], zorder=2)
        ax.errorbar(xs, ys, yerr=[el, eu], fmt="none", ecolor="#222", elinewidth=0.4, capsize=0.8, zorder=3)

    ax.set_xticks(xfw); ax.set_xticklabels([ABBR.get(m, m) for m in fws], rotation=30, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.55, xfw[-1] + 0.55); ax.set_ylabel("success"); ax.grid(axis="y", lw=0.3, alpha=0.35, zorder=0); ax.set_axisbelow(True)
    seeds = max((len(s) for m in fws for s in cell["fw"][m].values()), default=0)
    note = ""
    ax.set_title(f"{family} · n = {n:,} · {dist} · {REGIME_NAME.get(reg, reg)}   ({len(fws)} frameworks × {len(srcs)} shortlists, ≤ {seeds} seeds){note}")
    ax.legend(ncol=min(6, len(srcs) + 2), frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.06))
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
        pd.DataFrame(csv_rows, columns=["family", "regime", "dist", "n", "arm", "shortlist", "mean", "ci_lo", "ci_hi", "seeds", "rerun_outstanding"]).to_csv(f"{OUT}/{family}.csv", index=False)
    with open(f"{OUT}/INDEX.md", "w") as f:
        f.write("# figures/shortlist -- every framework under every shortlist source, per condition\n\n"
                "One figure per (family, n, population shape, liar regime); one panel and one row. The oracle (dotted) and MIDIAN routing\n"
                "the whole population (solid) are horizontal lines with their 95% seed-bootstrap band; each framework group carries one bar per\n"
                "shortlist source. Error bars are the 95% seed bootstrap; * marks a bar with an erratum-28 rerun outstanding. b = 3 only.\n"
                "Do-not-add arms (extra_figs.excluded) are never drawn.\n\n"
                "Shortlist sources: " + "; ".join(f"`{k}` = {v}" for k, v, _ in SOURCES) + ".\n\n")
        for family in families:
            f.write(f"\n## {family}\n")
            for fam, stem in index:
                if fam == family: f.write(f"- `{stem}.png`\n")
    print(f"\n{len(index)} figures -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["live", "routereval"])
