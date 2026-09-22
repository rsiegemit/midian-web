"""v6.* entries: the framework shortlist variants (pre-registered TF-IDF, dedup, MiniLM embed, MIDIAN-VA cohort) and the live
10^5 peer-halving cells. Read straight from each grid's rows.csv / rows.d (<= 1,500 rows per grid; never through rte.analyze.load).
    python scripts/fw_variant_numbers.py            # standalone: prints the entries
    from fw_variant_numbers import collect; collect(N)   # from paper_numbers.py
Keys: v6.<variant>.n<n>.<dist>.<regime>.<framework|frameworks_mean|frameworks_best>  and  v6.halving_live.n100000.<regime>.seed<k>
Regimes follow RESULTS: beta0 (liar-free, one cell), beta<x>_random, beta<x>_cartel (low_skill_first); the cartel at beta = 0.5 is 'cartel'."""
from __future__ import annotations
import glob, json, os
import numpy as np, pandas as pd

RTE_DATA = os.environ.get("RTE_DATA", "/scratch/rte"); R = f"{RTE_DATA}/results"
PENDING: set = set()
VARIANTS = {                                   # variant -> [(grid, filter on params)]; the pre-registered arms are the tfidf rows of the source grids
    "tfidf": [("fw_live_n100", "plain"), ("fw_live_n1000", "plain"), ("fw_live_n100_lowskill", "plain"), ("fw_live_n1000_lowskill", "plain"),
              ("live_n10k_v2", "plain"), ("fw_live_n10k_cartel", "plain"), ("live_n100k", "plain")],
    "dedup": [(g, "dedup") for g in ("fw_live_n100_dd", "fw_live_n1000_dd", "fw_live_n100_lowskill_dd", "fw_live_n1000_lowskill_dd", "fw_live_n10k_dd", "fw_live_n10k_cartel_dd", "fw_live_n100k_dd")],
    "embed": [(g, "embed") for g in ("fw_live_n100_em", "fw_live_n1000_em", "fw_live_n100_lowskill_em", "fw_live_n1000_lowskill_em", "fw_live_n10k_em", "fw_live_n10k_cartel_em", "fw_live_n100k_em")],
    "va_cohort": [(g, "midian_va") for g in ("fw_live_n100_verified_va", "fw_live_n100_verified_va_lowskill", "fw_live_n1000_verified_va", "fw_live_n1000_verified_va_lowskill",
                                             "fw_live_n10k_verified_va", "fw_live_n10k_cartel_verified_va", "fw_live_n100k_verified_va")],
    "v_cohort": [(g, "midian") for g in ("fw_live_n100_verified", "fw_live_n1000_verified")],
}


def load(grid):
    rows = [json.load(open(f)) for f in glob.glob(f"{R}/{grid}/rows.d/*.json")]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    p = f"{R}/{grid}/rows.csv"
    if os.path.exists(p): df = pd.concat([df, pd.read_csv(p, low_memory=False)], ignore_index=True)
    if df.empty: return df
    if "rid" in df: df = df.drop_duplicates("rid")
    df["params"] = df.params.astype(str)
    return df.drop_duplicates(["n", "dist", "beta", "liar_select", "seed", "method", "params"])   # n: some grids hold several sizes


def pending_reruns():
    """{(grid, framework, dist, regime)} whose erratum-28 rerun is still outstanding -- those numbers carry an asterisk.
    Empty once finalize_stage2 has fired; the ablation grids stay pending between stage 1 and stage 2 (campaign_tick.sh)."""
    p = f"{R}/quarantine_units.tsv"
    if not os.path.exists(p) or os.path.exists(f"{RTE_DATA}/logs/DONE_stage2"): return set()
    u = pd.read_csv(p, sep="\t")
    if os.path.exists(f"{RTE_DATA}/logs/DONE_stage1"): u = u[u.grid.str.fullmatch(r"fw_live_n(1000|100)(_lowskill)?_sota")]
    return {(g, m, d, regime(b, l)) for g, m, d, b, l in zip(u.grid, u.method, u.dist, u.beta, u.liar_select)}


def select(df, kind):
    fw = df[df.method.astype(str).str.startswith("fw_") & ~df.params.str.contains("supervisor")]   # the 14B Magentic arm is its own entry elsewhere
    if kind == "plain": return fw[~fw.params.str.contains("retrieval|dedup")]
    if kind == "dedup": return fw[fw.params.str.contains('"dedup"') & ~fw.params.str.contains("retrieval")]
    if kind == "embed": return fw[fw.params.str.contains('"embed"')]
    if kind == "midian_va": return fw[fw.params.str.contains('"midian_va"') & fw.params.str.contains('"r":10')]
    if kind == "midian": return fw[fw.params.str.contains('"retrieval":"midian"') & fw.params.str.contains('"r":10')]
    raise ValueError(kind)


def regime(beta, ls):
    beta = float(beta)
    if beta == 0: return "beta0"
    tag = "cartel" if ls == "low_skill_first" else "random"
    return "cartel" if (beta == 0.5 and tag == "cartel") else f"beta{beta:g}_{tag}".replace(".", "")


def ci(x, B=2000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    if len(x) < 2: return float(x.mean()), float(x.mean())
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(B)]); return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def entry(per_seed, grid, note=None):
    lo, hi = ci(per_seed); e = dict(value=round(float(np.mean(per_seed)), 4), grid=grid, units=int(len(per_seed)), ci=[round(lo, 4), round(hi, 4)])
    return e | ({"note": note} if note else {})


def collect(N):
    global PENDING; PENDING = pending_reruns()
    for variant, grids in VARIANTS.items():
        for grid, kind in grids:
            df = load(grid)
            if df.empty: continue
            fw = select(df, kind)
            if fw.empty: continue
            fw = fw.assign(regime=[regime(b, l) for b, l in zip(fw.beta, fw.liar_select)])
            if variant == "tfidf" and grid == "live_n100k": pass
            for (n, dist, reg), q in fw.groupby(["n", "dist", "regime"]):
                if reg == "beta0" and q.liar_select.nunique() > 1: q = q[q.liar_select == "random"]     # liar-free: one cell
                base = f"v6.{variant}.n{int(n)}.{dist}.{reg}"
                per_fw = q.groupby(["method", "seed"]).success.mean().unstack(0)      # seeds x frameworks
                star = {m for m in per_fw if (grid, m, dist, reg) in PENDING}
                pend = lambda ms: "; * erratum-28 rerun outstanding" if ms & star else ""
                N[f"{base}.frameworks_mean"] = entry(per_fw.mean(axis=1), grid, f"{per_fw.shape[1]} frameworks, seed means" + pend(set(per_fw)))
                best = per_fw.mean().idxmax(); N[f"{base}.frameworks_best"] = entry(per_fw[best], grid, f"best = {best}" + pend({best}))
                for m in per_fw: N[f"{base}.{m}"] = entry(per_fw[m].dropna(), grid, pend({m})[2:] or None)
            # the identical-framework (clone) signature: cells where every framework has the same success
            same = fw.groupby(["n", "dist", "regime", "seed"]).success.nunique().eq(1).groupby(level=[0, 1, 2]).mean()
            for (n, dist, reg), v in same.items(): N[f"v6.{variant}.n{int(n)}.{dist}.{reg}.identical_cells_frac"] = dict(value=round(float(v), 4), grid=grid, units=None, ci=None)
    # live 10^5 peer-reported halving, per cell and seed (asterisked while the liar cells are partial)
    live = load("live_n100k")
    if not live.empty:
        h = live[(live.method == "sequential_halving") & live.params.str.contains("peer_reported")]
        o = live[live.method == "oracle"].set_index(["beta", "liar_select", "seed"]).success
        for _, r in h.iterrows():
            key = f"v6.halving_live.n100000.{regime(r.beta, r.liar_select)}.seed{int(r.seed)}"
            N[key] = dict(value=round(float(r.success), 4), grid="live_n100k", units=1, ci=None,
                          note=f"oracle same cell {float(o.get((r.beta, r.liar_select, int(r.seed)), float('nan'))):.3f}; misroute_to_liar {float(r.get('misroute_to_liar', float('nan'))):.3f}")
        done = h.groupby(["beta", "liar_select"]).seed.nunique().to_dict()
        N["v6.halving_live.n100000.cells_done"] = dict(value={f"{regime(b, l)}": int(k) for (b, l), k in done.items()}, grid="live_n100k", units=None, ci=None,
                                                       note="seeds landed per cell out of 3; partial cells carry an asterisk in RESULTS")
    return N


if __name__ == "__main__":
    N = collect({})
    for k in sorted(N):
        if k.endswith(("frameworks_mean", "frameworks_best")) or "halving_live" in k: print(k, N[k]["value"], N[k].get("ci"), N[k].get("note", ""))
    print(len(N), "entries")
