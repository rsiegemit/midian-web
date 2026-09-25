"""Pre-registered TF-IDF shortlist vs the deduplicated one (erratum 25): same cells, seeds, frameworks, paired by row.
    python scripts/dedup_compare.py            # every _dd grid against its source
Reads the _dd grid's rows.d + rows.csv (deduped by rid) and the source grid's rows.csv; pairs on
(cell, method, params minus dedup, seed); prints per (grid, dist, regime) means, the paired delta with a 95% seed
bootstrap, and how many cells had every framework byte-identical (the clone signature) before and after."""
from __future__ import annotations
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from rte.config import RTE_DATA  # noqa: E402
R = f"{RTE_DATA}/results"
SRC = {"fw_live_n1000_dd": "fw_live_n1000", "fw_live_n100_dd": "fw_live_n100", "fw_live_n1000_lowskill_dd": "fw_live_n1000_lowskill",
       "fw_live_n100_lowskill_dd": "fw_live_n100_lowskill", "fw_live_n10k_dd": "live_n10k_v2", "fw_live_n100k_dd": "live_n100k",
       "fw_live_n10k_cartel_dd": "fw_live_n10k_cartel", "fw_k_sensitivity_dd": "fw_k_sensitivity", "fw_appendix_dd": "fw_appendix",
       "budget_b10_fw_dd": "budget_b10_shapes", "churn_n1000_fw_dd": "churn_n1000"}
SRC.update({k.replace("_dd", "_em"): v for k, v in SRC.items() if k.startswith("fw_live")})   # embed grids pair with the same sources
CELL = ["n", "dist", "beta", "liar_select", "seed"]

def load(grid):
    rows = [json.load(open(f)) for f in glob.glob(f"{R}/{grid}/rows.d/*.json")]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    p = f"{R}/{grid}/rows.csv"
    if os.path.exists(p):
        c = pd.read_csv(p, low_memory=False); df = pd.concat([df, c], ignore_index=True)
    if "rid" in df: df = df.drop_duplicates("rid")
    df = df[df.method.astype(str).str.startswith("fw_")].copy()
    df["params"] = df.params.astype(str)
    df["key"] = df.params.str.replace(r'"(dedup|retrieval)":\s*("embed"|true),?\s*', "", regex=True).str.replace(r',\s*}', "}", regex=True).str.replace("{}", "{}")
    return df

def regime(r):
    return "beta=0" if float(r.beta) == 0 else f"beta={float(r.beta):g} {'cartel' if r.liar_select == 'low_skill_first' else 'random'}"

def boot(x, B=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(B)]); return np.percentile(m, [2.5, 97.5])

grids = sys.argv[1:] or list(SRC)
for g in grids:
    new, old = load(g), load(SRC[g])
    old = old[~old.params.str.contains("retrieval")]                        # tfidf arms only
    for d in (new, old):
        d["regime"] = d.apply(regime, axis=1)
        if "churn" in d: d["key"] = d.key + d.churn.where(d.churn.notna() & ~d.churn.astype(str).isin(["None", "nan", "{}"]), "").astype(str)
    keys = CELL + ["method", "key", "regime"]
    new, old = new.drop_duplicates(keys), old.drop_duplicates(keys)   # rows.d and rows.csv may both hold a row
    m = new.merge(old, on=keys, suffixes=("_new", "_old"))
    if m.empty: print(f"\n## {g}: no paired rows yet"); continue
    print(f"\n## {g} <- {SRC[g]}: {len(m)} paired rows ({new.method.nunique()} frameworks new, {old.method.nunique()} old); "
          f"frameworks present in new: {sorted(new.method.unique())}")
    for (dist, reg), q in m.groupby(["dist", "regime"]):
        per_seed = q.groupby("seed").agg(new=("success_new", "mean"), old=("success_old", "mean"))
        delta = per_seed.new - per_seed.old; lo, hi = boot(delta) if len(delta) > 1 else (delta.mean(), delta.mean())
        cells_old = q.groupby("seed").success_old.nunique(); cells_new = q.groupby("seed").success_new.nunique()
        print(f"  {dist:<11} {reg:<18} old {per_seed.old.mean():.3f} -> new {per_seed.new.mean():.3f}  delta {delta.mean():+.3f} [{lo:+.3f},{hi:+.3f}]  "
              f"seeds {len(per_seed)}  identical-framework cells: {int((cells_old == 1).sum())}/{len(cells_old)} -> {int((cells_new == 1).sum())}/{len(cells_new)}")
    fw = m.groupby("method").agg(old=("success_old", "mean"), new=("success_new", "mean"), n=("success_new", "size"))
    fw["delta"] = fw.new - fw.old
    print("  per framework (all cells pooled):\n" + fw.round(3).sort_values("new", ascending=False).to_string())
