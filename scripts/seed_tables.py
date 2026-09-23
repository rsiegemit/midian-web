"""Per-seed success tables (seed x arm) for every condensed A / B cell and budget, read straight from the raw rows, and the
cross-fitted "best of a pool" that uses them.
    from seed_tables import tables, crossfit
tables() -> {(family, group, n, regime): {b: DataFrame(index = seed, columns = arm label, values = success)}}
  live n        b = 3: bar_figs.LIVE_GRIDS[n] (specialist, self-described channel, grids averaged per seed as bar_figs does)
                b = 1 / 5: va_b_* + rivals_b_*;  tuned_wsb_* (n0 = 0.5) at whatever b it ran
  every family  + pool_fill_* (the candidates a cell lacked at some b; configs/grid.yaml)
  RouterEval    the 5,000-LLM leaderboard pool: routereval_mmlu5k (b = 3) + va_b / rivals_b routereval5k
  LLMRouterBench  llmrouterbench_pool (b = 3) + va_b / rivals_b llmrouterbench
  bernoulli 1e7 bernoulli_scale_v5 (b = 1, 3) + va_b / rivals_b bernoulli_1e7 (b = 5)
  replay 1e6    replay_scale_v5 (b = 1, 3) + va_b / rivals_b replay_1e6; shapes pooled per seed, only seeds with ALL three
Regimes: beta0 = beta 0 (any liar_select: no liars, so the tag is inert); cartel = beta 0.5, low_skill_first.
crossfit(T): winner's-curse-free "best of a pool": for each seed the arm is picked on the OTHER seeds' means and scored on
this seed; the bar is the mean of those per-seed scores."""
from __future__ import annotations
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rte.analyze import ALIAS

R = os.environ.get("RTE_DATA", "/scratch/rte") + "/results"
COLS = ["rid", "n", "b", "dist", "beta", "liar_select", "declared_source", "seed", "method", "params", "success"]
LIVE = {100: "n100", 1000: "n1000", 10000: "n10k", 100000: "n100k"}


def label(method, params):
    p = json.loads(params) if isinstance(params, str) and params.startswith("{") else {}
    short = ",".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(p.items()))
    l = method if not p else f"{method}[{short}]"; return ALIAS.get(l, l)


def rows(grid, n):
    """One grid's rows at population n (rows.csv + rows.d), de-duplicated on rid; never the framework arms."""
    fr = [pd.DataFrame([json.load(open(f)) for f in glob.glob(f"{R}/{grid}/rows.d/*.json")])]
    if os.path.exists(f"{R}/{grid}/rows.csv"):
        head = pd.read_csv(f"{R}/{grid}/rows.csv", nrows=0).columns
        fr.append(pd.concat(c[c.n == n] for c in pd.read_csv(f"{R}/{grid}/rows.csv", usecols=[c for c in COLS if c in head], chunksize=500000, low_memory=False)))
    df = pd.concat([f for f in fr if not f.empty], ignore_index=True) if any(not f.empty for f in fr) else pd.DataFrame(columns=COLS)
    df = df[(df.n == n) & ~df.method.astype(str).str.startswith("fw_")]
    if "rid" in df: df = df.drop_duplicates("rid")
    return df.assign(grid=grid, params=df.params.astype(str))


def _regime(beta, ls):
    return "beta0" if float(beta) == 0 else ("cartel" if float(beta) == 0.5 and ls == "low_skill_first" else None)


def _table(df, pooled_shapes=False):
    """rows -> {(regime, b): seed x arm}. Rows of one (seed, arm) from several grids are averaged, as bar_figs does."""
    if df.empty: return {}
    df = df.assign(label=[label(m, p) for m, p in zip(df.method, df.params)], regime=[_regime(b, l) for b, l in zip(df.beta, df.liar_select)])
    df = df[df.regime.notna()]
    out = {}
    for (reg, b), q in df.groupby(["regime", "b"]):
        if pooled_shapes:                                    # a seed counts only if the arm ran on every shape
            s = q.groupby(["label", "seed", "dist"]).success.mean().unstack("dist")
            s = s.dropna().mean(axis=1) if s.shape[1] == 3 else s.iloc[0:0, 0]
            T = s.unstack("label")
        else:
            T = q.groupby(["seed", "label"]).success.mean().unstack("label")
        out[(reg, int(b))] = T
    return out


def tables():
    from bar_figs import LIVE_GRIDS
    b_grids = lambda t: [f"va_b_{t}", f"rivals_b_{t}", f"pool_fill_{t}"]
    spec = [(("live", "specialist", n), LIVE_GRIDS[n] + b_grids(t) + [f"tuned_wsb_{t}", f"pool_seeds_{t}"], "live") for n, t in LIVE.items()]
    spec += [(("routereval", "strong_to_weak", 5000), ["routereval_mmlu5k"] + b_grids("routereval5k"), None),
             (("llmrouterbench", "20 models", 20), ["llmrouterbench_pool"] + b_grids("llmrouterbench"), None),
             (("bernoulli", "specialist", 10 ** 7), ["bernoulli_scale_v5"] + b_grids("bernoulli_1e7"), "specialist"),
             (("replay", "all shapes pooled", 10 ** 6), ["replay_scale_v5"] + b_grids("replay_1e6"), "pooled")]
    out = {}
    for (fam, grp, n), grids, kind in spec:
        df = pd.concat([rows(g, n) for g in grids], ignore_index=True)
        if kind in ("live", "specialist") and not df.empty: df = df[df.dist == "specialist"]
        if kind == "live" and "declared_source" in df:       # the live headline is the self-described channel (bar_figs)
            df = df[df.declared_source.isna() | (df.declared_source == "self_described")]
        for (reg, b), T in _table(df, pooled_shapes=kind == "pooled").items():
            out.setdefault((fam, grp, n, reg), {})[b] = T
    return out


def crossfit(T, pool):
    """Cross-fitted best of `pool` in a seed x arm table. For each seed s: the arm with the highest mean over the OTHER
    seeds (among arms that ran on s) is picked, and its success ON s is s's score. Returns (per-seed scores, picks)."""
    T = T[[a for a in pool if a in T.columns]].dropna(how="all")
    vals, picks = {}, {}
    for s in T.index:
        rest = T.drop(s).mean()                              # skipna: an arm is judged on the other seeds it has
        rest = rest[T.loc[s].notna() & rest.notna()]
        if rest.empty: continue
        c = rest.idxmax(); vals[s], picks[s] = float(T.at[s, c]), c
    return pd.Series(vals, dtype=float), pd.Series(picks, dtype=object)
