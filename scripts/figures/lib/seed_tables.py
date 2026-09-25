"""Per-seed success tables (seed x arm) for every condensed A / B cell and budget, read straight from the raw rows.
    from scripts.figures.lib.seed_tables import tables
tables() -> {(family, group, n, regime): {b: DataFrame(index = seed, columns = arm label, values = success)}}
  live n        b = 3: grids.LIVE_GRIDS[n] (specialist, self-described channel, grids averaged per seed as
                bar_figs does)
                b = 1 / 5: va_b_* + rivals_b_*;  tuned_wsb_* (n0 = 0.5) at whatever b it ran
  every family  + pool_fill_* (the candidates a cell lacked at some b; configs/grids/)
  RouterEval    the 5,000-LLM leaderboard pool: routereval_mmlu5k (b = 3) + va_b / rivals_b routereval5k
  LLMRouterBench  llmrouterbench_pool (b = 3) + va_b / rivals_b llmrouterbench
  bernoulli 1e7 bernoulli_scale_v5 (b = 1, 3) + va_b / rivals_b bernoulli_1e7 (b = 5)
  replay 1e6    replay_scale_v5 (b = 1, 3) + va_b / rivals_b replay_1e6; shapes pooled per seed, only seeds with
                ALL three
Regimes: beta0 = beta 0 (any liar_select: no liars, so the tag is inert); cartel = beta 0.5, low_skill_first.
The best of a pool is cross-fitted from these tables (stats.crossfit)."""

from __future__ import annotations

import json

import pandas as pd

from scripts.figures.lib import grids
from scripts.figures.lib.regimes import ab
from scripts.figures.lib.rows import label, read_rows

COLS = ["rid", "n", "b", "dist", "beta", "liar_select", "declared_source", "seed", "method", "params", "success"]


def rows(grid, n):
    """One grid's rows at population n (rows.csv + rows.d), de-duplicated on rid; never the framework arms."""
    df = read_rows(grid, COLS, where=lambda c: c[c.n == n])
    if df.empty:
        df = pd.DataFrame(columns=COLS)
    df = df[(df.n == n) & ~df.method.astype(str).str.startswith("fw_")]
    if "rid" in df:
        df = df.drop_duplicates("rid")
    return df.assign(grid=grid, params=df.params.astype(str))


def _table(df, pooled_shapes=False):
    """rows -> {(regime, b): seed x arm}. Rows of one (seed, arm) from several grids are averaged, as bar_figs does."""
    if df.empty:
        return {}
    df = df.assign(
        label=[label(m, p) for m, p in zip(df.method, df.params)],
        regime=[ab(b, l) for b, l in zip(df.beta, df.liar_select)],
    )
    df = df[df.regime.notna()]
    out = {}
    for (reg, b), q in df.groupby(["regime", "b"]):
        if pooled_shapes:  # a seed counts only if the arm ran on every shape
            s = q.groupby(["label", "seed", "dist"]).success.mean().unstack("dist")
            s = s.dropna().mean(axis=1) if s.shape[1] == 3 else s.iloc[0:0, 0]
            T = s.unstack("label")
        else:
            T = q.groupby(["seed", "label"]).success.mean().unstack("label")
        out[(reg, int(b))] = T
    return out


# Erratum 30: the non-live families switch to the calibrated-claims / split / no-repeat reruns once that grid has
# every row
# (complete()); until then the old rows stand. bernoulli: only the claim-reading arms change (the world, liars, stream
# and probes
# do not depend on declared_source), so their programmatic rows are swapped for the calibrated ones; the others move
# wholesale.
CLAIM_READERS = {
    "cluster_head_router",
    "disrouter_cascade",
    "warm_start_bandit",
    "warm_start_bandit[n0=0.5]",
    "declared_argmax",
}
_done = {}


def planned(grid):
    """The set of (n, b, dist, beta, liar_select, seed, arm label) rows the grid loader plans for `grid` (oracle rows
    excluded)."""
    from midian.run import blocks, cells, method_specs, seeds

    cfg = grids.config()
    return {
        (
            int(c["n"]),
            int(c["b"]),
            str(c["dist"]),
            float(c["beta"]),
            str(c["liar_select"]),
            int(s),
            label(m["name"], json.dumps(m["params"])),
        )
        for blk in blocks(cfg, grid)
        for c in cells(blk)
        for s in seeds(blk["seeds"])
        for m in method_specs(blk)
    }


def complete(grid, load=None):
    """True once EVERY planned row of `grid` has landed (set comparison, so stray or duplicate rows cannot stand in for
    a missing one). `load(grid)` returns its rows; default rows(), which drops framework arms -- pass a loader for
    those."""
    if grid not in _done:
        want = planned(grid)
        df = load(grid) if load else pd.concat([rows(grid, n) for n in {k[0] for k in want}], ignore_index=True)
        have = (
            set()
            if df.empty
            else {
                (int(n), int(b), str(d), float(be), str(ls), int(s), label(m, p))
                for n, b, d, be, ls, s, m, p in zip(
                    df.n, df.b, df.dist, df.beta, df.liar_select, df.seed, df.method, df.params.astype(str)
                )
            }
        )
        _done[grid] = want <= have
    return _done[grid]


def switched():
    """The families whose B cell reads the erratum-30 rows."""
    return {f for f, g in grids.ERRATUM30.items() if complete(g)}


def tables():
    sw = switched()
    E30 = grids.ERRATUM30
    spec = [
        (
            ("live", "specialist", n),
            grids.LIVE_GRIDS[n] + grids.b_grids(t) + [f"tuned_wsb_{t}", f"pool_seeds_{t}"],
            "live",
        )
        for n, t in grids.SIZE.items()
    ]
    spec += [
        (
            ("routereval", "strong_to_weak", 5000),
            [E30["routereval"]] if "routereval" in sw else ["routereval_mmlu5k"] + grids.b_grids("routereval5k"),
            None,
        ),
        (
            ("llmrouterbench", "20 models", 20),
            [E30["llmrouterbench"]]
            if "llmrouterbench" in sw
            else ["llmrouterbench_pool"] + grids.b_grids("llmrouterbench"),
            None,
        ),
        (
            ("bernoulli", "specialist", 10**7),
            [grids.MATRICES["bernoulli"]]
            + grids.b_grids("bernoulli_1e7")
            + ([E30["bernoulli"]] if "bernoulli" in sw else []),
            "specialist",
        ),
        (
            ("replay", "all shapes pooled", 10**6),
            [E30["replay"]] if "replay" in sw else [grids.MATRICES["replay"]] + grids.b_grids("replay_1e6"),
            "pooled",
        ),
    ]
    out = {}
    for (fam, grp, n), gs, kind in spec:
        df = pd.concat([rows(g, n) for g in gs], ignore_index=True)
        if not df.empty:  # TrueSkill: post probe-duplicate-fix rows only (erratum 30)
            df = df[
                (df.method != "trueskill_per_family")
                | df.grid.str.startswith("trueskill_fix_")
                | df.grid.isin(list(E30.values()))
            ]
        if fam == "bernoulli" and fam in sw:  # claim readers: calibrated rows only
            reads = pd.Series([label(m, p) in CLAIM_READERS for m, p in zip(df.method, df.params)], index=df.index)
            df = df[~reads | (df.declared_source == "calibrated")]
        if kind in ("live", "specialist") and not df.empty:
            df = df[df.dist == "specialist"]
        if kind == "live" and "declared_source" in df:  # the live headline is the self-described channel (bar_figs)
            df = df[df.declared_source.isna() | (df.declared_source == "self_described")]
        for (reg, b), T in _table(df, pooled_shapes=kind == "pooled").items():
            out.setdefault((fam, grp, n, reg), {})[b] = T
    return out
