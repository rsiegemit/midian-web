"""Reading stored result rows, and the per-row helpers every figure / number script shares.

    read_rows(grid)           rows.d/*.json + rows.csv of one grid under the current method keys (rte.methods.keys)
    load_fw(grid)             read_rows de-duplicated the way the framework-shortlist scripts count (every row, b
    included) label(method, params)     the arm's label, as rte.analyze builds it stat(df, key)             one number
    per row from the method_stats JSON column pending_reruns()          {(grid, framework, dist, regime)} whose
    erratum-28 rerun is outstanding
RESULTS is $RTE_DATA/results (rte.config).
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from rte.analyze import ALIAS
from rte.config import RTE_DATA
from rte.methods import keys
from scripts.figures.lib import grids
from scripts.figures.lib.regimes import tag

RESULTS = f"{RTE_DATA}/results"


def read_rows(grid, usecols=None, where=None, rowsd=True, csv_first=False):
    """Every stored row of `grid`, method keys normalised; rows.d first (a rerun writes rows.d only), then rows.csv.
    usecols: rows.csv columns to read (those present); where(frame) -> frame filters rows.csv chunk by chunk after
    normalising (rows.d is small and read whole); rowsd=False skips rows.d; csv_first puts rows.csv first.
    No de-duplication: callers drop duplicate rids as they need. Nothing stored -> an empty frame."""
    d = f"{RESULTS}/{grid}"
    fr = (
        [
            keys.normalize(
                pd.DataFrame(
                    [
                        {**json.load(open(f)), "rid": os.path.basename(f)[:-5]}  # the file name IS the rid
                        for f in glob.glob(f"{d}/rows.d/*.json")
                    ]
                ),
                d,
            )
        ]
        if rowsd
        else []
    )
    if os.path.exists(f"{d}/rows.csv"):
        cols = None if usecols is None else [c for c in usecols if c in pd.read_csv(f"{d}/rows.csv", nrows=0).columns]
        if where is None:
            fr.append(keys.normalize(pd.read_csv(f"{d}/rows.csv", usecols=cols, low_memory=False), d))
        else:
            fr.append(
                pd.concat(
                    where(keys.normalize(c, d))
                    for c in pd.read_csv(f"{d}/rows.csv", usecols=cols, chunksize=500000, low_memory=False)
                )
            )
    fr = [f for f in (fr[::-1] if csv_first else fr) if not f.empty]
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()


def load_fw(grid):
    """rows.d + rows.csv, one row per rid and per (cell, seed, method, params); params as strings."""
    df = read_rows(grid)
    if df.empty:
        return df
    if "rid" in df:
        df = df.drop_duplicates("rid")
    df["params"] = df.params.astype(str)
    return df.drop_duplicates(
        [c for c in ("n", "b", "dist", "beta", "liar_select", "seed", "method", "params") if c in df]
    )


def label(method, params):
    p = json.loads(params) if isinstance(params, str) and params.startswith("{") else {}
    short = ",".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(p.items()))
    lab = method if not p else f"{method}[{short}]"
    return ALIAS.get(lab, lab)


def stat(df, key):
    """One number per row from the method_stats JSON column (NaN when absent)."""
    return df.get("method_stats", pd.Series(index=df.index, dtype=object)).map(
        lambda s: (json.loads(s) if isinstance(s, str) and s.startswith("{") else {}).get(key, np.nan)
    )


def pending_reruns():
    """{(grid, framework, dist, regime)} whose erratum-28 rerun is still outstanding -- those numbers carry an asterisk.
    Empty once finalize_stage2 has fired; the ablation grids stay pending between stage 1 and stage 2."""
    p = f"{RESULTS}/quarantine_units.tsv"
    if not os.path.exists(p) or os.path.exists(f"{RTE_DATA}/logs/DONE_stage2"):
        return set()
    u = pd.read_csv(p, sep="\t")
    if os.path.exists(f"{RTE_DATA}/logs/DONE_stage1"):
        u = u[u.grid.str.fullmatch(r"fw_live_n(1000|100)(_lowskill)?_sota")]
    u = u[
        [not landed(*k) for k in zip(u.grid, u.method, u.dist, u.beta, u.liar_select, u.seed)]
    ]  # a rerun on disk is not outstanding
    return {(g, m, d, tag(b, l)) for g, m, d, b, l in zip(u.grid, u.method, u.dist, u.beta, u.liar_select)}


def landed(grid, method, dist, beta, ls, seed, _c={}):
    """True when every param variant of `method` in `grid` has a row for (dist, beta, liar_select, seed)."""
    if grid not in _c:
        from rte.run import blocks, method_specs

        cfg = grids.config()
        want = {}
        for blk in blocks(cfg, grid) if grid in cfg["grids"] else []:
            for sp in method_specs(blk):
                want.setdefault(sp["name"], set()).add(json.dumps(sp["params"], sort_keys=True, separators=(",", ":")))
        df = load_fw(grid)
        have = (
            df.groupby(["method", "dist", "beta", "liar_select", "seed"]).params.nunique().to_dict()
            if not df.empty
            else {}
        )
        _c[grid] = (want, have)
    want, have = _c[grid]
    return have.get((method, dist, float(beta), ls, int(seed)), 0) >= len(want.get(method, {None}))
