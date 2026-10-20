"""Shared figure library: paths and the figure-CSV publishing here; figspec (style, names, colours), pools, rows,
stats, regimes, grids, exclusions, legend_rank (opt-in), seed_tables. No module draws, reads data or changes matplotlib
state at import (figspec selects the Agg backend)."""

import os
import shutil

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
AGG = os.path.join(ROOT, "results", "aggregates")  # the CSVs every paper figure is drawn from (make_all --from-csv)
DRAW = os.path.join(AGG, "figures")  # per figure: <name>.csv, its display list <name>.draw.csv, its lines in refs.csv
# Opt-in: RTE_SEEDS_4_10=1 is the 10-seed build: it pools every seed mirror (grids.TEN = S4_10 + S_TO_10) into the
# figures and writes each one it changes as <name>_new, next to the originals, which are never replaced.
SEEDS_4_10 = os.environ.get("RTE_SEEDS_4_10") == "1"
NEW = "_new" if SEEDS_4_10 else ""


def fig_out(path=None):
    """Where the paper figures go: `path`, else $RTE_FIG_OUT, else figures/paper; created."""
    out = path or os.environ.get("RTE_FIG_OUT") or os.path.join(ROOT, "figures", "paper")
    os.makedirs(out, exist_ok=True)
    return out


def publish(name, out, rec, dl=None, refs=None):
    """A figure's CSV `rec` to <out>/<name>.csv and DRAW/<name>.csv, its display list `dl` to DRAW/<name>.draw.csv, and
    its reference lines `refs` (column figure = name) in place of its old rows of DRAW/refs.csv."""
    os.makedirs(DRAW, exist_ok=True)
    rec.to_csv(f"{out}/{name}.csv", index=False)
    rec.to_csv(f"{DRAW}/{name}.csv", index=False)
    if dl is not None:
        dl.to_csv(f"{DRAW}/{name}.draw.csv", index=False)
    if refs is not None:
        p = f"{DRAW}/refs.csv"
        old = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=refs.columns)
        pd.concat([old[old.figure != name], refs], ignore_index=True).to_csv(p, index=False)


def republish(name, out, refs=False):
    """--from-csv: DRAW/<name>.csv copied to <out>; returns it (and, with `refs`, its rows of DRAW/refs.csv)."""
    shutil.copyfile(f"{DRAW}/{name}.csv", f"{out}/{name}.csv")
    rec = pd.read_csv(f"{DRAW}/{name}.csv")
    if not refs:
        return rec
    r = pd.read_csv(f"{DRAW}/refs.csv")
    return rec, r[r.figure == name]
