"""Shared figure library: paths here; figspec (style, names, colours), rows, stats, regimes, grids, exclusions,
legend_rank (opt-in), seed_tables. No module draws, reads data or changes matplotlib state at import (figspec selects
the Agg backend)."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
AGG = os.path.join(ROOT, "results", "aggregates")  # the CSVs every paper figure is drawn from (make_all --from-csv)
# Opt-in: RTE_SEEDS_4_10=1 is the 10-seed build: it pools every seed mirror (grids.TEN = S4_10 + S_TO_10) into the
# figures and writes each one it changes as <name>_new, next to the originals, which are never replaced.
SEEDS_4_10 = os.environ.get("RTE_SEEDS_4_10") == "1"
NEW = "_new" if SEEDS_4_10 else ""


def fig_out(path=None):
    """Where the paper figures go: `path`, else $RTE_FIG_OUT, else figures/paper; created."""
    out = path or os.environ.get("RTE_FIG_OUT") or os.path.join(ROOT, "figures", "paper")
    os.makedirs(out, exist_ok=True)
    return out
