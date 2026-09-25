"""Shared figure library: paths here; figspec (style, names, colours), rows, stats, regimes, grids, exclusions,
legend_rank (opt-in), seed_tables. No module draws, reads data or changes matplotlib state at import (figspec selects
the Agg backend)."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
AGG = os.path.join(ROOT, "results", "aggregates")  # the CSVs every paper figure is drawn from (make_all --from-csv)


def fig_out(path=None):
    """Where the paper figures go: `path`, else $RTE_FIG_OUT, else figures/paper; created."""
    out = path or os.environ.get("RTE_FIG_OUT") or os.path.join(ROOT, "figures", "paper")
    os.makedirs(out, exist_ok=True)
    return out
