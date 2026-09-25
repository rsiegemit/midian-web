"""Aggregate rows -> paired statistics -> diagnostic figures -> pre-registered target check.

    python -m rte.analyze --grid smoke [--grids replay_scale,live_f1_n1000] [--out DIR]

Paired by seed: a cell is one point of the CELL axes and its seeds share a task stream, so
Reference-vs-rival deltas (reference REF = MIDIAN w/o defenses) are taken seed by seed, never as a difference of two
means. A delta is WITHIN_FLOOR when it does not exceed the reference's own seed envelope (max-min of its per-seed
success in that cell). Each method's `group` (framework | midian | declared | verified_central |
verified_decentral | floor | ceiling) comes from its declared `needs` and the fw_ prefix.

The code lives in rte.analysis.{load,stats,legacy_figures,legacy_targets,report}; this module re-exports it.
"""
# ruff: noqa: F401  -- every name below is re-exported
from .analysis.load import (ALIAS, BUILD, CELL_COLS, COST, FLAT, FLAT_ON, FLOOR, PLAIN, REF, RTE_DATA, STATS,
                            cells, consolidate, fmt, group_of, load, log, prepare, reads_declared)
from .analysis.stats import B_BOOT, aggregate, boot, delta, envelope, exponents, fit, pair, paired, sign_test
from .analysis.legacy_figures import MARK, at_n, figures, heat, panel_plot
from .analysis.legacy_targets import targets, targets_v2
from .analysis.report import UPPER, by_channel, by_method, latency, main, md, roll_up, sec, strict, summary

if __name__ == "__main__":
    main()
