"""Every paper figure, in dependency order, one process per step (so no step inherits another's matplotlib state).

    python scripts/figures/make_all.py [--out DIR] [--from-csv] [--scale-matrix] [--only STEP ...]

Full run (needs $RTE_DATA):
  [scale_matrix]      scripts/analysis/scale_matrix.py for the bernoulli / replay sweeps (--scale-matrix; only when
                      their rows changed: it rewrites $RTE_DATA/results/<grid>/matrix_success.csv, read by bar_figs
                      and A / B)
  bar_figs            -> results/aggregates/bars/<family>.csv
  shortlist_figs      -> results/aggregates/shortlist/<family>.csv
  condensed_figs      A, B          (reads bars/, shortlist rows; writes the A / B display lists)
  shortlist_condensed E, F, G, H    (reads shortlist/)
  efficiency_figs     C, D          (re-reads results/aggregates/cost_by_n.csv from the rows)
  lie_max_fig         I             (reads <out>/A_live_allb.csv)
  lie_max_shortlists  F_max_lie     (reads <out>/F_shortlists_1e5.csv)
  lie_max_ab          A_max_lie, B_max_lie  (condensed_figs.build + the lie_max_* rows)
--from-csv redraws A-I and the max-lie panels from results/aggregates alone ($RTE_DATA is pointed at a directory that
does not exist): bars/ and shortlist/ are the inputs of A / B and E-H, cost_by_n.csv of C, figures/*.csv + *.draw.csv +
refs.csv of A / B / D / I and the max-lie panels (A / B max lie: skipped while their csv is not shipped).
Output: <out> = --out, else $RTE_FIG_OUT, else figures/paper.
RTE_SEEDS_4_10=1 python scripts/figures/make_all.py: the 10-seed build. Every figure whose non-framework arms gain seeds
from the seed mirrors (grids.TEN: configs/grids/seeds_4_10.yaml + seeds_to_10.yaml) is written as <name>_new (bars_new/,
shortlist_new/, A_*_new, B_*_new, D_*_new, E-H *_new, I_max_lie_new, F_max_lie_new, A / B_max_lie_new) beside the
originals, which are never replaced; C (bernoulli only) is not rebuilt. Fails until every seed-mirror grid has landed.
It also draws the fixed MIDIAN (midian{reagg: keep}, grids.fixes()) as MIDIAN in every beta > 0 cell, the scale
matrices included: --scale-matrix writes matrix_success_new.csv (with the fixed rows), which bar_figs / A / B read.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from scripts.figures.lib import ROOT, fig_out  # noqa: E402

FIG = "scripts/figures"
STEPS = [
    "scale_matrix",
    "bar_figs",
    "shortlist_figs",
    "condensed_figs",
    "shortlist_condensed",
    "efficiency_figs",
    "lie_max_fig",
    "lie_max_shortlists",
    "lie_max_ab",
]
FROM_CSV = STEPS[3:]


def commands(step, out, from_csv):
    """The command lines of one step."""
    py = [sys.executable]
    if step == "scale_matrix":
        from scripts.figures.lib.grids import MATRICES

        return [py + ["scripts/analysis/scale_matrix.py", g] for g in MATRICES.values()]
    if step in ("bar_figs", "shortlist_figs"):
        return [py + [f"{FIG}/{step}.py"]]
    flags = ["--from-csv"] if from_csv else (["--refresh"] if step == "efficiency_figs" else [])
    return [py + [f"{FIG}/{step}.py", "--out", out] + flags]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", help="output directory (default: $RTE_FIG_OUT or figures/paper)")
    p.add_argument("--from-csv", action="store_true", help="redraw every paper figure from results/aggregates only")
    p.add_argument("--scale-matrix", action="store_true", help="rebuild the scale matrices first (full run only)")
    p.add_argument("--only", nargs="+", choices=STEPS, help="run just these steps (in the fixed order)")
    a = p.parse_args(argv)
    out = os.path.abspath(fig_out(a.out))
    steps = FROM_CSV if a.from_csv else [s for s in STEPS if s != "scale_matrix" or a.scale_matrix]
    steps = [s for s in steps if not a.only or s in a.only]
    env = {**os.environ, "PYTHONPATH": ROOT, "MPLBACKEND": "Agg"}
    if a.from_csv:
        env["RTE_DATA"] = os.path.join(out, ".no_rte_data")  # anything that still reads rows fails loudly
    for step in steps:
        for cmd in commands(step, out, a.from_csv):
            t = time.time()
            r = subprocess.run(cmd, cwd=ROOT, env=env)
            print(f"[make_all] {' '.join(cmd[1:])}: exit {r.returncode}, {time.time() - t:.0f} s", flush=True)
            if r.returncode:
                sys.exit(f"[make_all] {step} failed")


if __name__ == "__main__":
    main()
