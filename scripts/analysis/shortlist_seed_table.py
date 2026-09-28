"""Per-seed framework x shortlist table behind Figure F (live, n = 10^5, specialist), for paired tests.

One row per (regime, shortlist, framework, seed) with that seed's task success, taken from the same per-seed series
Figure F averages (scripts/figures/shortlist_figs.collect: same grids, filters and "grid with the most seeds wins" rule).
`body` marks the six shortlists Figure F's body draws (dense and fusion + reranker at their competence-instruction
variant). The bar of Figure F is the mean over full-seed frameworks of the per-framework seed means.

    RTE_DATA=... PYTHONPATH=. python scripts/analysis/shortlist_seed_table.py [--n 100000] [--out results/aggregates/...]
"""

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/scripts/figures")
from shortlist_figs import collect  # noqa: E402

BODY = {"va_cohort", "tfidf", "embed", "dense_icomp", "sota_icomp", "declared"}
REGIMES = {"beta0": "honest", "cartel": "cartel"}


def table(n: int) -> pd.DataFrame:
    cells = collect("live")
    rows = []
    for (cn, dist, reg), cell in cells.items():
        if cn != n or dist != "specialist" or reg not in REGIMES:
            continue
        for fw, by_src in cell["fw"].items():
            for src, per_seed in by_src.items():
                for seed, success in per_seed.items():
                    rows.append(dict(n=n, regime=REGIMES[reg], shortlist=src, body=src in BODY, framework=fw,
                                     seed=int(seed), success=float(success)))
    return pd.DataFrame(rows).sort_values(["regime", "shortlist", "framework", "seed"], ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--out", default=f"{ROOT}/results/aggregates/shortlist/live_1e5_by_seed.csv")
    a = ap.parse_args()
    t = table(a.n)
    t.to_csv(a.out, index=False)
    print(f"{len(t)} rows -> {a.out}")
