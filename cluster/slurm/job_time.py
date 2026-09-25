"""Walltime for one framework unit: 1.5 x the measured p95 runtime of that (population size, framework), from
results/job_sizing.csv (rebuilt from sacct). Rule (JDS, 2026-09-22): never a flat multi-day limit -- a flat 1-3 day
limit on jobs whose median is minutes used 16 % of allocated time. p95s that sat at the old 24 h cap are censored, so
those get 2 x p95. Floor 1 h, cap 3 days. Durations track FLEET LOAD, so rebuild the table when the fleet changes.
    python cluster/slurm/job_time.py fw_live_n1000_sota fw_camel_workforce  ->  2-00:00:00"""

import re

import pandas as pd


def minutes(grid: str, method: str) -> int:
    from midian.config import RTE_DATA

    t = pd.read_csv(f"{RTE_DATA}/results/job_sizing.csv")
    size = (re.search(r"fw_live_(n\d+k?)", grid) or [None, None])[1]
    row = t[(t["size"] == size) & (t["method"] == method)]
    if row.empty:  # unmeasured combination: the largest p95 seen for that size
        row = t[t["size"] == size].nlargest(1, "p95")
    p95 = float(row.p95.iloc[0]) if len(row) else 720.0
    m = 2.0 * p95 if p95 >= 0.95 * 1440 else 1.5 * p95  # censored at the old 1-day cap
    return int(min(max(m, 60), 4320))


def slurm(m: int) -> str:
    d, r = divmod(m, 1440)
    h, mm = divmod(r, 60)
    return f"{d}-{h:02d}:{mm:02d}:00" if d else f"{h:02d}:{mm:02d}:00"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grid")
    ap.add_argument("method")
    a = ap.parse_args()
    print(slurm(minutes(a.grid, a.method)))
