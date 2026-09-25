"""Stored-row id check (refactor invariant G2): rte.run.rid_of_row(row) must equal the rid the row is stored under.

For every results directory holding rows.d/ and/or rows.csv, sample up to --n rows from each source (seeded, so the
sample is reproducible): a rows.d row's rid is its file name, a rows.csv row's rid is its `rid` column (CSV rows with
no rid are counted, not checked). Prints one line per directory and the total; exits 1 on any mismatch.
Directories starting with "_" are archives of retired runners (e.g. _legacy_smoke_oldrunner, whose ids predate the
current row_id) and are skipped unless --all.

    PYTHONPATH=. python scripts/checks/rid_check.py [--results $RTE_DATA/results] [--n 30] [--seed 0]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from rte.run import RTE_DATA, rid_of_row


def sample_rows_d(path, n, rng):
    """[(stored_rid, row)] for up to n files of rows.d."""
    names = sorted(f for f in os.listdir(path) if f.endswith(".json"))
    pick = sorted(rng.choice(len(names), min(n, len(names)), replace=False)) if names else []
    out = []
    for i in pick:
        try:
            with open(f"{path}/{names[i]}") as fh:
                out.append((names[i][:-5], json.load(fh)))
        except (FileNotFoundError, json.JSONDecodeError):  # merged-and-pruned or mid-write under a live job
            pass
    return out


def sample_rows_csv(path, n, rng):
    """[(stored_rid, row)] for up to n data lines of rows.csv; read with pandas (same dtype path as the runner)."""
    with open(path, "rb") as fh:
        lines = sum(1 for _ in fh) - 1
    if lines <= 0:
        return []
    keep = {int(i) + 1 for i in rng.choice(lines, min(n, lines), replace=False)}  # +1: skip the header line
    df = pd.read_csv(path, skiprows=lambda i: i != 0 and i not in keep, low_memory=False)
    if "rid" not in df.columns:
        return [(None, r) for _, r in df.iterrows()]
    return [(r["rid"] if isinstance(r["rid"], str) else None, r) for _, r in df.iterrows()]


def check_dir(d, n, seed):
    """{'rows.d': (checked, mismatched, no_rid), 'rows.csv': (...)} for one results directory."""
    res, bad = {}, []
    for src, fn in (("rows.d", sample_rows_d), ("rows.csv", sample_rows_csv)):
        path = f"{d}/{src}"
        if not os.path.exists(path):
            continue
        rows = fn(path, n, np.random.default_rng(seed))
        checked = [(rid, r) for rid, r in rows if rid is not None]
        miss = [(rid, got) for rid, got in ((rid, rid_of_row(r)) for rid, r in checked) if got != rid]
        bad += [(src, *m) for m in miss]
        res[src] = (len(checked), len(miss), len(rows) - len(checked))
    return res, bad


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=f"{RTE_DATA}/results")
    p.add_argument("--n", type=int, default=30, help="rows sampled per source (rows.d, rows.csv) per directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dirs", help="comma-separated subset of result directory names")
    p.add_argument("--all", action="store_true", help='also check "_"-prefixed archive directories')
    a = p.parse_args(argv)
    names = a.dirs.split(",") if a.dirs else [d for d in sorted(os.listdir(a.results)) if a.all or d[0] != "_"]
    dirs = [d for d in names if any(os.path.exists(f"{a.results}/{d}/{s}") for s in ("rows.d", "rows.csv"))]
    tot_checked = tot_bad = tot_norid = 0
    print("dir\trows.d checked/mismatch/no_rid\trows.csv checked/mismatch/no_rid")
    for d in dirs:
        res, bad = check_dir(f"{a.results}/{d}", a.n, a.seed)
        fmt = lambda s: "/".join(map(str, res[s])) if s in res else "-"
        print(f"{d}\t{fmt('rows.d')}\t{fmt('rows.csv')}", flush=True)
        for src, stored, got in bad:
            print(f"  MISMATCH {d}/{src}: stored={stored} recomputed={got}", flush=True)
        tot_checked += sum(v[0] for v in res.values())
        tot_bad += sum(v[1] for v in res.values())
        tot_norid += sum(v[2] for v in res.values())
    print(f"# {len(dirs)} dirs, {tot_checked} rows checked, {tot_bad} mismatches, {tot_norid} CSV rows without rid")
    sys.exit(1 if tot_bad else 0)


if __name__ == "__main__":
    main()
