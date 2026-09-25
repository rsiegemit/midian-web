"""Figure-CSV comparison (refactor invariant G4): two figures/ trees must hold value-identical CSVs.

Every *.csv under either tree (matched by relative path) is read with pandas; rows are sorted on all columns so row
order does not matter; numeric columns must agree within --atol (default 1e-12, NaN == NaN), all others exactly.
Reports, one line each:
    MISSING <path> (only in A|B)      COLUMNS <path> (column sets differ)      DIFF <path> (<n> cells, first few shown)
Exit 1 on any of them.

    python scripts/checks/figure_csvs.py base/figures new/figures [--atol 1e-12] [--ignore cost_by_n.csv]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


def csvs(root):
    return {os.path.relpath(os.path.join(d, f), root) for d, _, fs in os.walk(root) for f in fs if f.endswith(".csv")}


def canon(df):
    """Columns in name order, rows sorted on every column (NaN last), index reset."""
    df = df[sorted(df.columns)]
    return df.sort_values(list(df.columns), na_position="last", kind="stable").reset_index(drop=True)


def diff_cells(a, b, atol):
    """[(row, col, a, b)] for the cells that differ between two canonical frames of the same shape."""
    out = []
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            bad = ~(np.isclose(xv, yv, rtol=0, atol=atol) | (np.isnan(xv) & np.isnan(yv)))
        else:
            bad = ~((x.astype(str) == y.astype(str)) | (x.isna() & y.isna())).to_numpy()
        out += [(int(i), c, x.iloc[i], y.iloc[i]) for i in np.flatnonzero(bad)]
    return out


def compare(root_a, root_b, atol, ignore=()):
    """List of report lines; empty when the trees agree."""
    a, b = csvs(root_a), csvs(root_b)
    lines = [
        f"MISSING {p} (only in {'A' if p in a else 'B'})" for p in sorted(a ^ b) if os.path.basename(p) not in ignore
    ]
    for p in sorted(a & b):
        if os.path.basename(p) in ignore:
            continue
        x, y = pd.read_csv(os.path.join(root_a, p)), pd.read_csv(os.path.join(root_b, p))
        if set(x.columns) != set(y.columns):
            lines.append(
                f"COLUMNS {p}: only A {sorted(set(x.columns) - set(y.columns))}, "
                f"only B {sorted(set(y.columns) - set(x.columns))}"
            )
            continue
        if len(x) != len(y):
            lines.append(f"DIFF {p}: {len(x)} vs {len(y)} rows")
            continue
        cells = diff_cells(canon(x), canon(y), atol)
        if cells:
            shown = "; ".join(f"row {r} {c}: {va!r} vs {vb!r}" for r, c, va, vb in cells[:3])
            lines.append(f"DIFF {p}: {len(cells)} cells ({shown})")
    return lines


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a", help="figures/ tree A (e.g. base)")
    p.add_argument("b", help="figures/ tree B (e.g. refactor)")
    p.add_argument("--atol", type=float, default=1e-12)
    p.add_argument("--ignore", nargs="*", default=[], help="CSV base names to skip (e.g. cache files)")
    a = p.parse_args(argv)
    lines = compare(a.a, a.b, a.atol, set(a.ignore))
    print("\n".join(lines) if lines else f"OK: {len(csvs(a.a))} CSVs value-identical (atol {a.atol:g})")
    sys.exit(1 if lines else 0)


if __name__ == "__main__":
    main()
