"""Grid enumeration fingerprint (refactor invariant G1).

For every grid in the config, enumerate blocks x cells x seeds x method_specs exactly as `python -m midian.run` does and
print one TSV line: grid, units, method_rows, sha256 of the sorted row ids (first 16 hex chars), rte_data_dep.
`--dry-run` of the runner is no substitute: it counts only UNFINISHED units.

rte_data_dep = 1 when some backend_kwargs string changes under $RTE_DATA / env-var expansion, i.e. the grid's row ids
depend on where RTE_DATA points (decision D1 changes exactly these ids).

    PYTHONPATH=. python scripts/checks/grid_fingerprint.py [--config configs/grids] [--grids a,b] > fp.tsv
    PYTHONPATH=. python scripts/checks/grid_fingerprint.py --compare tests/golden/grid_fingerprints.tsv
"""

import argparse
import hashlib
import os
import sys

from midian import run

COLUMNS = ("grid", "units", "method_rows", "sha256_16", "rte_data_dep")


def env_dependent(blk) -> bool:
    """True if any backend_kwargs string of the block is rewritten by the runner's expansion."""
    raw = [v for v in (blk.get("backend_kwargs") or {}).values() if isinstance(v, str)]
    return any(os.path.expandvars(v.replace("$RTE_DATA", run.RTE_DATA)) != v for v in raw)


def fingerprint(cfg, grid) -> dict:
    """units, method_rows, id digest and RTE_DATA dependence of one grid (oracle rows are not counted)."""
    ids, units, dep = [], 0, False
    for blk in run.blocks(cfg, grid):
        specs = run.method_specs(blk)
        dep |= env_dependent(blk)
        for cell in run.cells(blk):
            for seed in run.seeds(blk["seeds"]):
                units += 1
                ids += [run.row_id(cell, s["name"], s["params"], seed) for s in specs]
    digest = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]
    return {"grid": grid, "units": units, "method_rows": len(ids), "sha256_16": digest, "rte_data_dep": int(dep)}


def grid_names(cfg):
    """Every real grid (non-dict entries such as fill_arms / cohort_arms are YAML anchor holders, not grids)."""
    return [g for g, v in cfg["grids"].items() if isinstance(v, dict)]


def read_tsv(path) -> dict:
    with open(path) as fh:
        lines = [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]
    head, body = lines[0], lines[1:]
    return {r[0]: dict(zip(head, r)) for r in body}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--config", help="a config directory or single YAML file (default: midian.run.load_config's configs/grids)"
    )
    p.add_argument("--grids", help="comma-separated subset (default: all)")
    p.add_argument("--compare", metavar="TSV", help="diff against a golden TSV; exit 1 on any difference")
    a = p.parse_args(argv)
    cfg = run.load_config(a.config)
    names = a.grids.split(",") if a.grids else grid_names(cfg)
    rows = [fingerprint(cfg, g) for g in names]
    print("\t".join(COLUMNS))
    for r in rows:
        print("\t".join(str(r[c]) for c in COLUMNS), flush=True)
    dep = [r["grid"] for r in rows if r["rte_data_dep"]]
    print(
        f"# {len(rows)} grids, {sum(r['units'] for r in rows):,} units, {sum(r['method_rows'] for r in rows):,} "
        f"method rows; {len(dep)} RTE_DATA-dependent: {','.join(dep)}",
        file=sys.stderr,
    )
    if a.compare:
        gold = read_tsv(a.compare)
        bad = [r["grid"] for r in rows if any(str(r[c]) != gold.get(r["grid"], {}).get(c) for c in COLUMNS)]
        extra = sorted(set(gold) - set(names)) if not a.grids else []
        for g in bad:
            print(f"DIFF {g}: golden={gold.get(g)}", file=sys.stderr)
        for g in extra:
            print(f"MISSING {g} (in golden, not in config)", file=sys.stderr)
        print(f"# compare: {len(bad)} differing, {len(extra)} missing", file=sys.stderr)
        sys.exit(1 if bad or extra else 0)


if __name__ == "__main__":
    main()
