"""Rewrite stored rows to decision-D1 row ids, ONCE per results directory (dry run by default).

D1: midian.run.row_id hashes backend_kwargs as the config writes them ("$RTE_DATA/populations/..."), not the path they
expand to, so ids and stored rows no longer depend on where RTE_DATA points. Rows written before D1 store the expanded
path and their rid hashes it. Per directory, every row of rows.csv and rows.d/*.json gets the config's unexpanded
backend_kwargs and the new rid (midian.run.rid_of_row on the rewritten row; rows.d files are renamed to it). Afterwards
rid_of_row, which hashes what a row stores, reproduces the new ids, and the D1 runner resumes the directory.

    RTE_DATA=... PYTHONPATH=. python scripts/checks/migrate_rte_data_ids.py [<results_dir> ...] [--apply]

Without directories: the results directory of every grid whose backend_kwargs mention $RTE_DATA (11 grids). Run --apply
only while no job writes to these directories, together with the D1 runner (master code must not run them afterwards).

Before rewriting, the originals go to <dir>/_premigration_rte_data_v1/ (rows.csv.gz and the rows.d files); the
directory then gets SENTINEL and is never migrated again. rows.csv is streamed: fields other than backend_kwargs and
rid are written back byte-identical (checked on the dry run by re-writing the unchanged file). Refuses, changing
nothing, if: the sentinel exists; a stored rid differs from rid_of_row of its stored row; a stored backend_kwargs is
neither the grid's expanded nor its unexpanded form; two stored rows would get one new rid; the CSV does not
round-trip byte-identically.
"""

import argparse
import csv
import glob
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import time

from midian import run

SENTINEL = ".rte_data_ids_v1"
BACKUP = "_premigration_rte_data_v1"
CSV_TYPES = {
    "n": int,
    "K": int,
    "b": int,
    "Q": int,
    "beta": float,
    "seed": int,
    "collude": lambda s: {"True": True, "False": False}[s],
}


def dependent_grids(cfg):
    """Grids with a backend_kwargs string that $RTE_DATA / env-var expansion changes."""
    out = []
    for g, v in cfg["grids"].items():
        blks = run.blocks(cfg, g)
        if any(run.expand(b.get("backend_kwargs") or {}) != (b.get("backend_kwargs") or {}) for b in blks):
            out.append(g)
    return out


def unexpand_map(cfg, grid):
    """jkey(stored backend_kwargs) -> the config's backend_kwargs, for the expanded and the unexpanded form."""
    out = {}
    for blk in run.blocks(cfg, grid):
        raw = dict(blk.get("backend_kwargs") or {})
        out[run.jkey(run.expand(raw))] = raw
        out[run.jkey(raw)] = raw
    return out


def typed(rec):
    """A rows.csv record as the typed read the runner's rid_of_row expects."""
    return {k: CSV_TYPES[k](v) if k in CSV_TYPES else v for k, v in rec.items()}


class Plan:
    """What one directory's migration would change; `bad` lists the reasons to refuse."""

    def __init__(self, d, bkmap):
        self.d, self.bkmap = d, bkmap
        self.bad, self.new_of = [], {}  # new_of: new rid -> old rid (collision check)
        self.counts = dict.fromkeys(
            ("csv_rows", "csv_rewritten", "csv_no_rid", "files", "files_rewritten", "already_unexpanded"), 0
        )

    def remap(self, row, stored_rid, where):
        """(new backend_kwargs string, new rid) of a stored row; records refusal reasons."""
        if stored_rid and run.rid_of_row(row) != stored_rid:
            self.bad.append(("rid-mismatch", where, stored_rid))
        bk = row.get("backend_kwargs") or "{}"
        raw = self.bkmap.get(run.jkey(json.loads(bk)))
        if raw is None:
            self.bad.append(("unknown-backend_kwargs", where, bk))
            return bk, stored_rid
        new_bk = run.jkey(raw)
        if new_bk == bk:
            self.counts["already_unexpanded"] += 1
        rid = run.rid_of_row({**row, "backend_kwargs": new_bk})
        old = self.new_of.setdefault(rid, stored_rid)
        if old != stored_rid:
            self.bad.append(("collision", where, rid))
        return new_bk, rid


def stream_csv(plan, out_fh):
    """Rewrite rows.csv into out_fh (a text handle, or None: plan only).

    Returns the sha256 of the file re-written UNCHANGED, for the byte-identical round-trip check."""
    path = f"{plan.d}/rows.csv"
    same = hashlib.sha256()
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        head = next(reader)
        i_bk, i_rid = head.index("backend_kwargs"), (head.index("rid") if "rid" in head else None)
        buf = io.StringIO()
        echo = csv.writer(buf, lineterminator="\n")
        out = csv.writer(out_fh, lineterminator="\n") if out_fh else None
        for w in [echo] + ([out] if out else []):
            w.writerow(head)
        for n, rec in enumerate(reader, 1):
            echo.writerow(rec)
            same.update(buf.getvalue().encode())
            buf.seek(0)
            buf.truncate()
            row = typed(dict(zip(head, rec)))
            stored_rid = rec[i_rid] if i_rid is not None else ""
            plan.counts["csv_rows"] += 1
            plan.counts["csv_no_rid"] += not stored_rid
            new_bk, rid = plan.remap(row, stored_rid, f"rows.csv:{n + 1}")
            plan.counts["csv_rewritten"] += new_bk != rec[i_bk]
            if out:
                rec[i_bk] = new_bk
                if i_rid is not None and stored_rid:
                    rec[i_rid] = rid
                out.writerow(rec)
        same.update(buf.getvalue().encode())
    return same.hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def migrate(d, cfg, apply):
    grid = os.path.basename(d.rstrip("/"))
    if os.path.exists(f"{d}/{SENTINEL}"):
        return f"{grid}: already migrated"
    if grid not in cfg["grids"]:
        return f"{grid}: REFUSED, not a grid in the config"
    plan = Plan(d, unexpand_map(cfg, grid))
    csv_path = f"{d}/rows.csv"
    if os.path.exists(csv_path) and stream_csv(plan, None) != file_sha(csv_path):
        plan.bad.append(("csv-roundtrip", "rows.csv", "re-written unchanged CSV differs from the file"))
    files = []
    for f in sorted(glob.glob(f"{d}/rows.d/*.json")):
        with open(f) as fh:
            r = json.load(fh)
        rid0 = os.path.basename(f)[:-5]
        new_bk, rid = plan.remap(r, rid0, f"rows.d/{rid0}")
        plan.counts["files"] += 1
        if new_bk != r.get("backend_kwargs"):
            plan.counts["files_rewritten"] += 1
            files.append((f, {**r, "backend_kwargs": new_bk}, rid))
    if plan.bad:
        kinds = {}
        for k, *_ in plan.bad:
            kinds[k] = kinds.get(k, 0) + 1
        return f"{grid}: REFUSED, nothing changed: {kinds} (e.g. {plan.bad[:3]}) {plan.counts}"
    if not apply:
        return f"{grid}: dry run {plan.counts}"
    # ---- apply: back up originals, then rewrite
    bk = f"{d}/{BACKUP}"
    os.makedirs(f"{bk}/rows.d", exist_ok=True)
    if os.path.exists(csv_path) and plan.counts["csv_rewritten"]:
        with open(csv_path, "rb") as src, gzip.open(f"{bk}/rows.csv.gz", "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 24)
        tmp = f"{csv_path}.tmp{os.getpid()}"
        check = Plan(d, plan.bkmap)
        with open(tmp, "w", newline="") as out:
            stream_csv(check, out)
        os.replace(tmp, csv_path)
    for f, r2, rid in files:
        shutil.copy2(f, f"{bk}/rows.d/{os.path.basename(f)}")
        tmp = f"{d}/rows.d/{rid}.json.tmp{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(r2, fh, default=str)
        os.replace(tmp, f"{d}/rows.d/{rid}.json")
        if os.path.basename(f) != f"{rid}.json":
            os.remove(f)
    with open(f"{d}/{SENTINEL}", "w") as fh:
        json.dump({"migrated": time.strftime("%Y-%m-%dT%H:%M:%S"), **plan.counts}, fh)
    return f"{grid}: MIGRATED {plan.counts}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="*", help="results directories (default: every $RTE_DATA-dependent grid's)")
    p.add_argument("--apply", action="store_true", help="rewrite (default: dry run)")
    a = p.parse_args(argv)
    csv.field_size_limit(sys.maxsize)
    cfg = run.load_config()
    dirs = a.dirs or [f"{run.RTE_DATA}/results/{g}" for g in dependent_grids(cfg)]
    for d in dirs:
        print(migrate(d, cfg, a.apply), flush=True)


if __name__ == "__main__":
    main()
