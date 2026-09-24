"""Golden parity harness (refactor invariant G3): re-run stored rows and demand identical output.

Generalises scripts/equivalence.py from a fixed MIDIAN fingerprint to every stored arm. For each (backend, method,
params) present in the stored rows of the non-LLM backends (bernoulli, replay, routereval) it picks up to --per rows
(cheap cells n <= 10^4 first, then the most recently written rows.csv, smallest n, rid; one row per directory before
a second from the same one: recent grids are the ones today's code should reproduce; the sample moves with file mtimes,
--baseline freezes it), rebuilds the row's cell from the row itself (the rebuilt row_id must equal the stored rid),
re-runs that unit for that one method through rte.run.run_unit into a temp dir, and compares every column except
wall_clock_*: ints, strings and lists by value, floats by exact equality (NaN == NaN; a column missing on one side
must be empty on the other). One exception: a row read from rows.csv (its rows.d file was pruned) is compared with
rtol 1e-12, because rte.run.consolidate re-reads rows.csv with pandas' default float parser, which is not round-trip
exact (0.08399999999999996 -> 0.0839999999999999), and writes the parsed value back on every merge.

Stored rows are only as current as the code that wrote them: a row written before a later method or backend change
differs from today's rerun ("stale"). So the gate for a refactor is --baseline: re-run exactly the rows of a report
made on the base tree and require the fresh rows to be identical to the base tree's fresh rows (exact, every
non-wall_clock column), while still reporting each row against the stored one.

Re-running one method alone is faithful: run_method re-seeds the world from the method's own key (world.reset) and the
oracle line is replayed first exactly as in the original unit, so a unit's rows do not depend on its other methods.

Exempt, and why:
  * fw_* (framework rivals, even on routereval): each task is a request to an LLM supervisor endpoint with no memo that
    reproduces the stored response, so the rows are not reproducible offline.
  * the llm backend: every probe/execute is an LLM call; parity there needs the production memo and live endpoints.
  * cells with n >= --max-n (default 10^6): the 10^6..10^7 scale units take minutes to hours and GBs each.
kNN/MLP rows of EMBED_BATCH_GRIDS were produced with RTE_EMBED_BATCH=1 (one batched MiniLM encode; per-prompt encodes
differ at float level), so the harness sets it for exactly those rows.

Rows are re-run round-robin (first sample of every arm, then the second, ...) until --max-seconds is spent, so a
budgeted run still covers every arm once. Report: JSON with per-row status vs the stored row (equal | diff | error |
skipped; new_cols = equal on every column the stored row has, the rerun only adds columns written by later code), its
diffs, the fresh row, and with --baseline the status/diffs vs the baseline's fresh row. Exit 1 on an
error or a baseline mismatch (and on a stored-row diff with --strict).

    PYTHONPATH=. python scripts/checks/parity.py --out base.json [--max-seconds 3600] [--only-backend replay]
        [--only-method midian,random] [--per 3] [--max-n 1000000]          # on the base tree
    PYTHONPATH=. python scripts/checks/parity.py --out new.json --baseline base.json       # on the refactored tree
"""
import argparse
import json
import math
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from rte import run

BACKENDS = ("bernoulli", "replay", "routereval")
EMBED_BATCH_GRIDS = {"rivals_b_n10k", "rivals_b_n100k", "routereval5k_norep_cal"}
EMBED_METHODS = {"knn_router", "mlp_router"}
INDEX_COLS = ("backend", "n", "method", "params", "rid")


def load_index(results, backends):
    """One frame (dir, line, backend, n, method, params, rid) over every rows.csv of the given backends."""
    frames = []
    for d in sorted(os.listdir(results)):
        f = f"{results}/{d}/rows.csv"
        if d[0] == "_" or not os.path.exists(f):
            continue
        head = pd.read_csv(f, nrows=1)
        if not set(INDEX_COLS) <= set(head.columns) or head.backend[0] not in backends:
            continue
        df = pd.read_csv(f, usecols=list(INDEX_COLS), low_memory=False)
        frames.append(df.assign(dir=d, line=np.arange(len(df)) + 1, mtime=os.path.getmtime(f)))   # line 0: header
    df = pd.concat(frames, ignore_index=True)
    return df[(df.method != "oracle") & df.backend.isin(backends) & df.rid.notna()]


def pick(index, per, max_n):
    """{(backend, method, params): [index rows]}, up to `per` each, in the preference order of the module docstring."""
    out, index = {}, index.assign(big=index.n > 10 ** 4, age=-index.mtime)
    for key, g in index.sort_values(["big", "age", "n", "rid"]).groupby(["backend", "method", "params"], sort=True):
        g = g[g.n < max_n]
        first = g.drop_duplicates("dir")
        chosen = pd.concat([first, g.drop(first.index)]).head(per)
        out[key] = [r for _, r in chosen.iterrows()]
    return out


def stored_row(results, ref):
    """The stored row: its rows.d JSON when still on disk (exact types), else its rows.csv line."""
    j = f"{results}/{ref.dir}/rows.d/{ref.rid}.json"
    if os.path.exists(j):
        with open(j) as fh:
            return {**json.load(fh), "rid": ref.rid}, "rows.d"
    line = int(ref.line)
    df = pd.read_csv(f"{results}/{ref.dir}/rows.csv", skiprows=lambda i: i not in (0, line),
                     low_memory=False, float_precision="round_trip")
    row = df.iloc[0].to_dict()
    assert row["rid"] == ref.rid, f"{ref.dir}: line {line} holds {row['rid']}, expected {ref.rid}"
    return row, "rows.csv"


def cell_of(row):
    """The runner's cell dict rebuilt from a stored row (same casting as rte.run.rid_of_row)."""
    cell = {f: row[f].item() if hasattr(row[f], "item") else row[f] for f in run.CELL}
    for k in ("n", "K", "b", "Q"):
        cell[k] = int(cell[k])
    cell["beta"] = float(cell["beta"])
    bk = row.get("backend_kwargs", "{}")
    cell["backend_kwargs"] = json.loads(bk) if isinstance(bk, str) and bk.startswith("{") else {}
    ch = row.get("churn", "")
    cell["churn"] = json.loads(ch) if isinstance(ch, str) and ch.startswith("{") else None
    return cell


def norm(v):
    """Comparable form: missing -> None, lists -> str (as rows.csv stores them), numbers -> float, else str."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or v == "":
        return None
    if isinstance(v, (list, dict)):
        return str(v)
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    return str(v)


def compare(stored, fresh, rtol=0.0):
    """[{col, stored, rerun}] for every non-wall_clock column that differs (floats within rtol count as equal)."""
    diffs = []
    for c in sorted(set(stored) | set(fresh)):
        if c.startswith("wall_clock_") or c == "rid":
            continue
        a, b = norm(stored.get(c)), norm(fresh.get(c))
        if a != b and not (rtol and isinstance(a, float) and isinstance(b, float) and math.isclose(a, b, rel_tol=rtol)):
            diffs.append({"col": c, "stored": repr(stored.get(c)), "rerun": repr(fresh.get(c))})
    return diffs


def rerun(results, ref, base=None):
    """Re-run one stored row; returns a report entry (compared with `base`, a baseline entry, when given)."""
    row, src = stored_row(results, ref)
    params = json.loads(row["params"]) if isinstance(row["params"], str) else row["params"]
    cell, seed = cell_of(row), int(row["seed"])
    rid = run.row_id(cell, row["method"], params, seed)
    entry = {"dir": ref.dir, "rid": ref.rid, "line": int(ref.line), "source": src, "backend": cell["backend"],
             "method": row["method"], "params": row["params"], "n": cell["n"], "Q": cell["Q"], "seed": seed}
    if rid != ref.rid:
        return {**entry, "status": "error", "error": f"rebuilt cell gives rid {rid}"}
    batch = row["method"] in EMBED_METHODS and row["grid"] in EMBED_BATCH_GRIDS
    old = os.environ.pop("RTE_EMBED_BATCH", None)
    if batch:
        os.environ["RTE_EMBED_BATCH"] = "1"
    t0 = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="rte_parity_") as tmp:
            failed = run.run_unit(cell, seed, [{"name": row["method"], "params": params}], tmp, row["grid"])
            if failed:
                return {**entry, "status": "error", "error": "; ".join(failed)}
            with open(f"{tmp}/{rid}.json") as fh:
                fresh = json.load(fh)
    finally:
        os.environ.pop("RTE_EMBED_BATCH", None)
        if old is not None:
            os.environ["RTE_EMBED_BATCH"] = old
    fresh = {k: v for k, v in fresh.items() if not k.startswith("wall_clock_")}
    diffs = compare(row, fresh, rtol=1e-12 if src == "rows.csv" else 0.0)
    new_cols = all(norm(row.get(d["col"])) is None for d in diffs)       # only columns the stored row predates
    status = "equal" if not diffs else "new_cols" if new_cols else "diff"
    entry.update(embed_batch=batch, seconds=round(time.perf_counter() - t0, 2), status=status, diffs=diffs, fresh=fresh)
    if base is not None:
        bd = compare(base["fresh"], fresh) if "fresh" in base else [{"col": "*", "stored": base["status"], "rerun": ""}]
        entry.update(baseline_status="diff" if bd else "equal", baseline_diffs=bd)
    return entry


def stub(ref, status, error):
    """A report entry for a row that was not compared."""
    return {"dir": ref.dir, "rid": ref.rid, "line": int(ref.line), "n": int(ref.n), "backend": ref.backend,
            "method": ref.method, "params": ref.params, "status": status, "error": error}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=f"{run.RTE_DATA}/results")
    p.add_argument("--out", required=True, help="JSON report path")
    p.add_argument("--per", type=int, default=3, help="stored rows re-run per (backend, method, params)")
    p.add_argument("--max-seconds", type=float, default=float("inf"), help="stop starting new reruns after this")
    p.add_argument("--max-n", type=int, default=10 ** 6, help="skip cells with n >= this")
    p.add_argument("--only-backend", help="comma-separated subset of " + ",".join(BACKENDS))
    p.add_argument("--only-method", help="comma-separated method names")
    p.add_argument("--baseline", help="re-run exactly this report's rows and compare with its fresh rows")
    p.add_argument("--strict", action="store_true", help="exit 1 on a diff vs a stored row too")
    a = p.parse_args(argv)
    backends = tuple(a.only_backend.split(",")) if a.only_backend else BACKENDS
    t0 = time.perf_counter()
    base = {}
    if a.baseline:
        with open(a.baseline) as fh:
            base = {e["rid"]: e for e in json.load(fh)["rows"] if "line" in e}
        index = pd.DataFrame([{k: e[k] for k in ("dir", "line", "rid", "backend", "n", "method", "params")}
                              for e in base.values()]).assign(mtime=0.0)
        backends = tuple(sorted(set(index.backend)))
    else:
        index = load_index(a.results, backends)
        index = index[~index.method.str.startswith("fw_")]
    if a.only_method:
        index = index[index.method.isin(a.only_method.split(","))]
    picks = pick(index, len(index) if a.baseline else a.per, a.max_n)
    print(f"[parity] index {len(index):,} rows, {len(picks)} arms, {time.perf_counter() - t0:.0f}s", file=sys.stderr)
    depth = max(map(len, picks.values()), default=0)
    order = [(k, i) for i in range(depth) for k in picks if i < len(picks[k])]     # round-robin over arms
    report = []
    for j, (key, i) in enumerate(order, 1):
        ref = picks[key][i]
        if time.perf_counter() - t0 > a.max_seconds:
            report.append(stub(ref, "skipped", "time budget"))
            continue
        try:
            e = rerun(a.results, ref, base.get(ref.rid) if a.baseline else None)
        except Exception as ex:                                  # report and keep going: one arm must not stop the rest
            e = stub(ref, "error", f"{type(ex).__name__}: {ex}")
        report.append(e)
        print(f"[{j}/{len(order)}] {e['status']:6} {e.get('baseline_status', '')} {key[0]} {key[1]} {key[2]} "
              f"n={ref.n} {ref.dir} {e.get('seconds', '')}s " + " ".join(d["col"] for d in e.get("diffs", [])),
              file=sys.stderr, flush=True)
    status = pd.Series([e["status"] for e in report]).value_counts().to_dict()
    vs_base = pd.Series([e.get("baseline_status", "-") for e in report]).value_counts().to_dict()
    uncovered = sorted(" ".join(k) for k, v in picks.items() if not v)
    summary = {"arms": len(picks), "arms_rerun": len({(e["backend"], e["method"], e["params"]) for e in report
                                                      if e["status"] in ("equal", "diff")}),
               "rows": len(report), "status": status, "baseline_status": vs_base,
               "seconds": round(time.perf_counter() - t0, 1),
               "uncovered_arms_n_ge_max_n": uncovered, "max_n": a.max_n, "per": a.per, "backends": list(backends),
               "exempt": "fw_* methods and the llm backend (no offline-reproducible LLM responses)"}
    with open(a.out, "w") as fh:
        json.dump({"summary": summary, "rows": report}, fh, indent=1, default=str)
    print(json.dumps(summary, indent=1), file=sys.stderr)
    sys.exit(1 if status.get("error") or vs_base.get("diff") or (a.strict and status.get("diff")) else 0)


if __name__ == "__main__":
    main()
