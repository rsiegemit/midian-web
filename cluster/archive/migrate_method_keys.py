"""Rewrite stored rows to the current MIDIAN keys, ONCE per results directory (rte/methods/keys.py has the mapping).
    RTE_DATA=... python scripts/ops/migrate_method_keys.py <results_dir> [...] [--apply]      (default: dry run)
Per directory: every row of rows.csv and rows.d/*.json whose (method, params) changes under keys.to_new gets the new key and
a new rid (rte.run.rid_of_row on the rewritten row; rows.d files are renamed to it); withdrawn variants (midian_sh /
midian_sha) are removed. Every changed or removed original goes to <dir>/_premigration_v2/ first (rows_changed.csv.gz
and the original rows.d files), so the step is reversible. Unchanged values are written back byte-identical (the CSV is
read as text; rids are computed from a typed read, as the runner computes them). The directory then gets keys.SENTINEL
and is never migrated again: a second pass would turn the new full-method `midian` rows into the old undefended ones.
Refuses (and changes nothing) if: the sentinel exists, a stored rid does not match the rid recomputed from its OLD key,
or two rows would share a new rid."""
import glob, gzip, json, os, shutil, sys, time
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from rte.methods import keys
from rte.run import jkey, rid_of_row

load = lambda p: json.loads(p) if isinstance(p, str) and p.startswith("{") else (p if isinstance(p, dict) else {})


def touched(m, p):
    return m in keys.OLD or m == "midian" or (str(m).startswith("fw_") and "midian" in str(p))


def migrate(d, apply):
    tag = os.path.basename(d.rstrip("/"))
    if os.path.exists(f"{d}/{keys.SENTINEL}"): return f"{tag}: already migrated"
    rep = {"csv_changed": 0, "csv_withdrawn": 0, "files_changed": 0, "files_withdrawn": 0}
    new_rids, bad = {}, []
    # ---- rows.csv
    csv = f"{d}/rows.csv"; df = None
    if os.path.exists(csv):
        df = pd.read_csv(csv, dtype=str, keep_default_na=False)
        if "method" in df:
            hit = [i for i, (m, p) in enumerate(zip(df.method, df.get("params", pd.Series(["{}"] * len(df))))) if touched(m, p)]
            if hit:
                typed = pd.read_csv(csv, low_memory=False).iloc[hit]          # typed values: the rid the runner computed
                drop, rows_new = [], {}
                for i, (_, r) in zip(hit, typed.iterrows()):
                    r = r.to_dict(); r["params"] = r.get("params") if isinstance(r.get("params"), str) else "{}"
                    if "rid" in r and isinstance(r["rid"], str) and rid_of_row(r) != r["rid"]: bad.append(("csv", r["rid"]))
                    n = keys.to_new(r["method"], load(r["params"]))
                    if n is None: drop.append(i); continue
                    if (n[0], jkey(n[1])) == (r["method"], jkey(load(r["params"]))): continue
                    r2 = {**r, "method": n[0], "params": jkey(n[1])}; rid = rid_of_row(r2)
                    if rid in new_rids: bad.append(("collision", rid)); continue
                    new_rids[rid] = ("csv", i); rows_new[i] = (n[0], jkey(n[1]), rid)
                changed = sorted(rows_new) + drop
                rep["csv_changed"], rep["csv_withdrawn"] = len(rows_new), len(drop)
                keep_rids = set(df.rid) - {df.rid.iloc[i] for i in changed} if "rid" in df else set()
                bad += [("collision-with-kept", r) for r in new_rids if r in keep_rids]
    # ---- rows.d
    plan = []
    for f in sorted(glob.glob(f"{d}/rows.d/*.json")):
        r = json.load(open(f))
        if not touched(r.get("method"), r.get("params")): continue
        rid0 = os.path.basename(f)[:-5]
        if rid_of_row(r) != rid0: bad.append(("file", rid0))
        n = keys.to_new(r["method"], load(r.get("params", "{}")))
        if n is None: plan.append((f, None, None)); continue
        if (n[0], jkey(n[1])) == (r["method"], jkey(load(r.get("params", "{}")))): continue
        pstr = jkey(n[1]) if isinstance(r.get("params"), str) or "params" not in r else n[1]
        r2 = {**r, "method": n[0], "params": pstr}; rid = rid_of_row(r2)
        if rid in new_rids and new_rids[rid] != ("file", rid0) and new_rids[rid][0] == "file": bad.append(("collision", rid))
        new_rids.setdefault(rid, ("file", rid0)); plan.append((f, r2, rid))
    rep["files_changed"] = sum(1 for _, r2, _ in plan if r2 is not None); rep["files_withdrawn"] = sum(1 for _, r2, _ in plan if r2 is None)
    if bad:
        kinds = pd.Series([k for k, _ in bad]).value_counts().to_dict()
        return f"{tag}: REFUSED, nothing changed: {kinds} (e.g. {bad[:3]})"
    if not apply: return f"{tag}: dry run {rep}"
    # ---- apply: back up originals, then rewrite
    bk = f"{d}/_premigration_v2"; os.makedirs(f"{bk}/rows.d", exist_ok=True)
    if df is not None and (rep["csv_changed"] or rep["csv_withdrawn"]):
        with gzip.open(f"{bk}/rows_changed.csv.gz", "wt") as fh: df.iloc[changed].to_csv(fh, index=False)
        for i, (m, p, rid) in rows_new.items():
            df.iat[i, df.columns.get_loc("method")] = m; df.iat[i, df.columns.get_loc("params")] = p
            if "rid" in df: df.iat[i, df.columns.get_loc("rid")] = rid
        df = df.drop(df.index[drop])
        tmp = f"{csv}.tmp{os.getpid()}"; df.to_csv(tmp, index=False); os.replace(tmp, csv)
    for f, r2, rid in plan:
        shutil.copy2(f, f"{bk}/rows.d/{os.path.basename(f)}")
        if r2 is not None:
            tmp = f"{d}/rows.d/{rid}.json.tmp{os.getpid()}"; json.dump(r2, open(tmp, "w"), default=str); os.replace(tmp, f"{d}/rows.d/{rid}.json")
        if r2 is None or os.path.basename(f) != f"{rid}.json": os.remove(f)
    json.dump({"migrated": time.strftime("%Y-%m-%dT%H:%M:%S"), **rep}, open(f"{d}/{keys.SENTINEL}", "w"))
    return f"{tag}: MIGRATED {rep}"


if __name__ == "__main__":
    apply = "--apply" in sys.argv
    for d in [a for a in sys.argv[1:] if not a.startswith("--")]:
        print(migrate(d, apply), flush=True)
