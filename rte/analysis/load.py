"""Rows of the result grids: read, key-normalise, label, group.

`load(grids)` reads `$RTE_DATA/results/<grid>/rows.csv` (after consolidating rows.d), translates old MIDIAN keys
(rte.methods.keys) and hands the frame to `prepare`, which adds the arm label (method + params, ALIAS'd to one name per
arm), the method group (from the method's `needs`), the framework accountings and the total-communication columns.
"""
import json, os, sys
import numpy as np, pandas as pd
from .. import config, run as _run
from ..methods import keys

RTE_DATA, consolidate = str(config.RTE_DATA), _run.consolidate   # str: scripts concatenate it
CELL_COLS = _run.CELL
REF, FLOOR = "midian_wo_defenses", "WITHIN_FLOOR"   # the reference arm: pre-rename label "midian" (plain)
FLAT, FLAT_ON = "flat_probe_argmax_frozen", "flat_probe_argmax_online"
ALIAS = {"flat_probe_argmax": FLAT, "flat_probe_argmax[online=True]": FLAT_ON, "knn_router[online=True]": "knn_router_online",        # one name per arm everywhere
         "midian[verify=False]": "midian_wo_verify", "midian[audit=False]": "midian_wo_audit",
         "midian[audit=False,verify=False]": "midian_wo_defenses", "midian[audit=False,r=5]": "midian_wo_audit_r5",
         "sequential_halving[peer_reported=True]": "sequential_halving_peer", "midian[audit=False,stratify=True,verify=False]": "midian_stratified",
         "sequential_halving[churn_mode=rebuild,peer_reported=True]": "sequential_halving_peer_rebuild",
         "sequential_halving[churn_mode=stale,peer_reported=True]": "sequential_halving_peer_stale"}
STATS = ("success_strict", "fallback_rate")               # framework accountings carried inside method_stats (0.2)
COST = ["comparisons_per_task", "hops_per_task", "messages_per_task", "total_comm_per_task"]   # no wall-clock: memo-mixed
BUILD = ["build_probes", "build_reports", "build_messages", "build_total_comm"]
log = lambda m: print(m, file=sys.stderr, flush=True)
cells = lambda df: [c for c in CELL_COLS if c in df.columns]
fmt = lambda d: ", ".join(f"{k} {v:+.3f}" for k, v in sorted(d.items()))
PLAIN = lambda df: df[(df.method == "midian") & df.params.str.contains('"audit":false') & df.params.str.contains('"verify":false')]   # every MIDIAN w/o defenses row (any r, delta, ...)


def group_of(name):
    """framework | midian | ceiling | floor | declared | verified_decentral | verified_central."""
    if name.startswith("fw_"): return "framework"
    if name.startswith("midian"): return "midian"
    if name in ("oracle", "random"): return "ceiling" if name == "oracle" else "floor"
    try:
        from ..methods import load_method
        needs = frozenset(load_method(name).needs)
    except Exception: return "unknown"                     # optional dep or LLM-only file
    if not needs & {"probe", "reports"}: return "declared"
    return "verified_decentral" if needs & {"reports", "bus"} else "verified_central"
def reads_declared(name):
    """True for every method whose `needs` include the declared channel (frameworks read self-descriptions)."""
    if name.startswith("fw_"): return True
    try:
        from ..methods import load_method
        return "declared" in load_method(name).needs
    except Exception: return False
def load(grids):
    """Rows of every grid, plus label, group, and the total-communication columns when absent."""
    frames = []
    for g in grids:
        d = f"{RTE_DATA}/results/{g}"
        if os.path.isdir(f"{d}/rows.d"): consolidate(d)    # refresh rows.csv from the per-row files
        if os.path.exists(f"{d}/rows.csv"): frames.append(keys.normalize(pd.read_csv(f"{d}/rows.csv"), d))   # old MIDIAN keys -> current
        else: log(f"[analyze] no rows for grid {g!r} at {d}")
    if not frames: raise SystemExit(f"no rows found for grids {grids}")
    return prepare(pd.concat(frames, ignore_index=True))
def prepare(df):
    """Label, group, method_stats -> columns, churn fraction, total-communication columns when absent."""
    df["params"] = df.params.fillna("{}")
    stats = [json.loads(x) if isinstance(x, str) and x.startswith("{") else {} for x in df.get("method_stats", pd.Series([""] * len(df)))]
    for c in STATS: df[c] = [d.get(c, np.nan) for d in stats]
    ch = df.get("churn", pd.Series([""] * len(df))).fillna("")
    df["churn_frac"] = [json.loads(str(x).replace("'", '"'))["frac"] if str(x).startswith("{") else 0.0 for x in ch]
    short = lambda p: ",".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in sorted(json.loads(p).items()))
    df["label"] = [ALIAS.get(l, l) for l in (m if p == "{}" else f"{m}[{short(p)}]" for m, p in zip(df.method, df.params))]
    df["group"] = df.method.map(group_of)
    legacy = (df.method.str.startswith("midian") & ~df.method.isin(["midian_llm_descent"]) & np.array([d.get("observe_charged") is None for d in stats])
              & ~df.params.str.contains('"online": ?false', regex=True)).to_numpy()
    if legacy.any():                                          # observe-time path recompute was uncharged before 2026-09-03 15:20: r*depth comparisons + depth messages per task
        r = np.array([json.loads(p).get("r", 10) for p in df.params]); depth = np.ceil(np.log(df.n.to_numpy()) / np.log(r)).astype(int)
        df.loc[legacy, "comparisons_per_task"] = df.loc[legacy, "comparisons_per_task"].to_numpy() + (r * depth)[legacy]
        df.loc[legacy, "messages_per_task"] = df.loc[legacy, "messages_per_task"].to_numpy() + depth[legacy]
        if "total_comm_per_task" in df: df.loc[legacy, "total_comm_per_task"] = df.loc[legacy, "total_comm_per_task"].to_numpy() + depth[legacy]
    per = ["probes_per_task", "reports_per_task", "messages_per_task", "tasks_per_task"]
    for c in per + BUILD[:3]:                              # pre-rewrite CSVs lack some counters
        if c not in df: df[c] = 0.0
    if "total_comm_per_task" not in df: df["total_comm_per_task"] = df[per].sum(axis=1)
    if "build_total_comm" not in df: df["build_total_comm"] = df[BUILD[:3]].sum(axis=1)
    return df
