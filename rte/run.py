"""Grid runner (SPEC §7).   python -m rte.run --grid smoke [--seeds 1-3] [--methods a,b] [--workers N] [--dry-run]

A cell = one point of the CELL axes; a unit = (cell, seed): one World, one paired task stream shared by every
method, the oracle line executed once. Each row is its own JSON file under results/<grid>/rows.d (atomic, resumable);
rows.csv is materialised at the end."""
import argparse, hashlib, json, os, pkgutil, sys, time, traceback
from itertools import product
from multiprocessing import get_context
import numpy as np, yaml
from .budget import Budget
from .methods import load_method
from .world import World
import rte.methods
from rte.methods import keys

RTE_DATA = os.environ.get("RTE_DATA", "/scratch/rte")
CELL = ("backend", "n", "K", "dist", "beta", "liar_select", "collude", "declared_source", "lie_mode", "demand", "b", "Q")
log = lambda m: print(m, file=sys.stderr, flush=True)
jkey = lambda d: json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)


# ------------------------------------------------------------------ config -> units
def seeds(spec):
    """'1-10' | '1,3,5' | 5 -> list[int]"""
    if isinstance(spec, int): return list(range(1, spec + 1))
    out = []
    for p in str(spec).split(","):
        lo, _, hi = p.partition("-"); out += range(int(lo), int(hi or lo) + 1)
    return out


def all_methods(backend):
    """Every algorithmic method file (frameworks are listed explicitly in their own grids: one supervisor call per task);
    LLM-only classes only on the llm backend."""
    names = [m.name for m in pkgutil.iter_modules(rte.methods.__path__) if not m.ispkg and not m.name.startswith("_") and m.name not in ("base", "keys")]
    return [n for n in names if backend == "llm" or not getattr(load_method(n), "requires_llm", False)]


MIDIAN_ABLATIONS = [{"name": "midian", "params": p} for p in ({"verify": False}, {"audit": False}, {"audit": False, "verify": False})]


def method_specs(block):
    ms = block["methods"]
    ms = (all_methods(block["backend"]) + MIDIAN_ABLATIONS + list(block.get("extra") or []) if ms == "all"
          else [x for m in ms for x in (m if isinstance(m, list) else [m])])
    ms = [{"name": m, "params": {}} if isinstance(m, str) else {"name": m["name"], "params": m.get("params") or {}} for m in ms]
    ms = list({m["name"] + jkey(m["params"]): m for m in ms}.values())   # one spec per arm ("all" + extra can repeat one)
    drop = set(block.get("exclude") or [])           # LLM-only methods are dropped off the llm backend too, so a
    llm = block["backend"] == "llm" or bool(block.get("allow_llm_methods"))   # bernoulli mirror of a framework grid skips
                                                     # them, not fails them; `allow_llm_methods: true` opts a non-llm grid
                                                     # back in (the frameworks need a supervisor endpoint, not an llm world)
    return [m for m in ms if m["name"] not in drop and (llm or not getattr(load_method(m["name"]), "requires_llm", False))]


def blocks(cfg, grid):
    """Grid -> list of axis blocks: defaults < mirror_of source < grid < each `blocks:` entry."""
    g = dict(cfg["grids"][grid])
    while "mirror_of" in g: g = {**cfg["grids"][g.pop("mirror_of")], **g}      # mirrors may chain (n100 -> n1000 -> ...)
    base = {**cfg["defaults"], **{k: v for k, v in g.items() if k != "blocks"}}
    out = []
    for b in g.get("blocks") or [{}]:
        blk = {**base, **b}
        blk.update({f: v if isinstance(v, list) else [v] for f, v in blk.items() if f in CELL and f != "backend"})
        out.append(blk)
    return out


def cells(blk):
    axes = [blk[f] if f != "backend" else [blk["backend"]] for f in CELL]
    for combo in product(*axes):
        c = dict(zip(CELL, combo)); c.update(n=int(c["n"]), K=int(c["K"]), b=int(c["b"]), Q=int(c["Q"]), beta=float(c["beta"]))
        c["backend_kwargs"] = {k: (os.path.expandvars(str(v).replace("$RTE_DATA", RTE_DATA)) if isinstance(v, str) else v)
                               for k, v in (blk.get("backend_kwargs") or {}).items()}
        c["churn"] = blk.get("churn")                                   # optional {frac, every}; absent -> None
        cal = c["backend_kwargs"].get("calibrate_from")
        assert not cal or os.path.exists(cal), f"calibrate_from={cal} missing: measure the live S first (bernoulli_scale must be calibrated)"
        yield c


def row_id(cell, method, params, seed):
    churn = f"|churn={jkey(cell['churn'])}" if cell.get("churn") else ""       # only churn cells carry it: old ids unchanged
    return hashlib.blake2b(f"{jkey({f: cell[f] for f in CELL})}|{jkey(cell['backend_kwargs'])}|{method}|{jkey(params)}|{seed}{churn}".encode(),
                           digest_size=16).hexdigest()


def rid_of_row(r) -> str:
    """row_id recomputed from a stored row (CSV or rows.d): lets a CSV written without `rid` be re-keyed."""
    cell = {f: r[f] for f in CELL}
    for k in ("n", "K", "b", "Q"): cell[k] = int(cell[k])
    cell["beta"] = float(cell["beta"])
    for k in list(cell):
        if hasattr(cell[k], "item"): cell[k] = cell[k].item()         # numpy scalars -> python, so jkey matches
    bk = r.get("backend_kwargs", "{}"); cell["backend_kwargs"] = json.loads(bk) if isinstance(bk, str) and bk.startswith("{") else (bk or {})
    ch = r.get("churn", "")
    if isinstance(ch, str) and ch.startswith("{"): cell["churn"] = json.loads(ch)
    p = r.get("params", "{}"); params = json.loads(p) if isinstance(p, str) else (p or {})
    return row_id(cell, r["method"], params, int(r["seed"]))


# ------------------------------------------------------------------ one unit
COMM = ("probes", "reports", "messages", "tasks")      # total communication = these four (CONTRACT)


def metrics(outcomes, liar, build, run, Q, wall_build, wall_route, wall_total):
    late = min(500, max(1, Q // 4)); s = float(np.mean(outcomes))
    blocks = np.asarray(outcomes, float)[:Q // 100 * 100].reshape(-1, 100).mean(1) if Q >= 100 else np.array([s])
    return {"success": s, "success_late": float(np.mean(outcomes[-late:])), "n_late": late,
            "success_by_block": [round(float(x), 4) for x in blocks],        # per 100 tasks (learning / churn curves)
            "misroute_to_liar": float(np.mean(liar)),
            **{f"build_{k}": v for k, v in build.items() if k != "tasks"},
            "build_total_comm": sum(build[k] for k in COMM),
            **{f"{k}_per_task": v / Q for k, v in run.items()},
            "total_comm_per_task": sum(run[k] for k in COMM) / Q,
            "wall_clock_build": wall_build, "wall_clock_per_task": wall_route / Q, "wall_clock_per_task_total": wall_total / Q}


def execute(world, task, ret):
    """int -> execute it. list -> route-to-many: execute all, majority of outcomes (ties -> 0), see DEVIATIONS."""
    agents = [int(a) for a in np.atleast_1d(ret)]
    outs = [world.execute(a, task) for a in agents]
    return (outs[0] if len(outs) == 1 else int(2 * sum(outs) > len(outs))), agents, outs


def churn_due(churn, i):
    return bool(churn) and i > 0 and i % int(churn["every"]) == 0


def oracle_line(world, stream, churn):
    """The oracle re-picks by true S after every churn event and always knows the arrivals."""
    outs, liar = [], []
    for i, t in enumerate(stream):
        if churn_due(churn, i): world.churn(churn["frac"]); world.seen_epoch[:] = world.epoch
        a = world.oracle(t); outs.append(world.execute(a, t)); liar.append(world.liars[a])
    return outs, liar


def run_method(world, stream, spec, b, churn=None):
    m = load_method(spec["name"])(**spec["params"]); view = world.view(m.needs)
    lm, lp = keys.legacy(spec["name"], spec["params"])                  # the pre-rename key seeds the world: reruns reproduce stored rows
    world.reset(lm + jkey(lp)); t0 = time.perf_counter(); m.build(view, Budget(b)); wall_build = time.perf_counter() - t0
    build = world.ledger.snapshot(); world.ledger.reset()
    if build["probes"] > Budget(b).total_probes(world.n, world.K):
        log(f"  [WARNING] {spec['name']}: build spent {build['probes']} probes > budget {Budget(b).total_probes(world.n, world.K)}")
    outcomes, liar, route, t_run, repair, events = [], [], 0.0, time.perf_counter(), dict.fromkeys(build, 0), 0
    if not churn and hasattr(m, "prefetch"):                          # concurrent framework requests (FrameworkMethod.prefetch)
        t = time.perf_counter(); m.prefetch(stream); route += time.perf_counter() - t
    for i, task in enumerate(stream):
        if churn_due(churn, i):                                        # replace agents, let the method repair; cost is
            ids = world.churn(churn["frac"]); before = world.ledger.snapshot(); m.churn(ids, ids); events += 1
            for k, v in world.ledger.diff(before).items(): repair[k] += v   # charged like any other run-time spend
        t = time.perf_counter(); ret = m.fetch(task); route += time.perf_counter() - t
        o, agents, outs = execute(world, task, ret)
        t = time.perf_counter()
        for a, oa in zip(agents, outs): m.observe(task, a, oa)
        route += time.perf_counter() - t
        outcomes.append(o); liar.append(world.liars[agents[0]])
    return {**metrics(outcomes, liar, build, world.ledger.snapshot(), len(stream), wall_build, route, time.perf_counter() - t_run),
            **({f"repair_{k}_per_event": v / events for k, v in repair.items() if k != "tasks"} if events else {}),
            "method_stats": jkey(getattr(m, "stats", {}))}          # e.g. framework picks/fallbacks, LLM-descent parse failures


def run_unit(cell, seed, specs, rows_dir, grid):
    world = World(**{k: cell[k] for k in CELL if k not in ("b", "Q")}, seed=seed, backend_kwargs=cell["backend_kwargs"] or None)
    stream = world.tasks(cell["Q"]); st = world.stats()
    base = {**{f: cell[f] for f in CELL}, "backend_kwargs": jkey(cell["backend_kwargs"]), "seed": seed, "grid": grid,
            "n_agents": world.n, "n_liars": st.pop("n_liars"),
            **{("" if k.startswith("skill_") else "skill_") + k: v for k, v in st.items() if k not in ("n", "K", "dist", "beta", "backend")}}
    base["churn"] = jkey(cell["churn"]) if cell.get("churn") else ""
    oracle, liar = oracle_line(world, stream, cell.get("churn"))
    zero = dict.fromkeys(world.ledger.snapshot(), 0)
    rows = {"oracle": {"method": "oracle", "params": "{}", **metrics(oracle, liar, zero, {**zero, "tasks": len(stream)}, len(stream), 0, 0, 0)}}
    failed = []
    for s in specs:
        try:
            rows[s["name"] + jkey(s["params"])] = {"method": s["name"], "params": jkey(s["params"]), **run_method(world, stream, s, cell["b"], cell.get("churn"))}
        except Exception as e:                       # one bad method must not kill the unit
            failed.append(f"{s['name']}: {type(e).__name__}: {e}"); log(traceback.format_exc())
    for k, r in rows.items():
        rid = row_id(cell, r["method"], json.loads(r["params"]), seed)
        r = {**base, **r, "oracle_success": rows["oracle"]["success"]}; r["regret"] = r["oracle_success"] - r["success"]
        tmp = f"{rows_dir}/{rid}.json.tmp{os.getpid()}"; json.dump(r, open(tmp, "w"), default=str); os.replace(tmp, f"{rows_dir}/{rid}.json")
    return failed


def consolidate(out, prune: bool = False, force: bool = False):
    """Merge rows.d INTO rows.csv (additive, deduplicated on `rid`) and optionally delete the files merged.

    Additive because a sweep can write millions of one-row files: pruning keeps rows.d small so this stays cheap and
    the resume set stays fast to build. `rid` is the rows.d filename, carried into the CSV so resume can read it back.
    Safe under concurrency: the CSV is written atomically and re-read each call, so the worst a race costs is a
    duplicate row, which the dedup removes. Files are unlinked only AFTER the CSV that contains them is in place."""
    import pandas as pd
    csv = f"{out}/rows.csv"
    # A grid whose results dir holds `.merge_owner` has ONE dedicated merger; everyone else must keep its hands off
    # rows.csv. Read-then-write is not atomic, so concurrent consolidates overwrite each other with stale snapshots --
    # and with prune that deletes a row's file after a competing write has already dropped it from the CSV.
    if not force and os.path.exists(f"{out}/.merge_owner"):
        return 0
    names = sorted(f for f in os.listdir(f"{out}/rows.d") if f.endswith(".json"))   # snapshot: later arrivals wait
    frames = []
    if os.path.exists(csv):
        old = pd.read_csv(csv, low_memory=False)
        if "rid" not in old.columns or old.rid.isna().any():     # a CSV rebuilt by pre-guard code has no rid: re-key it
            old["rid"] = [rid_of_row(r) for _, r in old.iterrows()]   # so dedup works and nothing doubles
        frames.append(old)
    if names:
        def _read(f):                            # a concurrent merger may unlink between the listing and the read
            try:
                with open(f"{out}/rows.d/{f}") as fh: return {**json.load(fh), "rid": f[:-5]}
            except (FileNotFoundError, json.JSONDecodeError):
                return None                      # gone, or caught mid-write: it is in the CSV already or will be next pass
        from concurrent.futures import ThreadPoolExecutor   # NFS small-file reads are latency-bound: fan out
        with ThreadPoolExecutor(max_workers=32) as ex:
            got = [r for r in ex.map(_read, names, chunksize=256) if r is not None]
        names = [r["rid"] + ".json" for r in got]        # prune only what was actually read
        if got: frames.append(pd.DataFrame(got))
    if not frames: return 0
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="rid", keep="last") if df.rid.notna().all() else df
    lead = [c for c in (*CELL, "method", "params", "seed") if c in df.columns]
    tmp = f"{csv}.tmp{os.getpid()}"
    df[lead + [c for c in df.columns if c not in lead]].sort_values(lead).to_csv(tmp, index=False)
    os.replace(tmp, csv)
    if prune:
        done = set(df.rid.dropna())
        def _rm(f):
            try: os.unlink(f"{out}/rows.d/{f}")
            except FileNotFoundError: pass
        from concurrent.futures import ThreadPoolExecutor   # NFS unlinks are latency-bound too: ~80/s serial
        with ThreadPoolExecutor(max_workers=32) as ex:
            list(ex.map(_rm, [f for f in names if f[:-5] in done], chunksize=256))
    return len(df)


def _star(args):
    return run_unit(*args)


def main(argv=None):
    p = argparse.ArgumentParser("rte.run")
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "grid.yaml"))
    p.add_argument("--grid", required=True); p.add_argument("--seeds"); p.add_argument("--methods")
    p.add_argument("--workers", type=int); p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", help="cell filter, e.g. beta=0.25,collude=true (splits one grid over many jobs)")
    a = p.parse_args(argv)
    cfg = yaml.safe_load(open(a.config)); out = f"{RTE_DATA}/results/{a.grid}"; rows_dir = f"{out}/rows.d"
    have = {f[:-5] for f in os.listdir(rows_dir)} if os.path.isdir(rows_dir) else set()
    if os.path.exists(f"{out}/rows.csv"):        # rows merged into the CSV and pruned from rows.d are still DONE
        try:
            import pandas as pd
            have |= set(pd.read_csv(f"{out}/rows.csv", usecols=["rid"]).rid.dropna())
        except (ValueError, KeyError): pass      # pre-rid CSV: its rows.d files are still on disk, so `have` is right
    units = []
    for blk in blocks(cfg, a.grid):
        specs = [s for s in method_specs(blk) if not a.methods or s["name"] in a.methods.split(",")]
        only = dict(kv.split("=") for kv in a.only.split(",")) if a.only else {}
        for cell in cells(blk):
            if any(cell[k] != yaml.safe_load(v) for k, v in only.items()):     # yaml-typed: beta=0 == 0.0, collude=true
                continue
            for seed in seeds(a.seeds or blk["seeds"]):
                todo = [s for s in specs if row_id(cell, s["name"], s["params"], seed) not in have]
                if todo: units.append((cell, seed, todo, rows_dir, a.grid))
    llm = any(u[0]["backend"] == "llm" for u in units)
    workers = 1 if llm else (a.workers or min(8, os.cpu_count()))
    log(f"[rte.run] grid={a.grid} units_todo={len(units)} rows_done={len(have)} workers={workers} out={out}")
    if a.dry_run:
        for c, seed, todo, *_ in units[:50]: log(f"  {' '.join(f'{f}={c[f]}' for f in CELL)} seed={seed} methods={[s['name'] for s in todo]}")
        return
    if not os.path.isdir(out):                   # a new directory holds current keys only: mark it (keys.SENTINEL)
        os.makedirs(out); open(f"{out}/{keys.SENTINEL}", "w").write('{"created": "new"}')
    os.makedirs(rows_dir, exist_ok=True); t0 = time.perf_counter(); fails = []
    if workers > 1:
        with get_context("fork").Pool(workers) as pool:
            for i, f in enumerate(pool.imap_unordered(_star, units), 1):
                fails += f; log(f"[{i}/{len(units)}] +1 unit  {time.perf_counter()-t0:.0f}s")
    else:
        for i, u in enumerate(units, 1): fails += run_unit(*u); log(f"[{i}/{len(units)}] {u[0]['dist']} n={u[0]['n']} beta={u[0]['beta']} seed={u[1]}  {time.perf_counter()-t0:.0f}s")
    log(f"[rte.run] {consolidate(out)} rows -> {out}/rows.csv in {time.perf_counter()-t0:.0f}s" + (f"; FAILED: {sorted(set(fails))}" if fails else ""))


if __name__ == "__main__":
    main()
