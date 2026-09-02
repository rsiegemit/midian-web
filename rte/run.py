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

RTE_DATA = os.environ.get("RTE_DATA", "/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte")
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
    names = [m.name for m in pkgutil.iter_modules(rte.methods.__path__) if not m.ispkg and not m.name.startswith("_") and m.name != "base"]
    return [n for n in names if backend == "llm" or not getattr(load_method(n), "requires_llm", False)]


def method_specs(block):
    ms = block["methods"]
    ms = all_methods(block["backend"]) if ms == "all" else ms
    ms = [{"name": m, "params": {}} if isinstance(m, str) else {"name": m["name"], "params": m.get("params") or {}} for m in ms]
    drop = set(block.get("exclude") or [])           # LLM-only methods are dropped off the llm backend too, so a
    llm = block["backend"] == "llm"                   # bernoulli mirror of a framework grid skips them, not fails them
    return [m for m in ms if m["name"] not in drop and (llm or not getattr(load_method(m["name"]), "requires_llm", False))]


def blocks(cfg, grid):
    """Grid -> list of axis blocks: defaults < mirror_of source < grid < each `blocks:` entry."""
    g = dict(cfg["grids"][grid])
    if "mirror_of" in g: g = {**cfg["grids"][g.pop("mirror_of")], **g}
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
        if not os.path.exists(str(c["backend_kwargs"].get("calibrate_from", ""))):
            if c["backend_kwargs"].pop("calibrate_from", None):
                log(f"  [WARNING] calibrate_from is missing -- sampling dist={c['dist']!r} instead. "
                    f"These points are NOT calibrated to a measured S; label them so.")
        yield c


def row_id(cell, method, params, seed):
    return hashlib.blake2b(f"{jkey({f: cell[f] for f in CELL})}|{jkey(cell['backend_kwargs'])}|{method}|{jkey(params)}|{seed}".encode(),
                           digest_size=16).hexdigest()


# ------------------------------------------------------------------ one unit
COMM = ("probes", "reports", "messages", "tasks")      # total communication = these four (CONTRACT)


def metrics(outcomes, liar, build, run, Q, wall_build, wall_route, wall_total):
    late = min(500, max(1, Q // 4)); s = float(np.mean(outcomes))
    return {"success": s, "success_late": float(np.mean(outcomes[-late:])), "n_late": late,
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


def run_method(world, stream, spec, b):
    m = load_method(spec["name"])(**spec["params"]); view = world.view(m.needs)
    world.reset(spec["name"] + jkey(spec["params"])); t0 = time.perf_counter(); m.build(view, Budget(b)); wall_build = time.perf_counter() - t0
    build = world.ledger.snapshot(); world.ledger.reset()
    if build["probes"] > Budget(b).total_probes(world.n, world.K):
        log(f"  [WARNING] {spec['name']}: build spent {build['probes']} probes > budget {Budget(b).total_probes(world.n, world.K)}")
    outcomes, liar, route, t_run = [], [], 0.0, time.perf_counter()
    for task in stream:
        t = time.perf_counter(); ret = m.fetch(task); route += time.perf_counter() - t
        o, agents, outs = execute(world, task, ret)
        t = time.perf_counter()
        for a, oa in zip(agents, outs): m.observe(task, a, oa)
        route += time.perf_counter() - t
        outcomes.append(o); liar.append(world.liars[agents[0]])
    return metrics(outcomes, liar, build, world.ledger.snapshot(), len(stream), wall_build, route, time.perf_counter() - t_run)


def run_unit(cell, seed, specs, rows_dir, grid):
    world = World(**{k: cell[k] for k in CELL if k not in ("b", "Q")}, seed=seed, backend_kwargs=cell["backend_kwargs"] or None)
    stream = world.tasks(cell["Q"]); st = world.stats()
    base = {**{f: cell[f] for f in CELL}, "backend_kwargs": jkey(cell["backend_kwargs"]), "seed": seed, "grid": grid,
            "n_agents": world.n, "n_liars": st.pop("n_liars"),
            **{("" if k.startswith("skill_") else "skill_") + k: v for k, v in st.items() if k not in ("n", "K", "dist", "beta", "backend")}}
    picks = world.oracle_all()[[t.family for t in stream]]
    oracle = [world.execute(int(a), t) for a, t in zip(picks, stream)]
    zero = dict.fromkeys(world.ledger.snapshot(), 0)
    rows = {"oracle": {"method": "oracle", "params": "{}", **metrics(oracle, world.liars[picks], zero, {**zero, "tasks": len(stream)}, len(stream), 0, 0, 0)}}
    failed = []
    for s in specs:
        try:
            rows[s["name"] + jkey(s["params"])] = {"method": s["name"], "params": jkey(s["params"]), **run_method(world, stream, s, cell["b"])}
        except Exception as e:                       # one bad method must not kill the unit
            failed.append(f"{s['name']}: {type(e).__name__}: {e}"); log(traceback.format_exc())
    for k, r in rows.items():
        rid = row_id(cell, r["method"], json.loads(r["params"]), seed)
        r = {**base, **r, "oracle_success": rows["oracle"]["success"]}; r["regret"] = r["oracle_success"] - r["success"]
        tmp = f"{rows_dir}/{rid}.json.tmp{os.getpid()}"; json.dump(r, open(tmp, "w"), default=str); os.replace(tmp, f"{rows_dir}/{rid}.json")
    return failed


def consolidate(out):
    import pandas as pd
    df = pd.DataFrame([json.load(open(f"{out}/rows.d/{f}")) for f in sorted(os.listdir(f"{out}/rows.d")) if f.endswith(".json")])
    lead = [c for c in (*CELL, "method", "params", "seed") if c in df.columns]
    tmp = f"{out}/rows.csv.tmp{os.getpid()}"
    df[lead + [c for c in df.columns if c not in lead]].sort_values(lead).to_csv(tmp, index=False)
    os.replace(tmp, f"{out}/rows.csv"); return len(df)


def _star(args):
    return run_unit(*args)


def main(argv=None):
    p = argparse.ArgumentParser("rte.run")
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "grid.yaml"))
    p.add_argument("--grid", required=True); p.add_argument("--seeds"); p.add_argument("--methods")
    p.add_argument("--workers", type=int); p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    cfg = yaml.safe_load(open(a.config)); out = f"{RTE_DATA}/results/{a.grid}"; rows_dir = f"{out}/rows.d"
    have = {f[:-5] for f in os.listdir(rows_dir)} if os.path.isdir(rows_dir) else set()
    units = []
    for blk in blocks(cfg, a.grid):
        specs = [s for s in method_specs(blk) if not a.methods or s["name"] in a.methods.split(",")]
        for cell in cells(blk):
            for seed in seeds(a.seeds or blk["seeds"]):
                todo = [s for s in specs if row_id(cell, s["name"], s["params"], seed) not in have]
                if todo: units.append((cell, seed, todo, rows_dir, a.grid))
    llm = any(u[0]["backend"] == "llm" for u in units)
    workers = 1 if llm else (a.workers or min(8, os.cpu_count()))
    log(f"[rte.run] grid={a.grid} units_todo={len(units)} rows_done={len(have)} workers={workers} out={out}")
    if a.dry_run:
        for c, seed, todo, *_ in units[:50]: log(f"  {' '.join(f'{f}={c[f]}' for f in CELL)} seed={seed} methods={[s['name'] for s in todo]}")
        return
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
