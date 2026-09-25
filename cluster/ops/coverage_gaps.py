"""Rows the condensed figures need that are neither written nor in flight -- the SILENTLY missing ones.
    python scripts/ops/coverage_gaps.py [--plan-out gaps.tsv]
For every grid behind A / B / E-H, restricted to the figure cells (specialist on live; beta 0 and the beta 0.5 low-skill
cartel; RouterEval strong_to_weak + the 5,000 leaderboard), every expected row id (rte.run's own loader) is checked:
  done      its row is in rows.d / rows.csv
  inflight  a live SLURM job holds a unit whose filter covers it (scripts/ops/inflight.py)
  planned   a line of the focus pack plan not yet handed to a pack covers it
  GAP       none of the above -> listed, and written as pack-plan lines (grid, method, only, seed) with --plan-out."""
from __future__ import annotations
import argparse, glob, os, re, sys
from collections import defaultdict
import pandas as pd, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rte.run import blocks, cells, method_specs, row_id, seeds, RTE_DATA
from inflight import inflight, queue

L = f"{RTE_DATA}/logs"; R = f"{RTE_DATA}/results"
FIG_ARMS = {"midian", "flat_probe_argmax", "declared_argmax", "random", "knn_router", "mlp_router", "flat_nsw_router",
            "cluster_head_router", "disrouter_cascade", "ucb_per_family", "thompson_per_family", "warm_start_bandit", "linucb_honest",
            "trueskill_per_family"}                          # what A / B draw (condensed_figs ARMS + POOLS); E-H: the fw_* arms
MIDIAN_DRAWN = lambda p: ("audit" not in p and "verify" not in p) or (p.get("audit") is False and p.get("verify") is False and p.get("r", 10) == 10)   # MIDIAN; w/o defenses at r = 10
DRAWN = lambda name, params: name.startswith("fw_") or (name in FIG_ARMS and (name != "midian" or MIDIAN_DRAWN(params))
                                                        and (name != "flat_probe_argmax" or params.get("online")))
FOCUS = lambda c: (c["beta"] == 0 or (c["beta"] == 0.5 and c["liar_select"] == "low_skill_first")) and \
                  (c["backend"] != "llm" or c["dist"] == "specialist") and (c["backend"] != "routereval" or c["dist"] in ("strong_to_weak", "all"))


def figure_grids(cfg):
    g = set(cfg["grids"])
    live = {x for x in g if re.match(r"fw_live_n\d+", x) and "lietext" not in x} | {"live_n10k_v2", "live_n100k"}
    ab = {x for x in g if re.match(r"(va_b|rivals_b|pool_fill|tuned_wsb)_", x)} | {"routereval_mmlu5k", "llmrouterbench_pool",
          "fw_live_n100", "learned_n100", "live_core_n100", "fw_live_n100_lowskill", "fw_live_n1000", "live_f1_n1000", "variants_f1",
          "learned_f1", "fw_live_n1000_lowskill", "learned_n10k", "live_n10k_cartel_random"}
    re_ = {x for x in g if x.startswith(("fw_routereval_", "re_sl_"))}
    return sorted(x for x in (live | ab | re_) & g if "shuffled" not in x)   # shuffled MIDIAN cohort: removed from the figures


def have(grid):
    d = f"{R}/{grid}"; s = {f[:-5] for f in os.listdir(f"{d}/rows.d")} if os.path.isdir(f"{d}/rows.d") else set()
    if os.path.exists(f"{d}/rows.csv"):
        try: s |= set(pd.read_csv(f"{d}/rows.csv", usecols=["rid"]).rid.dropna())
        except (ValueError, KeyError): pass
    return s


def pooled_elsewhere(cfg, grid, _c={}):
    """(method, backend, n) a pool_fill_* grid runs for a b = 3 source grid's cell: counted there, not here."""
    if "k" not in _c:
        _c["k"] = {(sp["name"], blk["backend"], c["n"]) for g in cfg["grids"] if g.startswith("pool_fill_")
                   for blk in blocks(cfg, g) for sp in method_specs(blk) for c in cells(blk) if c["b"] == 3}
    return set() if grid.startswith(("pool_fill_", "fw_", "re_sl_")) else _c["k"]


def covers(flt, cell):
    return all(str(cell.get(k)) == str(yaml.safe_load(v)) or cell.get(k) == yaml.safe_load(v) for k, v in flt.items())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--plan-out"); a = ap.parse_args()
    cfg = yaml.safe_load(open(f"{ROOT}/configs/grid.yaml"))
    held = defaultdict(list)                                 # (grid, method, seed) -> [filter dicts]
    for (g, m, only, s) in inflight(queue()):
        held[(g, m, str(s))].append(dict(kv.split("=") for kv in only.split(",") if kv))
    packed = {l.split(None, 1)[1].strip() for l in open(f"{L}/focus_packed.txt")} if os.path.exists(f"{L}/focus_packed.txt") else set()
    for line in open(f"{L}/focus_pack_plan.tsv"):
        g, m, only, s = line.rstrip("\n").split("\t")
        if f"{g}|{m}|{only}|{s}" in packed: continue          # handed to a pack: counted via inflight if its job is alive
        for mm in m.split(","): held[(g, mm, str(s))].append(dict(kv.split("=") for kv in only.split(",") if kv))
    q = queue()                                              # whole-grid submitters (va_b / rivals_b / pool_fill jobs, *_pack)
    whole = {n[len("rte_"):].removesuffix("_pack") for s, n in q.values() if n.startswith("rte_") and "__" not in n}
    per_method = {tuple(n[len("rte_"):].split("__", 1)) for s, n in q.values() if n.startswith("rte_") and "__" in n}   # rte_<grid>__<method> jobs
    gaps = defaultdict(int); tot = defaultdict(lambda: [0, 0, 0])
    for grid in figure_grids(cfg):
        done = have(grid)
        for blk in blocks(cfg, grid):
            specs = [sp for sp in method_specs(blk) if DRAWN(sp["name"], sp["params"])]
            for cell in cells(blk):
                if not FOCUS(cell): continue
                for seed in seeds(blk["seeds"]):
                    for sp in specs:
                        t = tot[grid]; t[0] += 1
                        alt = [ls for ls in ("random", "low_skill_first") if cell["beta"] == 0] or [cell["liar_select"]]
                        if any(row_id({**cell, "liar_select": ls}, sp["name"], sp["params"], seed) in done for ls in alt): t[1] += 1; continue   # beta 0: the tag is inert
                        if (sp["name"], cell["backend"], cell["n"]) in pooled_elsewhere(cfg, grid): t[2] += 1; continue
                        if grid in whole or (grid, sp["name"]) in per_method or any(covers(f, cell) for f in held.get((grid, sp["name"], str(seed)), [])): t[2] += 1; continue
                        only = ",".join(f"{k}={cell[k]}" for k in ("dist", "beta", "liar_select"))
                        gaps[(grid, sp["name"], only, seed)] += 1
    print(f"{'grid':42s} {'expected':>8s} {'done':>6s} {'queued':>6s} {'GAP':>5s}")
    for g, (e, d, q) in sorted(tot.items()):
        if e - d - q: print(f"{g:42s} {e:8d} {d:6d} {q:6d} {e - d - q:5d}")
    print(f"TOTAL gap rows: {sum(gaps.values())} in {len(gaps)} units")
    if a.plan_out:
        with open(a.plan_out, "w") as f:
            for (g, m, only, s) in sorted(gaps): f.write(f"{g}\t{m}\t{only}\t{s}\n")


if __name__ == "__main__":
    main()
