"""Precompute the dense description embeddings the frameworks' SOTA retrieval stack reads, one .npy per population.
    python scripts/embed_populations.py [--model M] [--dists ...] [--ns ...] [--seeds K] [--list]
Writes <population>/descriptions_<slug>.npy (n x d) and families_<slug>.npy (K x d), exactly the files
FrameworkMethod._embeddings looks for, so the routing jobs never load the embedder. Needs a GPU for anything
larger than MiniLM; run it as a SLURM job, never on the login node. Re-running is free: finished populations
are skipped, and each file is written atomically so concurrent array tasks cannot corrupt one another."""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rte.backends import families
from rte.methods._learned import embed, resolve

RTE_DATA = os.environ.get("RTE_DATA", "/scratch/rte"); POP = f"{RTE_DATA}/populations"


def slug(model: str) -> str:
    import re
    return "minilm" if model == "all-MiniLM-L6-v2" else re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def save(path, arr):
    tmp = f"{path}.{os.getpid()}.tmp.npy"; np.save(tmp, arr); os.replace(tmp, path)


def population_dirs(dists, ns, seeds):
    out = []
    for d in sorted(os.listdir(POP)):
        p = f"{POP}/{d}"
        if not os.path.isdir(p) or not os.path.exists(f"{p}/descriptions.json"): continue
        try:
            dist, n, _K, seed = d.rsplit("_n", 1)[0], int(d.split("_n")[1].split("_")[0]), 0, int(d.split("_seed")[1])
        except (IndexError, ValueError):
            continue
        if dists and dist not in dists: continue
        if ns and n not in ns: continue
        if seeds and seed > seeds: continue
        out.append((n, dist, seed, p))
    return [t[3] for t in sorted(out)]                      # smallest first: cheap cells land while 10^5 runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--dists", nargs="*", default=[]); ap.add_argument("--ns", nargs="*", type=int, default=[])
    ap.add_argument("--seeds", type=int, default=0, help="keep seeds 1..S (0 = all)")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    sl = slug(a.model)
    dirs = population_dirs(a.dists, a.ns, a.seeds)[a.shard::a.shards]
    todo = [p for p in dirs if not os.path.exists(f"{p}/descriptions_{sl}.npy")]
    print(f"model {a.model} (local: {resolve(a.model)})\n{len(dirs)} populations in shard, {len(todo)} to do", flush=True)
    if a.list:
        for p in todo: print("  ", os.path.basename(p), len(json.load(open(f"{p}/descriptions.json"))))
        return
    fam = [families.describe(f) for f in families.FAMILIES_16]      # every live grid is K = 16
    for i, p in enumerate(todo, 1):
        desc = json.load(open(f"{p}/descriptions.json"))
        t0 = time.time(); E = embed(desc, a.model)
        save(f"{p}/descriptions_{sl}.npy", E)
        dt = time.time() - t0
        print(f"[{i}/{len(todo)}] {os.path.basename(p):38s} {len(desc):7,d} texts  {dt:7.1f}s  {len(desc)/max(dt,1e-9):7.1f}/s", flush=True)
        if not os.path.exists(f"{p}/families_{sl}.npy"):
            save(f"{p}/families_{sl}.npy", embed(fam, a.model, prompt_name="query"))
    print("EMBED_POPULATIONS_DONE", flush=True)


if __name__ == "__main__":
    main()
