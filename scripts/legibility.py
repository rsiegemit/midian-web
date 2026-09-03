"""H2 input: Spearman(self-described D, true S) per (n, shape, seed) of the framework grids -> results/extra_figs/legibility.json.
Runs offline (descriptions and self-ratings are memo hits); S is runner-only and never reaches a method.  sbatch-able."""
import json, os, sys, yaml
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rte.run import RTE_DATA, CELL, blocks, cells, seeds
from rte.world import World

cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "grid.yaml"))); out = []
for grid in ("fw_live_n100", "fw_live_n1000"):
    for blk in blocks(cfg, grid):
        for c in cells(blk):
            if c["beta"] != 0 or c["declared_source"] != "self_described": continue          # honest self-descriptions vs S
            for s in seeds(blk["seeds"]):
                w = World(**{k: c[k] for k in CELL if k not in ("b", "Q")}, seed=s, backend_kwargs=c["backend_kwargs"] or None)
                rho = spearmanr(w.D.ravel(), w.S.ravel()).correlation
                fam = np.nanmean([spearmanr(w.D[:, f], w.S[:, f]).correlation for f in range(w.K)])
                out.append({"n": int(w.n), "dist": c["dist"], "seed": s, "spearman": float(rho), "spearman_per_family": float(fam)})
                print(out[-1], flush=True)
os.makedirs(f"{RTE_DATA}/results/extra_figs", exist_ok=True)
json.dump(out, open(f"{RTE_DATA}/results/extra_figs/legibility.json", "w"), indent=1)
