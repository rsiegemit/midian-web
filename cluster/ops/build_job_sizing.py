"""Rebuild results/job_sizing.csv from sacct (OPS_RULES R2/R3): per (population size, framework) runtime p50/p95/max
and p95 MaxRSS over COMPLETED framework jobs. cluster/slurm/job_time.py reads it. Rebuild whenever the fleet changes --
unit runtime tracks supervisor latency, i.e. fleet load, not grid size.
    python cluster/ops/build_job_sizing.py [--since 2026-09-22T14:45]     # default: since the last fleet change
Note: sacct silently returns NOTHING for some long windows on this cluster; keep --since within ~1 week."""
import argparse, io, os, subprocess, sys
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--since", default="2026-09-22T14:45"); a = ap.parse_args()
raw = subprocess.run(["sacct", "-S", a.since, "-E", "now", "--format=JobID,JobName%80,State,ElapsedRaw,MaxRSS",
                      "-n", "-P", "--units=M"], capture_output=True, text=True).stdout
d = pd.read_csv(io.StringIO(raw), sep="|", header=None, names=["id", "name", "state", "sec", "rss"], dtype=str)
d["base"] = d.id.str.split(".").str[0]
top = d[~d.id.str.contains(".", regex=False)].copy()
bat = d[d.id.str.endswith(".batch")].copy(); bat["rssM"] = pd.to_numeric(bat.rss.str.rstrip("M"), errors="coerce")
j = top.merge(bat[["base", "rssM"]], on="base", how="left"); j["sec"] = pd.to_numeric(j.sec, errors="coerce")
j = j[j.state.eq("COMPLETED") & j.name.str.startswith("rte_fw_live")]
j["size"] = j.name.str.extract(r"fw_live_(n\d+k?)")[0]; j["method"] = j.name.str.split("__").str[-1]
g = j.groupby(["size", "method"]).agg(n=("sec", "size"), p50=("sec", lambda s: s.quantile(.5) / 60),
                                      p95=("sec", lambda s: s.quantile(.95) / 60), mx=("sec", lambda s: s.max() / 60),
                                      rss95=("rssM", lambda s: s.quantile(.95)))
g = g[g.n >= 10].round(0)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rte.config import RTE_DATA  # noqa: E402
out = f"{RTE_DATA}/results/job_sizing.csv"
if len(g) < 10:
    print(f"only {len(g)} (size, framework) cells with >= 10 completed jobs since {a.since}; keeping the existing table")
else:
    old = pd.read_csv(out).set_index(["size", "method"]) if os.path.exists(out) else None
    if old is not None:                                  # keep cells the new window has not measured yet
        g = pd.concat([g, old[~old.index.isin(g.index)]])
    g.to_csv(out); print(f"wrote {out}: {len(g)} cells from {len(j)} completed jobs since {a.since}")
