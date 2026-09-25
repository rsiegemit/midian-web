"""Resubmit the units of framework grids whose rows are missing (planned vs landed, set comparison), one 1-CPU job per
(method, dist, beta, liar_select, seed) exactly as scripts/launch_units.sh would. Refuses a grid that still has jobs queued
(its units may be in flight). Log: logs/rerun_missing.txt.
    RTE_DATA=... python scripts/ops/rerun_missing.py <grid> [<grid> ...] [--dry-run]"""
import os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path[:0] = [ROOT, f"{ROOT}/scripts"]
import seed_tables as t
from shortlist_figs import rows
from job_time import minutes, slurm

D = os.environ["RTE_DATA"]; dry = "--dry-run" in sys.argv
queued = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%j"], capture_output=True, text=True, check=True).stdout.split()
for g in [a for a in sys.argv[1:] if not a.startswith("--")]:
    if any(j.startswith(f"rte_{g}__") for j in queued): print(f"{g}: jobs still queued, skipped"); continue
    df = rows(g)
    have = set() if df.empty else {(int(n), int(b), str(d), float(be), str(ls), int(s), t.label(m, p)) for n, b, d, be, ls, s, m, p
                                   in zip(df.n, df.b, df.dist, df.beta, df.liar_select, df.seed, df.method, df.params.astype(str))}
    miss = t.planned(g) - have
    units = sorted({(lab.split("[")[0], d, be, ls, s) for (_, _, d, be, ls, s, lab) in miss})
    print(f"{g}: {len(miss)} rows missing -> {len(units)} units")
    for m, d, be, ls, s in units:
        only = f"dist={d},beta={be},liar_select={ls}"
        cmd = ["sbatch", "--parsable", "-p", "sapphire,serial_requeue,shared", "-A", "sompolinsky_lab", "-c", "1", "--mem=40G",
               f"--time={slurm(minutes(g, m))}", f"--job-name=rte_{g}__{m}", "-o", f"{D}/logs/units/%x-%j.out", "-e", f"{D}/logs/units/%x-%j.err",
               f"--export=ALL,RTE_PYTHON={D}/env/rte/bin/python,RTE_WORKERS=1,RTE_CONSOLIDATE=0,RTE_FW_PARALLEL=8,RTE_EMBED_CACHE_DIR={D}/cache/embed_routereval",
               "scripts/run_grid.sbatch", g, "--methods", m, "--only", only, "--seeds", str(s)]
        if dry: print("   ", m, only, s); continue
        j = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()
        subprocess.run(["scontrol", "update", f"JobId={j}", "Partition=sapphire,serial_requeue,shared,intermediate"], capture_output=True)
        open(f"{D}/logs/rerun_missing.txt", "a").write(f"{j} {g}|{m}|{only}|{s}\n")
