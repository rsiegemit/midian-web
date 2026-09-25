"""Resubmit the units of framework grids whose rows are missing (planned vs landed, set comparison), one 1-CPU job per
(method, dist, beta, liar_select, seed) exactly as cluster/slurm/launch_units.sh would. Refuses a grid that still has
jobs queued (its units may be in flight). Log: logs/rerun_missing.txt.
    RTE_DATA=... RTE_ACCOUNT=... python cluster/ops/rerun_missing.py <grid> [<grid> ...] [--dry-run]
Partitions: $RTE_CPU_PARTITIONS; account: $RTE_ACCOUNT (cluster/cluster.env.example)."""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, ROOT)
import argparse  # noqa: E402

import scripts.figures.lib.seed_tables as t  # noqa: E402
from cluster.slurm.job_time import minutes, slurm  # noqa: E402
from scripts.figures.lib.rows import label  # noqa: E402
from scripts.figures.shortlist_figs import rows  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("grids", nargs="+")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
D, PART, ACCT = os.environ["RTE_DATA"], os.environ["RTE_CPU_PARTITIONS"], os.environ["RTE_ACCOUNT"]
dry = args.dry_run
queued = subprocess.run(
    ["squeue", "-u", os.environ["USER"], "-h", "-o", "%j"], capture_output=True, text=True, check=True
).stdout.split()
for g in args.grids:
    if any(j.startswith(f"rte_{g}__") for j in queued):
        print(f"{g}: jobs still queued, skipped")
        continue
    df = rows(g)
    have = (
        set()
        if df.empty
        else {
            (int(n), int(b), str(d), float(be), str(ls), int(s), label(m, p))
            for n, b, d, be, ls, s, m, p in zip(
                df.n, df.b, df.dist, df.beta, df.liar_select, df.seed, df.method, df.params.astype(str)
            )
        }
    )
    miss = t.planned(g) - have
    units = sorted({(lab.split("[")[0], d, be, ls, s) for (_, _, d, be, ls, s, lab) in miss})
    print(f"{g}: {len(miss)} rows missing -> {len(units)} units")
    for m, d, be, ls, s in units:
        only = f"dist={d},beta={be},liar_select={ls}"
        cmd = [
            "sbatch",
            "--parsable",
            "-p",
            PART,
            "-A",
            ACCT,
            "-c",
            "1",
            "--mem=40G",
            f"--time={slurm(minutes(g, m))}",
            f"--job-name=rte_{g}__{m}",
            "-o",
            f"{D}/logs/units/%x-%j.out",
            "-e",
            f"{D}/logs/units/%x-%j.err",
            f"--export=ALL,RTE_PYTHON={D}/env/rte/bin/python,RTE_WORKERS=1,RTE_CONSOLIDATE=0,RTE_FW_PARALLEL=8,RTE_EMBED_CACHE_DIR={D}/cache/embed_routereval",
            "cluster/slurm/run_grid.sbatch",
            g,
            "--methods",
            m,
            "--only",
            only,
            "--seeds",
            str(s),
        ]
        if dry:
            print("   ", m, only, s)
            continue
        j = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()
        if os.environ.get("RTE_WIDE_PARTITIONS"):  # widen once queued (optional, cluster/cluster.env.example)
            subprocess.run(
                ["scontrol", "update", f"JobId={j}", f"Partition={os.environ['RTE_WIDE_PARTITIONS']}"],
                capture_output=True,
            )
        open(f"{D}/logs/rerun_missing.txt", "a").write(f"{j} {g}|{m}|{only}|{s}\n")
