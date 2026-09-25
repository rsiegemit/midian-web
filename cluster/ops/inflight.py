"""Which (grid, method, only, seed) units are in flight right now, from every submission log x the live SLURM queue.
    python cluster/ops/inflight.py            # summary + any unit held by more than one live job (a double run)
    from inflight import inflight; inflight() -> {(grid, method, only, seed): [(jobid, state), ...]}
Logs read (all under $RTE_DATA/logs): launch_<grid>.txt ("jobid method|only|seed"), resubmit_*.txt and *_packed.txt
("jobid grid|method|only|seed"). A pack job holds several units. Job-level submitters (launch_va_b / launch_rivals_b /
launch_pool_fill: "jobid key") run a whole grid slice per job and are listed separately by job name."""
from __future__ import annotations
import glob, os, re, subprocess, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rte.config import RTE_DATA  # noqa: E402
L = f"{RTE_DATA}/logs"


def queue():
    out = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%i %T %j"], capture_output=True, text=True, check=True).stdout
    return {j: (s, n) for j, s, n in (l.split(None, 2) for l in out.splitlines())}


def inflight(q=None):
    q = q if q is not None else queue(); units = defaultdict(list)
    for f in glob.glob(f"{L}/launch_*.txt") + glob.glob(f"{L}/resubmit_*.txt") + glob.glob(f"{L}/*_packed.txt"):
        base = os.path.basename(f); grid = base[len("launch_"):-4] if base.startswith("launch_") else None
        for line in open(f):
            p = line.split(None, 1)
            if len(p) != 2 or p[0] not in q: continue
            f4 = p[1].strip().split("|")
            if grid and len(f4) == 3: key = (grid, *f4)
            elif len(f4) == 4: key = tuple(f4)
            else: continue                                  # job-level submitters: see by job name
            units[key].append((p[0], q[p[0]][0]))
    return units


if __name__ == "__main__":
    q = queue(); u = inflight(q)
    dup = {k: v for k, v in u.items() if len({j for j, _ in v}) > 1}
    print(f"{len(q)} jobs in queue; {len(u)} units mapped to live jobs; {len(dup)} units held by >1 live job")
    for k, v in sorted(dup.items())[:50]: print("  DUP", "|".join(map(str, k)), v)
    mapped = {j for v in u.values() for j, _ in v}
    other = defaultdict(int)
    for j, (s, n) in q.items():
        if j not in mapped: other[(n, s)] += 1
    print("jobs not mapped to a unit (job-level submitters, fleet, packs of other plans):")
    for (n, s), c in sorted(other.items(), key=lambda x: -x[1])[:40]: print(f"  {c:5d} {s:9s} {n}")
