"""Remove fleet endpoints whose replica job is gone (a replica killed at its time limit never deregisters, and every
framework call routed to it fails as an infrastructure error). Only "<model>#<jobid>" entries are touched; base fleet
entries without a job id are left alone. Does nothing if the queue cannot be read.
    python cluster/ops/prune_endpoints.py"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rte.config import RTE_DATA  # noqa: E402
D = str(RTE_DATA)
q = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%i"], capture_output=True, text=True)
if q.returncode != 0 or not q.stdout.strip(): sys.exit("squeue unreadable: nothing pruned")
live = set(q.stdout.split())
dead = [k for k in json.load(open(f"{D}/endpoints.json")) if "#" in k and k.split("#", 1)[1] not in live]
for k in dead:
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "slurm", "_register_endpoint.py"), "remove", k], check=False)
print(f"pruned {len(dead)} dead endpoint(s): {dead}")
