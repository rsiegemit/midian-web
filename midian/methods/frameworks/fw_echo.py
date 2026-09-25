"""Protocol check, not a rival: bridge to echo_worker in the base env. Excluded from grids."""
from ._common import FrameworkMethod


class FwEcho(FrameworkMethod):
    name = "fw_echo"
    env = "rte"                 # any python works for the echo worker
    worker = "echo_worker.py"
