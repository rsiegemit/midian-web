"""smolagents rival. See docs/frameworks/NOTES_smolagents.md.

Ledger and params: as FrameworkMethod (rte.methods.frameworks._common)."""
from ._common import FrameworkMethod


class FwSmolagents(FrameworkMethod):
    name, env, worker = "fw_smolagents", "fw_smolagents", "smolagents_worker.py"
