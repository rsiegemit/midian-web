"""smolagents rival (SPEC §6A row 9, recipe 8). See NOTES_smolagents.md."""
from ._common import FrameworkMethod


class FwSmolagents(FrameworkMethod):
    name, env, worker = "fw_smolagents", "fw_smolagents", "smolagents_worker.py"
