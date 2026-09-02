"""LangGraph rival (SPEC §6A row 1): `langgraph_supervisor.create_supervisor` builds a supervisor
node whose `transfer_to_<name>` handoff tools are the selection primitive. Self-descriptions are NOT
injected by the library, so the worker puts the roster in the supervisor `prompt` (the docs' pattern).
See NOTES_langgraph.md."""
from ._common import FrameworkMethod


class FwLanggraph(FrameworkMethod):
    name = "fw_langgraph"
    env = "fw_langgraph"
    worker = "langgraph_worker.py"
