"""LangGraph rival (SPEC §6A row 1). See NOTES_langgraph.md."""
from ._common import FrameworkMethod


class FwLanggraph(FrameworkMethod):
    name = env = "fw_langgraph"
    worker = "langgraph_worker.py"
