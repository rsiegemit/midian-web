"""LangGraph rival. See docs/frameworks/NOTES_langgraph.md.

Ledger and params: as FrameworkMethod (rte.methods.frameworks._common)."""
from ._common import FrameworkMethod


class FwLanggraph(FrameworkMethod):
    name = env = "fw_langgraph"
    worker = "langgraph_worker.py"
