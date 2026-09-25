"""LlamaIndex rival, mode="selector"|"handoff". See docs/frameworks/NOTES_llamaindex.md.

Ledger and params: as FrameworkMethod (rte.methods.frameworks._common), plus mode="selector"."""
from ._common import FrameworkMethod


class FwLlamaIndex(FrameworkMethod):
    name = env = "fw_llamaindex"
    worker = "llamaindex_worker.py"

    def __init__(self, mode: str = "selector", **params):
        super().__init__(mode=mode, **params)     # `params` reaches the worker in the bridge request
