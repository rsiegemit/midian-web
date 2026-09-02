"""LlamaIndex rival (SPEC §6A row 8), mode="selector"|"handoff". See NOTES_llamaindex.md."""
from ._common import FrameworkMethod


class FwLlamaIndex(FrameworkMethod):
    name = env = "fw_llamaindex"
    worker = "llamaindex_worker.py"

    def __init__(self, mode: str = "selector", **params):
        super().__init__(mode=mode, **params)     # `params` reaches the worker in the bridge request
