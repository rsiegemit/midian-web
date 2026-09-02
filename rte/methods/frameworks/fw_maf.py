"""Microsoft Agent Framework rival (SPEC §6A row 5), mode="groupchat"|"handoff". See NOTES_maf.md."""
from ._common import FrameworkMethod


class FwMaf(FrameworkMethod):
    name = env = "fw_maf"
    worker = "maf_worker.py"

    def __init__(self, mode: str = "groupchat", **params):
        super().__init__(mode=mode, **params)     # `params` reaches the worker in the bridge request
