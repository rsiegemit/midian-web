"""AutoGen rival. See docs/frameworks/NOTES_autogen.md.

Ledger and params: as FrameworkMethod (rte.methods.frameworks._common)."""
from ._common import FrameworkMethod


class FwAutogen(FrameworkMethod):
    name = env = "fw_autogen"
    worker = "autogen_worker.py"
