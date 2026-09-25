"""Magentic-One rival; shares the AutoGen venv. See docs/frameworks/NOTES_magentic_one.md.

Ledger and params: as FrameworkMethod (midian.methods.frameworks._common)."""
from ._common import FrameworkMethod


class FwMagenticOne(FrameworkMethod):
    name = "fw_magentic_one"
    env = "fw_autogen"
    worker = "magentic_one_worker.py"
