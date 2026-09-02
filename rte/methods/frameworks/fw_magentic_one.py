"""Magentic-One rival (SPEC §6A row 4); shares the AutoGen venv. See NOTES_magentic_one.md."""
from ._common import FrameworkMethod


class FwMagenticOne(FrameworkMethod):
    name = "fw_magentic_one"
    env = "fw_autogen"
    worker = "magentic_one_worker.py"
