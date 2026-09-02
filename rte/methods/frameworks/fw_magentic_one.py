"""Magentic-One rival (SPEC §6A row 4): `MagenticOneGroupChat`'s orchestrator emits a JSON progress
ledger; its `next_speaker` field is the selection primitive. Shares the AutoGen venv.
See NOTES_magentic_one.md."""
from ._common import FrameworkMethod


class FwMagenticOne(FrameworkMethod):
    name = "fw_magentic_one"
    env = "fw_autogen"
    worker = "magentic_one_worker.py"
