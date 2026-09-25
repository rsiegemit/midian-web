"""CrewAI rival. See docs/frameworks/NOTES_crewai.md.

Ledger and params: as FrameworkMethod (midian.methods.frameworks._common)."""
from ._common import FrameworkMethod


class FwCrewai(FrameworkMethod):
    name = env = "fw_crewai"
    worker = "crewai_worker.py"
