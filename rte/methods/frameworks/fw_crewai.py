"""CrewAI rival (SPEC §6A row 2). See NOTES_crewai.md."""
from ._common import FrameworkMethod


class FwCrewai(FrameworkMethod):
    name = env = "fw_crewai"
    worker = "crewai_worker.py"
