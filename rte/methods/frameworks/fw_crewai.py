"""CrewAI rival (SPEC §6A row 2): `Crew(process=Process.hierarchical)` auto-builds a manager agent
whose `Delegate work to coworker` tool is the selection primitive. The manager's prompt lists only
each agent's `role`, so the worker puts the self-description in `role`. See NOTES_crewai.md."""
from ._common import FrameworkMethod


class FwCrewai(FrameworkMethod):
    name = "fw_crewai"
    env = "fw_crewai"
    worker = "crewai_worker.py"
