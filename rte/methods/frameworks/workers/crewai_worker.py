"""CrewAI rival worker (SPEC §6A recipe 2). Selection primitive: a `Process.hierarchical` crew, whose
auto-built "Crew Manager" agent is given the `Delegate work to coworker` tool. That tool's description is
`"...one of the following coworkers: {', '.join(agent.role)}"` -- the manager sees ONLY `role` strings, so
the self-description goes in `role` (prefixed with the agent id so the pick is invertible).

The pick is the manager's `coworker` argument. We read it at `BaseAgentTool._execute`, the first place it
is available synchronously, and raise a `BaseException` there so the chosen coworker never executes the
task (CrewAI's own `except Exception` handlers cannot swallow it). See NOTES_crewai.md.
"""
import os
import re
import sys

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bridge import serve_worker  # noqa: E402

from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402
from crewai.tools.agent_tools.base_agent_tools import BaseAgentTool  # noqa: E402

NAME = re.compile(r"agent_\d{6}")


class _Picked(BaseException):
    """Not an Exception: CrewAI wraps tool bodies in `except Exception` and would retry instead of stopping."""

    def __init__(self, coworker):
        self.coworker = coworker


BaseAgentTool._execute = lambda self, agent_name, task, context=None: (_ for _ in ()).throw(_Picked(agent_name))


def select(req):
    cands = req["candidates"]
    llm = LLM(model="openai/" + req["model"], base_url=req["base_url"], api_key=req["api_key"] or "EMPTY",
              temperature=0)
    agents = [Agent(role=f"{c['name']}: {c['description']}", goal="Solve tasks you are best at.",
                    backstory=c["description"], llm=llm, tools=[], allow_delegation=False, verbose=False)
              for c in cands]
    crew = Crew(agents=agents, tasks=[Task(description=req["task"], expected_output="The answer.")],
                process=Process.hierarchical, manager_llm=llm, memory=False, cache=False, verbose=False)
    try:
        crew.kickoff()
    except _Picked as p:
        raw = (p.coworker or "")[:500]
        by_role = {a.role: c["name"] for a, c in zip(agents, cands)}
        hit = NAME.search(raw)
        return (by_role.get(p.coworker) or (hit.group(0) if hit else None)), raw
    return None


if __name__ == "__main__":
    serve_worker(select)
