"""CrewAI, SPEC §6A recipe 2: a `Process.hierarchical` crew's auto-built manager delegates through
`Delegate work to coworker`, whose description lists only each agent's `role` -- so the self-description
goes there, prefixed with the agent id to stay invertible. See NOTES_crewai.md."""
import contextlib
import os
import re
import sys

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
from _bridge import serve_worker  # noqa: E402
from _wk import openai_kwargs  # noqa: E402

from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402
from crewai.events.event_context import restore_event_scope  # noqa: E402
from crewai.tools.agent_tools.base_agent_tools import BaseAgentTool  # noqa: E402

NAME = re.compile(r"agent_\d{6}")


class Picked(BaseException):
    """Not an `Exception`: CrewAI wraps the delegated call in `except Exception` and would turn the abort
    into an error string fed back to the manager instead of stopping."""


def _stop(self, agent_name, task, context=None):
    raise Picked(agent_name or "")


BaseAgentTool._execute = _stop           # the manager's `coworker` argument, before the coworker runs


def select(req):
    # CrewAI prints Rich panels and event warnings to stdout, which is the bridge's JSON-lines channel;
    # and aborting mid-tool leaves the started tool_usage event unclosed, so its per-process scope stack
    # would hit the depth-100 limit after 100 requests.
    restore_event_scope(())
    with contextlib.redirect_stdout(sys.stderr):
        kw = openai_kwargs(req)
        llm = LLM(temperature=0, **{**kw, "model": "openai/" + kw["model"]})
        agents = [Agent(role=f"{c['name']}: {c['description']}", goal="Solve tasks you are best at.",
                        backstory=c["description"], llm=llm, tools=[], allow_delegation=False)
                  for c in req["candidates"]]
        crew = Crew(agents=agents, tasks=[Task(description=req["task"], expected_output="The answer.")],
                    process=Process.hierarchical, manager_llm=llm)
        try:
            crew.kickoff()
        except Picked as p:
            hit = NAME.search(p.args[0])
            return (hit.group(0) if hit else None), p.args[0][:500]


if __name__ == "__main__":
    serve_worker(select)
