"""Microsoft Agent Framework, SPEC §6A recipe 4. groupchat: the GroupChatBuilder orchestrator answers with
`AgentOrchestrationOutput.next_speaker` over a `name: description` roster. handoff: the triage agent calls one
`handoff_to_<name>` tool per candidate, described by that candidate's description. We return at the first
selection event, so no candidate agent runs. See NOTES_maf.md."""
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
sys.path[:] = [p for p in sys.path if "/.local/lib/" not in p]   # ~/.local shadows this venv's deps
from _bridge import serve_worker  # noqa: E402
from _wk import openai_kwargs, run_async, sanitize  # noqa: E402

from agent_framework import Agent  # noqa: E402
from agent_framework.openai import OpenAIChatCompletionClient  # noqa: E402
from agent_framework.orchestrations import (GroupChatBuilder, GroupChatRequestSentEvent,  # noqa: E402
                                            HandoffBuilder, HandoffSentEvent)

for _log in ("opentelemetry", "asyncio", "agent_framework", "agent_framework_orchestrations"):
    logging.getLogger(_log).setLevel(logging.CRITICAL)   # abandoning the run at the pick is noisy by design
INSTR = "You are a router. Pick the single participant best suited to solve the task. Do not solve it yourself."


async def _select(req):
    mode = req.get("params", {}).get("mode", "groupchat")
    KW = {"default_options": {"temperature": 0.0},          # handoff refuses participants without the history flag
          "require_per_service_call_history_persistence": mode == "handoff"}
    client = OpenAIChatCompletionClient(**openai_kwargs(req))
    safe, back = sanitize([c["name"] for c in req["candidates"]])
    agents = [Agent(client, c["description"], name=s, description=c["description"], **kw)
              for s, c in zip(safe, req["candidates"])]
    router = Agent(client, INSTR, name="router", description="Routes the task.", **KW)
    if mode == "handoff":
        flow = HandoffBuilder(participants=[router] + agents).add_handoff(router, agents).with_start_agent(router)
        want, field = HandoffSentEvent, "target"
    else:
        flow = GroupChatBuilder(participants=agents, orchestrator_agent=router, max_rounds=1)
        want, field = GroupChatRequestSentEvent, "participant_name"
    async for event in flow.build().run(req["task"], stream=True):
        if isinstance(event.data, want):                # returning here abandons the run before the pick executes
            return back[getattr(event.data, field)], repr(event.data)[:500]


if __name__ == "__main__":
    serve_worker(lambda req: run_async(_select(req)))
