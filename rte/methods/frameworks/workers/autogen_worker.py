"""AutoGen rival worker (SPEC §6A recipe 3). Selection primitive: `SelectorGroupChat`, whose group-chat
manager formats a `{roles}` roster of `"<name> : <description>"` lines into `selector_prompt` and asks the
model for the next speaker; the pick surfaces as a `SelectSpeakerEvent` (emitted because
`emit_team_events=True`). Self-descriptions therefore go in `AssistantAgent(description=...)`.
We break out of `run_stream` at that event, so the selected agent never runs. See NOTES_autogen.md.
"""
import asyncio
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bridge import serve_worker  # noqa: E402

warnings.filterwarnings("ignore")                 # autogen re-serializes its own pydantic ledger models
from autogen_agentchat.agents import AssistantAgent  # noqa: E402
from autogen_agentchat.base import Response  # noqa: E402
from autogen_agentchat.conditions import MaxMessageTermination  # noqa: E402
from autogen_agentchat.messages import SelectSpeakerEvent, TextMessage  # noqa: E402
from autogen_agentchat.teams import SelectorGroupChat  # noqa: E402
from autogen_core.models import ModelFamily  # noqa: E402
from autogen_ext.models.openai import OpenAIChatCompletionClient  # noqa: E402

LOOP = asyncio.new_event_loop()          # one persistent event loop for the life of the worker process
MODEL_INFO = {"family": ModelFamily.UNKNOWN, "vision": False, "function_calling": True,
              "json_output": True, "structured_output": False}


class Idle(AssistantAgent):
    """A real AutoGen participant (so `description` reaches the selection prompt unchanged) that answers
    with nothing. Both AutoGen teams dispatch to the chosen speaker BEFORE emitting `SelectSpeakerEvent`,
    so a plain `AssistantAgent` would spend one model call on the task before we could abort; this makes
    that call impossible instead of racing it."""

    async def on_messages(self, messages, cancellation_token):
        return Response(chat_message=TextMessage(content="", source=self.name))

    async def on_messages_stream(self, messages, cancellation_token):
        yield await self.on_messages(messages, cancellation_token)


def agents_for(cands, mc):
    return [Idle(name=c["name"], model_client=mc, description=c["description"], system_message=c["description"])
            for c in cands]


def client(req, model_info=None):
    return OpenAIChatCompletionClient(model=req["model"], base_url=req["base_url"],
                                      api_key=req["api_key"] or "EMPTY", temperature=0,
                                      model_info=model_info or MODEL_INFO)


async def first_speaker(team, task):
    """Run the team only until its manager announces a speaker, then abort."""
    stream = team.run_stream(task=task)
    try:
        async for msg in stream:
            if isinstance(msg, SelectSpeakerEvent) and msg.content:
                return msg.content[0], str(msg.content)[:500]
    finally:
        await stream.aclose()
    return None


async def _select(req):
    cands = req["candidates"]
    if len(cands) == 1:
        return cands[0]["name"], "single candidate"
    mc = client(req)
    agents = agents_for(cands, mc)
    team = SelectorGroupChat(agents, model_client=mc, emit_team_events=True, allow_repeated_speaker=True,
                             termination_condition=MaxMessageTermination(2))
    try:
        return await first_speaker(team, req["task"])
    finally:
        await mc.close()


if __name__ == "__main__":
    serve_worker(lambda req: LOOP.run_until_complete(_select(req)))
