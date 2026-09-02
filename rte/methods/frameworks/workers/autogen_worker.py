"""AutoGen, SPEC §6A recipe 3: `SelectorGroupChat`'s manager formats a `"<name>: <description>"` roster
into its `selector_prompt` and the pick arrives as a `SelectSpeakerEvent`. Descriptions therefore go in
`description`. Shared with the Magentic-One worker. See NOTES_autogen.md."""
import os
import sys
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
from _bridge import serve_worker  # noqa: E402
from _wk import openai_kwargs, run_async  # noqa: E402

warnings.filterwarnings("ignore")                # autogen re-serializes its own pydantic ledger models
from autogen_agentchat.agents import AssistantAgent  # noqa: E402
from autogen_agentchat.base import Response  # noqa: E402
from autogen_agentchat.conditions import MaxMessageTermination  # noqa: E402
from autogen_agentchat.messages import SelectSpeakerEvent, TextMessage  # noqa: E402
from autogen_agentchat.teams import SelectorGroupChat  # noqa: E402
from autogen_core.models import ModelFamily  # noqa: E402
from autogen_ext.models.openai import OpenAIChatCompletionClient  # noqa: E402

MODEL_INFO = {"family": ModelFamily.UNKNOWN, "vision": False, "function_calling": True,
              "json_output": True, "structured_output": False}


class Idle(AssistantAgent):
    """A participant that answers with nothing. Both AutoGen teams dispatch to the chosen speaker BEFORE
    emitting `SelectSpeakerEvent`, so a live participant spends one model call on the task before we can
    abort; this makes that call impossible rather than racing it."""

    async def on_messages(self, messages, token):
        return Response(chat_message=TextMessage(content="", source=self.name))

    async def on_messages_stream(self, messages, token):
        yield await self.on_messages(messages, token)


def team_parts(req, **info):
    """The model client and the participants -- identical for both AutoGen teams."""
    mc = OpenAIChatCompletionClient(temperature=0, model_info=dict(MODEL_INFO, **info), **openai_kwargs(req))
    return mc, [Idle(name=c["name"], model_client=mc, description=c["description"]) for c in req["candidates"]]


async def first_speaker(team, mc, task):
    """Run the team only until its manager announces a speaker, then abort."""
    stream = team.run_stream(task=task)
    try:
        async for msg in stream:
            if isinstance(msg, SelectSpeakerEvent) and msg.content:
                return msg.content[0], str(msg.content)
    finally:
        await stream.aclose()
        await mc.close()


async def _select(req):
    mc, agents = team_parts(req)
    team = SelectorGroupChat(agents, model_client=mc, emit_team_events=True, allow_repeated_speaker=True,
                             termination_condition=MaxMessageTermination(2))
    return await first_speaker(team, mc, req["task"])


if __name__ == "__main__":
    serve_worker(lambda req: run_async(_select(req)))
