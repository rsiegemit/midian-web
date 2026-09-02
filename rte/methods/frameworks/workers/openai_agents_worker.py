"""OpenAI Agents SDK, SPEC §6A recipe 5: a triage agent whose handoffs become `transfer_to_<name>` tools
described by each candidate's `handoff_description`. `RunHooks.on_handoff` fires before the target agent's
turn, so raising out of it stops the run at the pick. The endpoint is reached with `use_responses=False`
because vLLM serves chat-completions, not the Responses API. See NOTES_openai_agents.md."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
sys.path[:] = [p for p in sys.path if "/.local/lib/" not in p]   # ~/.local shadows this venv's deps
from _bridge import serve_worker  # noqa: E402
from _wk import openai_kwargs, run_async, sanitize  # noqa: E402

from agents import Agent, ModelSettings, RunConfig, RunHooks, Runner  # noqa: E402
from agents.models.openai_provider import OpenAIProvider  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

INSTR = "You are a triage agent. Hand the task off to the single specialist best suited to solve it."


class Picked(Exception):
    def __init__(self, name):
        self.name = name


class Hooks(RunHooks):
    async def on_handoff(self, context, from_agent, to_agent):
        raise Picked(to_agent.name)                     # abort before the target agent runs


async def _select(req):
    kw = openai_kwargs(req)
    provider = OpenAIProvider(openai_client=AsyncOpenAI(base_url=kw["base_url"], api_key=kw["api_key"]),
                              use_responses=False)
    safe, back = sanitize([c["name"] for c in req["candidates"]])
    specialists = [Agent(name=s, handoff_description=c["description"], instructions=c["description"])
                   for s, c in zip(safe, req["candidates"])]
    triage = Agent(name="triage", instructions=INSTR, handoffs=specialists)
    config = RunConfig(model=kw["model"], model_provider=provider, tracing_disabled=True,
                       model_settings=ModelSettings(temperature=0.0))
    try:
        await Runner.run(triage, req["task"], max_turns=1, hooks=Hooks(), run_config=config)
    except Picked as picked:
        return back[picked.name], f"on_handoff -> {picked.name}"


if __name__ == "__main__":
    serve_worker(lambda req: run_async(_select(req)))
