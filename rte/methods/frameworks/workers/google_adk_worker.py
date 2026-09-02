"""Google ADK, SPEC §6A recipe 6: a router `LlmAgent(sub_agents=candidates)`. ADK's auto-delegation attaches a
`transfer_to_agent` tool and builds the roster from each sub-agent's `Agent name:` / `Agent description:`.
We return at the first `event.actions.transfer_to_agent`, so the sub-agent never runs. See NOTES_google_adk.md."""
import logging
import os
import sys
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
sys.path[:] = [p for p in sys.path if "/.local/lib/" not in p]   # ~/.local has a `google` pkg that shadows google.adk
from _bridge import serve_worker  # noqa: E402
from _wk import openai_kwargs, run_async, sanitize  # noqa: E402

warnings.filterwarnings("ignore")                       # ADK flags its JSON-schema tool path as experimental
from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.models.lite_llm import LiteLlm  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

for _log in ("google_adk", "opentelemetry", "asyncio"):
    logging.getLogger(_log).setLevel(logging.CRITICAL)   # stopping the run at the pick is noisy by design
INSTR = "You are a router. Transfer the task to the single sub-agent best suited to solve it."
APP = "rte"


async def _select(req):
    kw = openai_kwargs(req)
    llm = LiteLlm(model="openai/" + kw["model"], api_base=kw["base_url"], api_key=kw["api_key"], temperature=0.0)
    safe, back = sanitize([c["name"] for c in req["candidates"]])
    subs = [LlmAgent(name=s, description=c["description"], instruction=c["description"], model=llm)
            for s, c in zip(safe, req["candidates"])]
    router = LlmAgent(name="router", description="Routes tasks.", instruction=INSTR, model=llm, sub_agents=subs)
    runner = InMemoryRunner(agent=router, app_name=APP)
    session = await runner.session_service.create_session(app_name=APP, user_id=APP)
    message = types.Content(role="user", parts=[types.Part(text=req["task"])])
    async for event in runner.run_async(user_id=APP, session_id=session.id, new_message=message):
        target = getattr(event.actions, "transfer_to_agent", None)
        if target:
            return back[target], f"transfer_to_agent -> {target}"


if __name__ == "__main__":
    serve_worker(lambda req: run_async(_select(req)))
