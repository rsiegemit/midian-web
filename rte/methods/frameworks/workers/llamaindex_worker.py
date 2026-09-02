"""LlamaIndex, SPEC §6A recipe 7. selector: `LLMSingleSelector.select([ToolMetadata(name, description)], task)`
-- one LLM call, no agent execution, and the prompt shows the descriptions only. handoff: an `AgentWorkflow`
whose built-in `handoff(to_agent, reason)` tool is the primitive; we return at the first such call.
See NOTES_llamaindex.md."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
sys.path[:] = [p for p in sys.path if "/.local/lib/" not in p]   # ~/.local shadows this venv's deps
from _bridge import serve_worker  # noqa: E402
from _wk import openai_kwargs, run_async, sanitize  # noqa: E402

from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent  # noqa: E402
from llama_index.core.agent.workflow.workflow_events import ToolCall  # noqa: E402
from llama_index.core.selectors import LLMSingleSelector  # noqa: E402
from llama_index.core.tools import ToolMetadata  # noqa: E402
from llama_index.llms.openai_like import OpenAILike  # noqa: E402

INSTR = "You are a triage agent. Hand the task off to the single specialist best suited to solve it."


async def _handoff(req, llm):
    safe, back = sanitize([c["name"] for c in req["candidates"]])
    agents = [FunctionAgent(name=s, description=c["description"], system_prompt=c["description"], llm=llm)
              for s, c in zip(safe, req["candidates"])]
    triage = FunctionAgent(name="triage", description="Routes the task.", system_prompt=INSTR, llm=llm,
                           can_handoff_to=safe)
    handler = AgentWorkflow(agents=[triage] + agents, root_agent="triage").run(user_msg=req["task"])
    try:
        async for event in handler.stream_events():
            if isinstance(event, ToolCall) and event.tool_name == "handoff":
                target = event.tool_kwargs["to_agent"]
                return back[target], f"handoff(to_agent={target!r})"
    finally:
        await handler.cancel_run()
    await handler                                       # AgentWorkflow only raises when the handler is awaited


async def _select(req):
    kw = openai_kwargs(req)
    llm = OpenAILike(model=kw["model"], api_base=kw["base_url"], api_key=kw["api_key"],
                     is_chat_model=True, is_function_calling_model=True, temperature=0.0)
    if req.get("params", {}).get("mode", "selector") == "handoff":
        return await _handoff(req, llm)
    choices = [ToolMetadata(name=c["name"], description=c["description"]) for c in req["candidates"]]
    result = await LLMSingleSelector.from_defaults(llm=llm).aselect(choices, req["task"])
    ind = result.selections[0].index                    # first selection (a model that names several agents would raise on .ind)
    if 0 <= ind < len(choices):
        return req["candidates"][ind]["name"], f"LLMSingleSelector ind={ind} of {len(result.selections)}"


if __name__ == "__main__":
    serve_worker(lambda req: run_async(_select(req)))
