"""smolagents rival (SPEC §6A recipe 8), run inside $RTE_DATA/env/fw_smolagents.

A `ToolCallingAgent` whose `managed_agents` are the top-k candidates: smolagents exposes each managed
agent to the model as a tool named after the agent, described by its `description`, so the pick IS the
tool name in the first `ActionStep.tool_calls`. We consume `run(stream=True)`, which yields the
`ToolCall` in `process_tool_calls` BEFORE `execute_tool_call` runs it, and abandon the generator there:
no managed agent ever executes. Candidate names (`agent_%06d`) are already valid Python identifiers,
which is smolagents' only name constraint, so no sanitizing map is needed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bridge import serve_worker   # noqa: E402

_MODEL = {}


def _model(req):
    from smolagents import OpenAIServerModel
    key = (req["model"], req["base_url"], req["api_key"])
    if _MODEL.get("key") != key:
        _MODEL["key"] = key
        _MODEL["m"] = OpenAIServerModel(model_id=req["model"], api_base=req["base_url"],
                                        api_key=req["api_key"], temperature=0.0)
    return _MODEL["m"]


def select(req):
    from smolagents import LogLevel, ToolCallingAgent
    from smolagents.memory import ToolCall
    model = _model(req)
    managed = [ToolCallingAgent(tools=[], model=model, name=c["name"], description=c["description"],
                                max_steps=1, verbosity_level=LogLevel.OFF)
               for c in req["candidates"]]
    agent = ToolCallingAgent(tools=[], model=model, managed_agents=managed, max_steps=1,
                             verbosity_level=LogLevel.OFF)
    stream = agent.run(req["task"], stream=True)
    try:
        for event in stream:
            if isinstance(event, ToolCall):
                return event.name, str(event.arguments)[:500]
        return None
    finally:
        try:
            stream.close()      # smolagents' `finally: yield` makes close() raise; the run is dead either way
        except RuntimeError:
            pass


if __name__ == "__main__":
    serve_worker(select)
