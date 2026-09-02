"""smolagents recipe: managed agents ARE the tools, so the pick is the first tool call's name.
`run(stream=True)` yields the ToolCall before execute_tool_call runs it, so nothing executes."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bridge import serve_worker      # noqa: E402
from _wk import openai_kwargs, sanitize   # noqa: E402


def select(req):
    from smolagents import LogLevel, OpenAIServerModel, ToolCallingAgent
    from smolagents.memory import ToolCall
    kw = openai_kwargs(req)
    model = OpenAIServerModel(model_id=kw["model"], api_base=kw["base_url"], api_key=kw["api_key"],
                              temperature=0.0)
    safe, back = sanitize([c["name"] for c in req["candidates"]])   # smolagents requires identifiers
    managed = [ToolCallingAgent(tools=[], model=model, name=s, description=c["description"],
                                max_steps=1, verbosity_level=LogLevel.OFF)
               for s, c in zip(safe, req["candidates"])]
    agent = ToolCallingAgent(tools=[], model=model, managed_agents=managed, max_steps=1,
                             verbosity_level=LogLevel.OFF)
    stream = agent.run(req["task"], stream=True)
    try:
        for event in stream:
            if isinstance(event, ToolCall):
                return back.get(event.name, event.name), str(event.arguments)[:500]
    finally:
        try:
            stream.close()     # smolagents' `finally: yield` makes close() raise; the run is dead anyway
        except RuntimeError:
            pass


if __name__ == "__main__":
    serve_worker(select)
