"""LangGraph, SPEC §6A recipe 1: `create_supervisor` gives the supervisor one `transfer_to_<name>` handoff
tool per agent and injects no descriptions, so the roster goes in its `prompt`. See NOTES_langgraph.md."""
import os
import sys
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
from _bridge import serve_worker  # noqa: E402
from _wk import openai_kwargs  # noqa: E402

warnings.filterwarnings("ignore")                # langgraph V1 deprecation notice on create_react_agent
from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.prebuilt import create_react_agent  # noqa: E402
from langgraph_supervisor import create_supervisor  # noqa: E402

PROMPT = ("You are a supervisor managing a team of agents:\n{roster}\n\n"
          "Assign the user's task to exactly one agent by calling its transfer tool. "
          "Do not do any work yourself.")
PREFIX = "transfer_to_"


def select(req):
    model = ChatOpenAI(temperature=0, max_retries=0, **openai_kwargs(req))
    agents = [create_react_agent(model, tools=[], name=c["name"]) for c in req["candidates"]]
    roster = "\n".join(f"- {c['name']}: {c['description']}" for c in req["candidates"])
    app = create_supervisor(agents, model=model, prompt=PROMPT.format(roster=roster)).compile()
    stream = app.stream({"messages": [{"role": "user", "content": req["task"]}]}, stream_mode="updates")
    calls = (tc for update in stream for state in update.values()
             for msg in (state or {}).get("messages", []) or [] for tc in getattr(msg, "tool_calls", None) or [])
    try:                                         # closing the stream aborts the graph before the pick executes
        name = next((tc["name"] for tc in calls if tc["name"].startswith(PREFIX)), None)
        return (name[len(PREFIX):], name) if name else None
    finally:
        stream.close()


if __name__ == "__main__":
    serve_worker(select)
