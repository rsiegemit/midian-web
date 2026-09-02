"""LangGraph rival worker (SPEC §6A recipe 1). Selection primitive: `langgraph_supervisor.create_supervisor`,
whose supervisor node is handed one `transfer_to_<name>` handoff tool per agent. The library does NOT inject
agent descriptions, so the roster (name: description) goes into the supervisor `prompt`, as the docs prescribe.
We stream `updates` and stop at the supervisor's first `transfer_to_*` tool call, so no worker agent ever runs.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bridge import serve_worker  # noqa: E402

warnings.filterwarnings("ignore")               # langgraph V1 deprecation notice on create_react_agent
from langchain_openai import ChatOpenAI          # noqa: E402
from langgraph.prebuilt import create_react_agent  # noqa: E402
from langgraph_supervisor import create_supervisor  # noqa: E402

PROMPT = ("You are a supervisor managing a team of agents:\n{roster}\n\n"
          "Assign the user's task to exactly one agent by calling its transfer tool. "
          "Do not do any work yourself.")
PREFIX = "transfer_to_"


def select(req):
    cands = req["candidates"]
    model = ChatOpenAI(model=req["model"], base_url=req["base_url"], api_key=req["api_key"] or "EMPTY",
                       temperature=0, max_retries=0)
    agents = [create_react_agent(model, tools=[], name=c["name"], prompt="Answer the task.") for c in cands]
    roster = "\n".join(f"- {c['name']}: {c['description']}" for c in cands)
    app = create_supervisor(agents, model=model, prompt=PROMPT.format(roster=roster)).compile()
    stream = app.stream({"messages": [{"role": "user", "content": req["task"]}]}, stream_mode="updates")
    try:
        for update in stream:
            for state in update.values():
                for msg in (state or {}).get("messages", []) or []:
                    for tc in getattr(msg, "tool_calls", None) or []:
                        if tc["name"].startswith(PREFIX):
                            return tc["name"][len(PREFIX):], tc["name"][:500]
    finally:
        stream.close()          # abort the graph before the chosen agent executes
    return None


if __name__ == "__main__":
    serve_worker(select)
