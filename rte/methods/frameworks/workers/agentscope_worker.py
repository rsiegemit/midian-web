"""AgentScope has no selection primitive, so this is a DIY structured-output router (reported as such):
one OpenAIChatModel call, json_schema-constrained to {"agent": <name>}."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bridge import serve_worker            # noqa: E402
from _wk import openai_kwargs, run_async    # noqa: E402

SCHEMA = {"type": "json_schema", "json_schema": {"name": "route", "strict": True, "schema": {
    "type": "object", "properties": {"agent": {"type": "string"}},
    "required": ["agent"], "additionalProperties": False}}}
PROMPT = ("You are routing one task to exactly one agent.\n\nAgents:\n{roster}\n\nTask:\n{task}\n\n"
          'Answer with JSON {{"agent": "<the name of the single best agent>"}} and nothing else.')


def select(req):
    from agentscope.credential import OpenAICredential
    from agentscope.message import Msg, TextBlock
    from agentscope.model import OpenAIChatModel
    kw = openai_kwargs(req)
    model = OpenAIChatModel(credential=OpenAICredential(api_key=kw["api_key"], base_url=kw["base_url"]),
                            model=kw["model"], stream=False,
                            parameters=OpenAIChatModel.Parameters(temperature=0.0))
    roster = "\n".join(f"- {c['name']}: {c['description']}" for c in req["candidates"])
    text = PROMPT.format(roster=roster, task=req["task"])
    msgs = [Msg(name="user", role="user", content=[TextBlock(type="text", text=text)])]
    out = "".join(b.text for b in run_async(model(msgs, response_format=SCHEMA)).content)
    return json.loads(out)["agent"], out[:500]


if __name__ == "__main__":
    serve_worker(select)
