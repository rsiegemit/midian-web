"""AgentScope rival (SPEC §6A appendix), run inside $RTE_DATA/env/fw_agentscope.

AgentScope 2.0.7 ships NO multi-agent selection primitive (no supervisor, no handoff, no group chat
with a speaker selector), so this is a DIY structured-output router and is reported as such: one
`OpenAIChatModel` call carrying the roster, constrained by a json_schema `response_format` to
{"agent": <name>}. Only AgentScope's model/message layer is under test here, not a routing primitive.
A single event loop is reused for the life of the process.
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bridge import serve_worker   # noqa: E402

LOOP = asyncio.new_event_loop()
SCHEMA = {"type": "json_schema", "json_schema": {"name": "route", "strict": True, "schema": {
    "type": "object", "properties": {"agent": {"type": "string"}},
    "required": ["agent"], "additionalProperties": False}}}
PROMPT = ("You are routing one task to exactly one agent.\n\nAgents:\n{roster}\n\nTask:\n{task}\n\n"
          'Answer with JSON {{"agent": "<the name of the single best agent>"}} and nothing else.')
_MODEL = {}


def _model(req):
    from agentscope.credential import OpenAICredential
    from agentscope.model import OpenAIChatModel
    key = (req["model"], req["base_url"], req["api_key"])
    if _MODEL.get("key") != key:
        _MODEL["key"] = key
        _MODEL["m"] = OpenAIChatModel(
            credential=OpenAICredential(api_key=req["api_key"], base_url=req["base_url"]),
            model=req["model"], stream=False,
            parameters=OpenAIChatModel.Parameters(temperature=0.0))
    return _MODEL["m"]


def select(req):
    from agentscope.message import Msg, TextBlock
    roster = "\n".join(f"- {c['name']}: {c['description']}" for c in req["candidates"])
    text = PROMPT.format(roster=roster, task=req["task"])
    msgs = [Msg(name="user", role="user", content=[TextBlock(type="text", text=text)])]
    resp = LOOP.run_until_complete(_model(req)(msgs, response_format=SCHEMA))
    out = "".join(b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "") for b in resp.content)
    return json.loads(out)["agent"], out[:500]


if __name__ == "__main__":
    serve_worker(select)
