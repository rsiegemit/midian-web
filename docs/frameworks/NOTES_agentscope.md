# fw_agentscope — AgentScope 2.0.7 (appendix, **DIY router**)

**Venv** `$RTE_DATA/env/fw_agentscope` (python 3.12) · **worker** `workers/agentscope_worker.py` · **method** `fw_agentscope.py`

## What makes the pick — nothing in the library
AgentScope 2.0.7 has **no multi-agent selection primitive**. The package exposes `agentscope.agent`
(`Agent`, config dataclasses), `agentscope.model`, `agentscope.message`, `agentscope.tool`,
`agentscope.formatter` — there is no supervisor, no handoff tool, no group chat with a speaker selector,
and no `agentscope.pipeline` module (that existed in the 0.x line and is gone in 2.x). So unlike every
other row in SPEC §6A, this rival measures **our** prompt through AgentScope's model layer, not a routing
primitive the framework ships. It must be reported that way in the paper.

## What the worker actually does
One `agentscope.model.OpenAIChatModel` call carrying the roster, constrained to
`{"agent": "<name>"}` by an OpenAI `response_format` json_schema (`strict: true`,
`additionalProperties: false`). The reply is parsed with `json.loads`. Prompt template, in full, from
`workers/agentscope_worker.py`:

```
You are routing one task to exactly one agent.

Agents:
- <name>: <self-description>
  ... one line per top-k candidate ...

Task:
<task text>

Answer with JSON {"agent": "<the name of the single best agent>"} and nothing else.
```

## Model client
`OpenAIChatModel(credential=OpenAICredential(api_key=…, base_url=…), model=…, stream=False,
parameters=OpenAIChatModel.Parameters(temperature=0.0))`. `OpenAIChatModel.__call__` is a coroutine, so every request goes
through `workers/_wk.run_async`, which owns the single event loop shared by all framework workers.
Messages must be `Msg(name=…, role=…, content=[TextBlock(type="text", text=…)])`; `content` is
`list[ContentBlock]` and a bare `str` fails pydantic validation.

## Caveats
- **DIY**: no framework selection prompt exists to attribute results to. The comparison this row supports is
  "AgentScope's structured-output plumbing works", not "AgentScope routes well".
- An alternative would be `Agent.reply(..., structured_schema=…)`, but that runs AgentScope's ReAct loop,
  which would execute the selected agent — the opposite of a selection-only interception.
- Requires the endpoint to honour `response_format`; vLLM does (guided decoding), and the repo's
  `scripts/mock_openai_server.py` fills any json_schema it is handed.
