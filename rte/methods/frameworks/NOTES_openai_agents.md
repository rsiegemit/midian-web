# fw_openai_agents — OpenAI Agents SDK

**Venv** `$RTE_DATA/env/fw_openai_agents` · **Worker** `workers/openai_agents_worker.py` · **Method** `fw_openai_agents.py`

## Pins actually installed
`openai-agents==0.22.0` (the SPEC pin), which pulls `openai==3.7.0`.

## Who makes the pick
A triage `Agent(name="triage", instructions=..., handoffs=[Agent(name, handoff_description) for each top-k
candidate])`. `agents/handoffs/__init__.py` turns each entry into a tool:

```python
Handoff.default_tool_name(agent)        -> f"transfer_to_{agent.name}"          # function-style transformed
Handoff.default_tool_description(agent) -> f"Handoff to the {agent.name} agent to handle the request. "
                                           f"{agent.handoff_description or ''}"
```

So the candidate's self-description goes in `handoff_description` and lands verbatim at the end of the tool
description. Name + description are the whole basis for the selection. The SDK adds no roster to the system
prompt: `RECOMMENDED_PROMPT_PREFIX` in `agents.extensions.handoff_prompt` exists but is **opt-in** and we do
not use it, so the tool list is the roster.

`Runner.run(triage, task, max_turns=1, hooks=..., run_config=...)`. `RunHooks.on_handoff(context, from_agent,
to_agent)` fires when the handoff is resolved and **before** the target agent's turn; the worker raises a
`Picked(to_agent.name)` exception out of the hook, so the run dies there and no candidate executes.

## Endpoint wiring
The SDK defaults to the **Responses API**, which vLLM does not serve. The worker therefore builds
`OpenAIProvider(openai_client=AsyncOpenAI(base_url=..., api_key=...), use_responses=False)` and passes it as
`RunConfig(model=<model>, model_provider=..., tracing_disabled=True,
model_settings=ModelSettings(temperature=0.0))`. That is the explicit, per-run equivalent of the global
`set_default_openai_client` + `set_default_openai_api("chat_completions")` pair, and it avoids process-global
state so the worker can serve different endpoints. Tracing must be disabled or the SDK tries to POST traces
to api.openai.com.

## Caveats
- Agent names must survive `transform_string_function_style`; our `agent_%06d` names pass through unchanged,
  and the worker keeps a `_safe(name) -> name` reverse map anyway.
- `max_turns=1` alone is not enough to guarantee no candidate runs (a handoff counts as part of turn 1); the
  exception from `on_handoff` is what actually stops it.

## What still needs a real model
Everything about routing quality. The mock returns a tool call for the first `transfer_to_agent_*` tool it
sees, so the test only proves the tools carry the descriptions and that the hook intercepts the pick.
