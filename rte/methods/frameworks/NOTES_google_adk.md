# fw_google_adk — Google Agent Development Kit

**Venv** `$RTE_DATA/env/fw_google_adk` · **Worker** `workers/google_adk_worker.py` · **Method** `fw_google_adk.py`

## Pins actually installed
`google-adk==2.8.0` (the SPEC pin), plus `litellm==1.99.0` and `google-genai==2.21.0` as dependencies.

## Who makes the pick
A router `LlmAgent(name="router", instruction=..., model=LiteLlm(...), sub_agents=[LlmAgent(name,
description) for each top-k candidate])`. ADK's auto-delegation is implemented in
`google/adk/flows/llm_flows/agent_transfer.py`: because the router has sub-agents, the processor attaches a
`transfer_to_agent` tool and appends this instruction (`_build_transfer_instruction_body`, verbatim):

```
You have a list of other agents to transfer to:

Agent name: {name}
Agent description: {description}

Agent name: {name}
Agent description: {description}
...

If you are the best to answer the question according to your description,
you can answer it.

If another agent is better for answering the question according to its
description, call `transfer_to_agent` function to transfer the question to that
agent. When transferring, do not generate any text other than the function
call.

**NOTE**: the only available agents for `transfer_to_agent` function are
`agent_000001`, `agent_000002`, ...
```

Name + description, nothing else. The pick surfaces as the first event with
`event.actions.transfer_to_agent` set; the worker returns it and stops iterating `runner.run_async(...)`, so
the sub-agent never runs.

## Endpoint wiring
`LiteLlm(model="openai/<model>", api_base=<base_url>, api_key=<key>, temperature=0.0)` — LiteLLM's
`openai/` prefix routes to the OpenAI chat-completions API against `api_base`, which is what vLLM serves.
`InMemoryRunner(agent=router, app_name="rte")` plus a session from `runner.session_service.create_session`.

## Caveats
- **The `google` namespace package in `~/.local/lib/python3.12/site-packages` shadows `google.adk`.** Conda
  prefixes honour the user site directory, so the worker strips it from `sys.path` before importing anything;
  the env build scripts export `PYTHONNOUSERSITE=1` for the same reason.
- ADK prints two notices per request: an "App can transfer between agents but has no context_cache_config"
  info line and a `UserWarning: [EXPERIMENTAL] feature FeatureName.JSON_SCHEMA_FOR_FUNC_DECL is enabled`
  from `transfer_to_agent_tool`. The worker silences both. Neither affects the pick.
- Aborting `run_async` mid-stream makes OpenTelemetry log `Failed to detach context` /
  `ValueError: <Token ...> was created in a different Context`. Harmless artefact of stopping at the pick.
- LiteLLM is a second translation layer between ADK and vLLM, so ADK's latency and its exact wire format are
  LiteLLM's, not ADK's. Worth stating in the paper when reporting per-framework overhead.

## What still needs a real model
Routing quality. The mock returns `transfer_to_agent(agent_name=<first candidate>)` regardless of the roster.


## 2026-09-03 fix (v2 work order 0.2)
- `event.actions.transfer_to_agent` capture verified on the fleet (transfer events arrive with the sanitized
  sub-agent name; mapped back through `sanitize`). A run that ends without any transfer event means the router
  answered the task itself; the worker now returns `FAILURE: router answered itself` and the adapter counts it as a
  failure (strict success 0) instead of a fallback.
