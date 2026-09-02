# fw_maf — Microsoft Agent Framework

**Venv** `$RTE_DATA/env/fw_maf` · **Worker** `workers/maf_worker.py` · **Method** `fw_maf.py`

## Pins actually installed
`agent-framework-core==1.16.0`, `agent-framework-openai==1.14.1`, `agent-framework-orchestrations==1.1.1`.
The `agent-framework==1.16.0` meta-package in SPEC §6A **cannot be installed on linux-x86_64**: its `[all]`
extra pulls `agent-framework-hyperlight`, which requires `hyperlight-sandbox-backend-wasm`, and that package
has no distribution for this platform (`ResolutionImpossible`). The three sub-packages above are the pieces
the selection primitive needs; `agent-framework-openai` and `-orchestrations` are on their own version lines
and 1.16.0 does not exist for either (newest are 1.14.1 and 1.1.1). Recorded in DEVIATIONS.md.

## `mode="groupchat"` (default) — who makes the pick
`GroupChatBuilder(participants=<top-k agents>, orchestrator_agent=<supervisor Agent>, max_rounds=1).build()`
→ `agent_framework_orchestrations._group_chat.AgentBasedGroupChatOrchestrator`.

Each round it calls the orchestrator with `options={"response_format": AgentOrchestrationOutput}` and appends
this instruction as a **user** message (`_group_chat.py::_invoke_agent`, verbatim):

```
Decide what to do next. Respond with a JSON object of the following format:
{
  "terminate": <true|false>,
  "reason": "<explanation for the decision>",
  "next_speaker": "<name of the next participant to speak (if not terminating)>",
  "final_message": "<optional final message if terminating>"
}
If not terminating, here are the valid participant names (case-sensitive) and their descriptions:
<name>: <description>
<name>: <description>
...
```

So the roster is exactly `name: description` per participant — the agent's `description=` field is the only
thing about a candidate the orchestrator sees. `AgentOrchestrationOutput` is a pydantic model with
`model_config = {"extra": "forbid"}`, so the model's JSON must contain those four keys **and nothing else**.

The pick surfaces in `_base_group_chat_orchestrator.py::_send_request` as
`WorkflowEvent("group_chat", data=GroupChatRequestSentEvent(round_index, participant_name))`, emitted
immediately after `ctx.send_message(...)` and before the participant produces anything. The worker breaks
out of `workflow.run(task, stream=True)` at that event, so no candidate agent ever runs.

## `mode="handoff"`
`HandoffBuilder(participants=[triage] + top-k).add_handoff(triage, top-k).with_start_agent(triage).build()`.
`_handoff.py::get_handoff_tool_name` names one tool per target, `handoff_to_<agent name>`; `add_handoff`
defaults each tool's description to **the target agent's `description`**, so again name + self-description is
all the model selects on. `_AutoHandoffMiddleware` short-circuits the tool call, then the orchestrator emits
`WorkflowEvent("handoff_sent", data=HandoffSentEvent(source, target))` and the worker returns `target`.

## Caveats
- Handoff mode requires `require_per_service_call_history_persistence=True` on **every** participant; the
  builder raises otherwise. Group-chat mode does not, and the worker sets the flag only for handoff.
- Handoff mode runs the agent in **streaming** mode (`"stream": true` chat-completions request). The mock
  server had to grow an SSE branch for it; a real vLLM server handles this natively.
- `ResponseStream` has no `aclose`, so returning at the pick simply abandons the run; the pending task is
  destroyed when the worker's loop moves on. Aborting mid-round is noisy: OpenTelemetry raises
  `ValueError: <Token ...> was created in a different Context` from its span teardown and asyncio logs
  "Task was destroyed but it is pending". Both are harmless artefacts of killing the run after the pick; the
  worker silences the `opentelemetry`, `asyncio`, `agent_framework` and `agent_framework_orchestrations`
  loggers and installs a no-op loop exception handler.
- `HandoffBuilder` logs "No handoff configuration found for agent X" for every non-triage participant. That is
  expected: only the triage agent is given handoff targets, which is what makes it a router.
- Ordering: `HandoffBuilder` keeps a source's targets in a `set`, so the `handoff_to_<name>` tool order is
  Python's randomized string-hash order and differs run to run (measured: 5 runs over the same 10 candidates
  picked agents 2, 3, 7, 0, 5). `tests/test_fw_b.py` therefore asserts only top-k membership for this mode
  while every other recipe is held to "the pick is candidate 0".

## What still needs a real model
The mock always answers with the first candidate, so these tests prove plumbing only: that the roster reaches
the model, that the pick is intercepted at the right event, and that no candidate executes. Routing quality
(does the orchestrator pick the honestly-best-described agent, and does it get fooled by inflated
self-descriptions) needs Qwen2.5-7B-Instruct on a real vLLM endpoint.
