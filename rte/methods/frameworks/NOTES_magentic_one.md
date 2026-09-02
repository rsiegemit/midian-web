# fw_magentic_one — what actually makes the pick

**Versions installed**: shares `$RTE_DATA/env/fw_autogen` (`autogen-agentchat 0.7.5`, `autogen-ext 0.7.5`).

**Primitive.** `MagenticOneGroupChat`'s `MagenticOneOrchestrator`. Per step it asks the model for a
**progress ledger** and dispatches `progress_ledger["next_speaker"]["answer"]`
(`_magentic_one_orchestrator.py:428`). The ledger is the pydantic model `LedgerEntry`, five fields each
shaped `{reason, answer}`.

**What the model sees.** Three prompts, in this order (`_magentic_one/_prompts.py`), before any pick:

1. `ORCHESTRATOR_TASK_LEDGER_FACTS_PROMPT` — a "pre-survey" of facts about the task. No roster.
2. `ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT` — `"Fantastic. To address this request we have assembled the
   following team:\n\n{team}\n\nBased on the team composition ... devise a short bullet-point plan"`.
3. `ORCHESTRATOR_PROGRESS_LEDGER_PROMPT` — the one that selects:

```
Recall we are working on the following request:

{task}

And we have assembled the following team:

{team}

To make progress on the request, please answer the following questions, including necessary reasoning:

    - Is the request fully satisfied? ...
    - Are we in a loop ...
    - Are we making forward progress? ...
    - Who should speak next? (select from: {names})
    - What instruction or question would you give this team member? ...

Please output an answer in pure JSON format according to the following schema. ...
```

`{team}` is the orchestrator's `_team_description`, built from participant `name` + `description`, so the
self-description goes in `AssistantAgent(description=...)`. `{names}` is the participant name list.

**Interception.** After validating the ledger the orchestrator calls
`_log_message(f"Next Speaker: {progress_ledger['next_speaker']['answer']}")` and only then publishes the
work request. The worker replaces `MagenticOneOrchestrator._log_message` with a version that raises on that
prefix, which propagates cleanly out of `run_stream`. The `SelectSpeakerEvent` path (`line 444`) is kept as
a fallback. Measured against the mock: **3** chat-completions requests per `fetch` (facts, plan, ledger),
no worker agent executed.

**Caveats.**
- Magentic-One is roughly 3x the model calls of every other rival here, before it even picks, because its
  outer loop always writes a fact sheet and a plan first. That is inherent to the framework and should be
  reported as its cost, not engineered away.
- Without the `_log_message` patch the run continues past the pick: measured 5 calls per selection (a second
  ledger round plus the final-answer prompt) even with `max_turns=1` and inert participants.
- We declare `structured_output=True` in `model_info` so the orchestrator sends
  `json_output=LedgerEntry`, i.e. an OpenAI `response_format` JSON schema. With `structured_output=False`
  it asks for free-form JSON and every mis-shaped reply costs another model call (`max_json_retries`,
  default 10). **This needs validating on the real vLLM endpoint**: Qwen2.5-7B-Instruct served by vLLM must
  accept `response_format={"type":"json_schema", ...}` (vLLM's guided decoding). If it does not, drop
  `structured_output` to `False` and expect a higher and more variable call count.
- If the model names an agent that is not a participant, the orchestrator raises
  `ValueError("Invalid next speaker: ...")` rather than retrying; the bridge turns that into an error and
  `FrameworkMethod` counts a fallback.
