# fw_autogen — what actually makes the pick

**Versions installed** (`$RTE_DATA/env/fw_autogen`, shared with `fw_magentic_one`):
`autogen-agentchat 0.7.5`, `autogen-core 0.7.5`, `autogen-ext 0.7.5`, `openai 3.7.0`, `pydantic 2.13.5`.

**Primitive.** `SelectorGroupChat`. Its manager builds the roster in
`_selector_group_chat.py:205-207`:

```python
roles = ""
for topic_type, description in ...:
    roles += re.sub(r"\s+", " ", f"{topic_type}: {description}").strip() + "\n"
```

`topic_type` is the participant `name` and `description` is `ChatAgent.description` — so the
self-description goes in `AssistantAgent(description=...)`. Note that `re.sub(r"\s+", " ", ...)` collapses
every newline in a description onto one line.

**What the model sees.** The default `selector_prompt` (`_selector_group_chat.py:607`), formatted with
`{roles}`, `{participants}` (a `str()` of the name list) and `{history}`:

```
You are in a role play game. The following roles are available:
{roles}.
Read the following conversation. Then select the next role from {participants} to play. Only return the role.

{history}

Read the above conversation. Then select the next role from {participants} to play. Only return the role.
```

Because our `model_info["family"]` is `unknown` (not an OpenAI family), that prompt is sent as a **user**
message rather than a system message (`_select_speaker`, line 241). One model call, no tools.

**Interception.** The reply is matched against participant names by `_mentioned_agents`; a valid pick is
returned to the base group-chat manager, which emits `SelectSpeakerEvent(content=[name])` because we pass
`emit_team_events=True` (`_base_group_chat_manager.py:232`). The worker breaks out of `run_stream` at that
event and returns `content[0]`.

**Caveats.**
- The manager publishes the work request to the chosen speaker **before** emitting `SelectSpeakerEvent`,
  and the team keeps running in a background task while we consume the stream — so breaking at the event
  is a race, not a guarantee. Measured against the mock, a plain `AssistantAgent` participant did spend one
  model call answering the task. The worker therefore makes participants an `Idle(AssistantAgent)` subclass
  whose `on_messages` returns an empty `Response` without touching the model. With that, the measured cost
  is exactly **one** chat-completions request per `fetch`, and no worker agent executes anything.
- `MaxMessageTermination(2)` is a backstop only. `MaxMessageTermination(1)` is not usable: the termination
  condition is evaluated on the task message itself, so the team would stop before selecting a speaker.
- `allow_repeated_speaker=True` — otherwise the manager rejects a repeat of the previous speaker, which is
  meaningless for us since every `fetch` is a fresh single-turn team.
- `_mentioned_agents` requires exactly one *distinct* participant name in the reply; a model that names two
  candidates while reasoning burns a retry (`max_selector_attempts`, default 3) and then falls back to
  `participants[0]`. That fallback is invisible to us — it arrives as a normal `SelectSpeakerEvent` — so on
  a real model the pick rate reported by `FrameworkMethod.stats` will not distinguish a considered pick from
  a give-up. Needs a real model to quantify.
- `model_info` must be supplied explicitly for a non-OpenAI model id; we declare
  `function_calling=True, json_output=True, structured_output=False, family=unknown`.
