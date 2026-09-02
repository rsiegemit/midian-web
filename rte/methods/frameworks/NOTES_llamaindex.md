# fw_llamaindex — LlamaIndex

**Venv** `$RTE_DATA/env/fw_llamaindex` · **Worker** `workers/llamaindex_worker.py` · **Method** `fw_llamaindex.py`

## Pins actually installed
`llama-index-core==0.14.24` (the SPEC pin), `llama-index-llms-openai-like==0.8.0`, and an explicit `numpy`
(see caveats).

## `mode="selector"` (default) — who makes the pick
`LLMSingleSelector.from_defaults(llm=OpenAILike(...)).aselect([ToolMetadata(name, description)] * k, task)`.
This is pure selection: one LLM call, no agent, no tool execution, nothing to abort.

`llama_index/core/selectors/prompts.py::DEFAULT_SINGLE_SELECT_PROMPT_TMPL` (verbatim):

```
Some choices are given below. It is provided in a numbered list (1 to {num_choices}),
where each item in the list corresponds to a summary.
---------------------
{context_list}
---------------------
Using only the choices above and not prior knowledge, return the choice that is most relevant to the question: '{query_str}'
```

with `SelectionOutputParser.format` appending:

```
The output should be ONLY JSON formatted as a JSON instance.

Here is an example:
[
    {
        choice: 1,
        reason: "<insert reason for choice>"
    },
    ...
]
```

`_build_choices_text` renders each entry as `(<1-based index>) <description>` — **the `ToolMetadata.name` is
never shown to the model**, only the description. That is a real property of this framework, not a bug in our
adapter: LlamaIndex's single selector routes on self-descriptions alone. The reply is parsed by
`SelectionOutputParser`, and `_structured_output_to_selector_result` converts `choice` to a 0-based index
(`index = answer.choice - 1`), which the worker uses to look up the candidate.

## `mode="handoff"`
`AgentWorkflow(agents=[triage] + top-k FunctionAgents, root_agent="triage")`. The built-in `handoff(ctx,
to_agent, reason)` tool is described by `prompts.py::DEFAULT_HANDOFF_PROMPT`:

```
Useful for handing off to another agent.
If you are currently not equipped to handle the user's request, or another agent is better suited to handle the request, please hand off to the appropriate agent.

Currently available agents:
{agent_info}
```

where `agent_info` is `{cfg.name: cfg.description for cfg in self.agents.values()}` — so here the names *are*
visible, unlike the selector path. The pick is the first `ToolCall(tool_name="handoff").tool_kwargs["to_agent"]`
from `handler.stream_events()`; the worker returns it and calls `handler.cancel_run()`, so no candidate runs.

## Caveats
- `OpenAILike` must be constructed with **both** `is_chat_model=True` and `is_function_calling_model=True`.
  Without the second flag `FunctionAgent.take_step` raises `ValueError: LLM must be a FunctionCallingLLM`,
  and because `AgentWorkflow` only surfaces that when the handler is awaited, the failure is silent if you
  merely iterate `stream_events()`. The worker awaits the handler after the loop for exactly this reason.
- `llama-index-core` declares numpy, but pip saw the numpy in `~/.local/lib/python3.12/site-packages` as
  satisfying it and skipped the install; the env then broke as soon as the user site was hidden. `numpy` is
  now pinned explicitly in `requirements-frameworks/llamaindex.txt`.
- The mock server's `"choice"` field had to become 1-based to match this parser (see DEVIATIONS.md); the
  0-based `"index"` key it also returns is untouched.

## What still needs a real model
Routing quality, and in particular how much the selector's name-blindness matters: with a real model the
selector sees only descriptions, so a dishonest self-description has nothing to contradict it.
