# fw_smolagents — HF smolagents 1.26.0

**Venv** `$RTE_DATA/env/fw_smolagents` (python 3.12) · **worker** `workers/smolagents_worker.py` · **method** `fw_smolagents.py`

## What makes the pick
`smolagents.agents.ToolCallingAgent`. Managed agents are exposed to the model as ordinary tools
(`ToolCallingAgent.tools_and_managed_agents` = `list(self.tools.values()) + list(self.managed_agents.values())`,
`agents.py:1261`), so the routing decision *is* the tool name in the model's first tool call. Each managed
agent is a `ToolCallingAgent(tools=[], model=…, name=<candidate name>, description=<self-description>)`.

The name is the only identifier the model ever sees, and smolagents requires it to be a valid Python
identifier (`MultiStepAgent._validate_name` → `is_valid_name`, `agents.py:364`) — the one framework of the
four with a name constraint, so the worker runs candidate names through `workers/_wk.sanitize` and maps the
pick back. Our names are `agent_%06d`, so today the map is the identity; the call is what keeps it correct
if `_common` ever names agents differently.

## Where we intercept
SPEC §6A recipe 8 says `step_callbacks` + `ActionStep.tool_calls[0].name`. In smolagents 1.26.0 step
callbacks run in `MultiStepAgent._finalize_step` (`agents.py:623`), which is reached only **after**
`process_tool_calls` has already executed the chosen managed agent — i.e. after a full sub-agent run.
`process_tool_calls` yields the `ToolCall` object *before* calling `execute_tool_call` (`agents.py:1375-1383`),
and `_step_stream` re-yields it, so we consume `agent.run(task, stream=True)` and return on the first
`smolagents.memory.ToolCall`, then close the generator. Same value as `ActionStep.tool_calls[0].name`,
one step earlier, with nothing executed. Recorded in DEVIATIONS.md.

`stream.close()` raises `RuntimeError("generator ignored GeneratorExit")` because `_run_stream` has a
`finally: yield action_step`; we swallow it — the run is dead either way.

## Exact prompt template
`smolagents/prompts/toolcalling_agent.yaml`, `system_prompt`, the managed-agents block (lines 97-107):

```
You can also give tasks to team members.
Calling a team member works similarly to calling a tool: provide the task description as the 'task' argument. Since this team member is a real human, be as detailed and verbose as necessary in your task description.
You can also include any relevant variables or context using the 'additional_args' argument.
Here is a list of the team members that you can call:
{%- for agent in managed_agents.values() %}
- {{ agent.name }}: {{ agent.description }}
  - Takes inputs: {{agent.inputs}}
  - Returns an output of type: {{agent.output_type}}
{%- endfor %}
```

The same name/description pair is also sent as an OpenAI tool schema (`models.get_tool_json_schema`), so the
model sees the roster twice: once in the system prompt, once in `tools`.

## Model client
`OpenAIServerModel` (an alias of `OpenAIModel`, `models.py:1796`), built from `workers/_wk.openai_kwargs(req)`
as `OpenAIServerModel(model_id, api_base, api_key, temperature=0.0)`. Constructed per request: it opens no
connection, and caching it was duplicated plumbing for no measurable gain.

## Caveats
- `max_steps=1` on the outer agent and on every managed agent, `tools=[]` everywhere. The only base tool
  smolagents adds is `final_answer`; if the model calls that instead of a team member, the bridge returns
  `final_answer` and `_common.FrameworkMethod` counts it as `bad_name` and falls back to declared-argmax.
- Roster order is the retrieval order from `_common.retrieve`, and smolagents preserves it in both the
  prompt and the tool list — so any position bias in the supervisor model is measurable but not controlled.
