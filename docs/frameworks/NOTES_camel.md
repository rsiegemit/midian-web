# fw_camel_workforce — CAMEL 0.2.90 (`camel-ai`)

**Venv** `$RTE_DATA/env/fw_camel` (python 3.12) · **worker** `workers/camel_worker.py` · **method** `fw_camel_workforce.py`

## What makes the pick
`camel.societies.workforce.Workforce`. Its coordinator agent is asked, once per batch of pending tasks, to
return JSON assignments; `Workforce._call_coordinator_for_assignment` (`workforce.py:3693`) formats
`ASSIGN_TASK_PROMPT` and the result's `assignments[].assignee_id` is the chosen **node id**.

Roster line, from `Workforce._get_child_nodes_info` (`workforce.py:3603`):

```
f"<{child.node_id}>:<{child.description}>:<{self._get_node_info(child)}>\n"
```

so with our candidates it renders exactly `<agent_000041>:<...self-description...>:<>` (empty toolkit field,
since the workers carry no tools). The candidate name therefore has to live in `node_id`. `Workforce.
add_single_agent_worker(description, worker)` has no `node_id` parameter and `BaseNode.__init__` defaults it
to `str(id(self))` (`workforce/base.py:33`), so the worker sets `wf._children[-1].node_id = name` right after
adding. That is the one private-attribute touch in this rival.

## Where we intercept
SPEC §6A recipe 9. `Workforce._post_ready_tasks` builds a `TaskAssignedEvent(task_id, worker_id=assignment.
assignee_id, …)` and calls `cb.log_task_assigned(event)` for every registered callback (`workforce.py:4371-4380`)
**before** posting the task to the channel. Our `WorkforceCallback` subclass raises out of
`log_task_assigned`, which propagates through `process_task`, so no `SingleAgentWorker` ever runs.
`WorkforceCallback` is an ABC with 11 abstract `log_*` methods, so the subclass is built with `type()` over
no-ops plus the one real handler.

## Exact prompt template
`camel/societies/workforce/prompts.py`, `ASSIGN_TASK_PROMPT` (abridged; `{tasks_info}` is
`"Task ID: {id}\nContent: {content}\n---\n"` per task, `{child_nodes_info}` is the roster above):

```
You need to assign multiple tasks to worker nodes based on the information below.
...
Your response MUST be a valid JSON object containing an 'assignments' field with a list of task assignment dictionaries.
Each assignment dictionary should have:
- "task_id": the ID of the task
- "assignee_id": the ID of the chosen worker node
- "dependencies": list of task IDs that this task depends on (empty list if no dependencies)
...
Here are the tasks to be assigned:
==============================
{tasks_info}
==============================

Following is the information of the existing worker nodes. The format is <ID>:<description>:<toolkit_info and skill names>. Choose the most capable worker node ID for each task.

==============================
{child_nodes_info}
==============================
```

`use_structured_output_handler` defaults to `True`, so CAMEL asks for this JSON in the prompt and extracts it
with regex rather than using the OpenAI `response_format` parameter.

## Model client
`ModelFactory.create(model_platform=ModelPlatformType.VLLM, …)` over `workers/_wk.openai_kwargs(req)`,
with `model_config_dict={"temperature": 0.0}`. `default_model=` is passed to `Workforce` as well as
explicit `coordinator_agent`/`task_agent`, otherwise CAMEL falls back to `ModelPlatformType.DEFAULT` and
demands a real `OPENAI_API_KEY`.

## Caveats
- camel-ai 0.2.90 declares `mcp>=1.3.0` but imports `mcp.server.FastMCP`, which mcp 2.x removed; the
  requirements file pins `mcp<2`. See DEVIATIONS.md.
- If the coordinator returns an unusable assignment, CAMEL's fallback **invents a new worker** (a fresh uuid
  node id) and assigns to it. That id is not a candidate name, so `_common.FrameworkMethod` records it as
  `bad_name` and falls back to declared-argmax — a real-model failure mode worth reporting per run.
- A fresh `Workforce` (and k+2 `ChatAgent`s) is constructed per `fetch`; `wf.stop()` runs in a `finally`.
