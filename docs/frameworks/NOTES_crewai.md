# fw_crewai — what actually makes the pick

**Versions installed** (`$RTE_DATA/env/fw_crewai`): `crewai 1.15.18` (`crewai-core`, `crewai-cli` same
version), `openai 2.54.0`, `instructor 1.16.0`, `pydantic 2.12.5`, plus `numpy` (see caveats).

**Primitive.** `Crew(process=Process.hierarchical, manager_llm=...)`. At kickoff,
`Crew._create_manager_agent` (`crewai/crew.py:1518`) builds an un-declared manager agent with the fixed
persona `role="Crew Manager"`, and gives it `AgentTools(agents=self.agents).tools()` —
`Delegate work to coworker` and `Ask question to coworker`. The manager's choice is the `coworker`
argument of the delegate tool.

**What the model sees.** `AgentTools.tools()` (`crewai/tools/agent_tools/agent_tools.py`) formats
`coworkers = ", ".join(agent.role for agent in self.agents)` into the tool description from
`crewai/translations/en.json`:

```
Delegate a specific task to one of the following coworkers: {coworkers}
The input to this tool should be the coworker, the task you want them to do, and ALL necessary context
to execute the task, they know nothing about the task, so share absolutely everything you know, don't
reference things but instead explain them.
```

`role` is the **only** agent attribute in that string — `goal` and `backstory` never reach the manager.
That is why the worker sets `role = f"{agent_id}: {self-description}"`. The id prefix keeps the pick
invertible; `BaseAgentTool._execute` resolves the coworker back to an agent by
`sanitize_agent_name` (whitespace-collapse + `casefold` + strip quotes) equality on `role`, so we map
back by exact role first and by an `agent_\d{6}` match in the returned string second.

The manager's own system message is the fixed `hierarchical_manager_agent` persona ("You are a seasoned
manager with a knack for getting the best out of your team...").

**Interception.** SPEC recipe 2 asks for `ToolUsageStartedEvent`. In 1.15.18 the event bus dispatches
handlers on a `ThreadPoolExecutor` (`crewai/events/event_bus.py:572`), so a handler cannot stop the run
and its value is not available synchronously. We therefore read the same `coworker` argument one frame
later, at `BaseAgentTool._execute`, which the worker replaces with a function that raises. The exception
subclasses **`BaseException`**, not `Exception`: `_execute` and CrewAI's tool-usage layer both wrap the
delegated call in `except Exception` and would otherwise turn the abort into an error string fed back to
the manager, costing extra model calls. Verified against `scripts/mock_openai_server.py`: exactly **one**
chat-completions request per `fetch`, and `selected_agent.execute_task` is never reached.

**Caveats.**
- The roster is joined with `", "`, and our self-descriptions contain commas, so the coworker list the
  manager reads has ambiguous item boundaries. That is CrewAI's own formatting and we keep it.
- `numpy` is not declared by `crewai 1.15.18` but is imported by `crewai.rag.embeddings.factory` on the
  `crewai.LLM` import path; it is pinned explicitly in `requirements-frameworks/crewai.txt`.
- Telemetry is opted out in the worker (`CREWAI_TELEMETRY_OPT_OUT`, `OTEL_SDK_DISABLED`) so compute nodes
  without internet do not stall on an outbound connection.
- Deterministic decoding via `LLM(..., temperature=0)`; the model is addressed as `openai/<model id>`
  against `base_url`.
- **CrewAI writes to stdout, which is the bridge's JSON-lines protocol channel.** A Rich "Tracing
  Preference Saved" panel on first run, and `[CrewAIEventsBus] Warning: ...` lines whenever an event pair
  does not close, all go to `sys.stdout` and desynchronize the bridge (the reader gets a non-JSON line,
  kills the worker and counts an error). The worker runs the whole crew inside
  `contextlib.redirect_stdout(sys.stderr)`. Measured over 150 consecutive requests afterwards: 150 JSON
  lines, 0 stray lines. No other rival here prints to stdout, but it is worth checking per framework.
- Aborting inside the delegate tool leaves CrewAI's `tool_usage_started` scope unclosed. The scope stack
  lives in a process-wide `ContextVar` and `push_event_scope` raises `StackDepthExceededError` at depth
  100, so a long-lived worker began failing after ~100 requests (seen as one fallback in a 40-fetch run,
  and it would dominate a Q=1000 grid cell). The worker calls
  `crewai.events.event_context.restore_event_scope(())` at the top of every request.


## 2026-09-03 fix (v2 work order 0.2)
- **Root cause of the 79–84% "fallback" in the 2026-09-02 runs was infrastructure, not CrewAI's manager.** CrewAI resets
  a SQLite `latest_kickoff_task_outputs.db` under the user data dir at every kickoff; hundreds of concurrent workers on
  NFS corrupted it (`DatabaseOperationError: Error deleting task outputs: database disk image is malformed`, file mtime
  11:05) and every kickoff after that failed before the manager ran — the bridge reported an error and the adapter
  counted a fallback. The worker now sets `CREWAI_STORAGE_DIR` to a private temp dir per process.
- The manager is an explicit `manager_agent` ("Dispatcher": delegate every task to exactly one coworker, never solve it
  yourself), every worker has `allow_delegation=True`, and the task reads "Delegate this to exactly one coworker: ...".
  The first `Delegate work to coworker` call is still intercepted at `BaseAgentTool._execute` (one frame after the
  tool call that `step_callback` would see; the event bus is asynchronous, see above). A kickoff that finishes without a
  delegate call returns `FAILURE: manager answered itself`, which the adapter counts as a failure (strict success 0).
- Live check (fleet, 7B): synthesized descriptions 6/6 delegations; real specialist self-descriptions: see the v2
  report / DEVIATIONS.
