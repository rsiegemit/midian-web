# fw_langgraph — what actually makes the pick

**Versions installed** (`$RTE_DATA/env/fw_langgraph`): `langgraph 1.2.11`, `langgraph-supervisor 0.0.31`,
`langgraph-prebuilt 1.1.0`, `langchain-core 1.6.1`, `langchain-openai 1.6.0`, `openai 3.7.0`.

**Primitive.** `langgraph_supervisor.create_supervisor(agents, model=..., prompt=...)`
(`langgraph_supervisor/supervisor.py:211`) builds a `StateGraph` whose `supervisor` node is a
`create_react_agent` bound to one handoff tool per managed agent. The tools come from
`create_handoff_tool(agent_name=...)` (`langgraph_supervisor/handoff.py:56`), which names each tool
`transfer_to_<_normalize_agent_name(agent_name)>` — normalization is `whitespace -> "_"` plus `.lower()`.
Our ids `agent_000123` survive it unchanged, so the reverse map is the identity after stripping
`transfer_to_`.

**What the model sees.** Two things, and only two:

1. The system message = our `prompt` verbatim. LangGraph injects nothing about the agents into it, which
   is why the SPEC puts the roster there. The template the worker constructs is exactly:

   ```
   You are a supervisor managing a team of agents:
   - agent_000005: <self-description of agent 5>
   - agent_000009: <self-description of agent 9>
   ...

   Assign the user's task to exactly one agent by calling its transfer tool. Do not do any work yourself.
   ```

2. One OpenAI tool per candidate, whose auto-generated description is
   `Ask agent '<agent_name>' for help` and whose parameter schema is **empty** (`state` and `tool_call_id`
   are `InjectedState`/`InjectedToolCallId`, so they never reach the model). The self-description is
   therefore *not* in the tool schema — the roster in the prompt is the only place it appears.

Then a single user message with the task text.

**Interception.** `app.stream(..., stream_mode="updates")`; the first yielded update is the `supervisor`
node, already containing the `AIMessage` with the `transfer_to_*` tool call. We return that name and
close the generator, so the graph never advances to the selected agent's node. Verified against
`scripts/mock_openai_server.py`: exactly **one** chat-completions request per `fetch`. Worker agents are
`create_react_agent(model, tools=[])` as a second safety net — with no tools they could only emit text —
but they are never reached.

**Caveats.**
- `create_supervisor` collects agent names into a **`set`** (`supervisor.py:397`, used at line 185 to build
  the handoff tools), so the order of the tools presented to the model is Python-hash order, not the order
  we pass the candidates in. The roster in the prompt *is* ordered. Any position bias a real model has will
  therefore be driven by the prompt roster, while the tool list order is effectively arbitrary. Under the
  mock (which answers with the first tool in the list) this shows up as a pick that is a valid candidate but
  not necessarily candidate 0 — the tests assert membership in the retrieved top-k, not identity.
- `langgraph.prebuilt.create_react_agent` emits a `LangGraphDeprecatedSinceV10` warning (moved to
  `langchain.agents.create_agent` in V1). We keep the `langgraph.prebuilt` entry point because it is what
  `langgraph-supervisor` itself imports, and silence warnings in the worker.
- Deterministic decoding is requested with `temperature=0` on `ChatOpenAI`; `max_retries=0` so a broken
  endpoint surfaces as an error rather than being silently retried.
