# fw_metagpt — MetaGPT (appendix; **heaviest caveat of the four**)

**Venv** `$RTE_DATA/env/fw_metagpt` (python 3.11 — metagpt requires `<3.12`) · **worker** `workers/metagpt_worker.py`
· **method** `fw_metagpt.py`

## The pin does not contain the primitive
SPEC §6A names `metagpt` 0.8.2 and the recipe `TeamLeader.publish_team_message(send_to)`. Those do not go
together: **`metagpt` 0.8.2 on PyPI has no `TeamLeader` class and no `publish_team_message` anywhere in the
package.** `metagpt/roles/di/` in 0.8.2 contains only `data_interpreter.py`. Routing in 0.8.2 is entirely
SOP-hardwired with no LLM in the loop: `Environment.publish_message` dispatches on `Message.send_to` and each
`Role._observe` filters by the actions it watches, so no agent self-description is ever read and there is
nothing to intercept that would answer "which agent does this framework pick".

`TeamLeader` (and `publish_team_message(content, send_to)`) exist only on the GitHub `main` branch, which
self-reports version 1.0.0. That is the version this rival is built against; see DEVIATIONS.md.

## What makes the pick (on the GitHub version)
`metagpt.roles.di.team_leader.TeamLeader`, a `RoleZero` subclass. The roster the model sees comes from
`TeamLeader._get_team_info`:

```python
for role in self.rc.env.roles.values():
    team_info += f"{role.name}: {role.profile}, {role.goal}\n"
```

so a candidate's name goes in `Role.name` and its self-description in `profile`/`goal`. `_think` injects that
roster via `TL_INSTRUCTION.format(team_info=…)`, and the pick is the `send_to` argument of the
`TeamLeader.publish_team_message` command the model emits (registered in `_update_tool_execution`).
Interception is a per-instance override of `publish_team_message` that raises with `send_to`, so no member
role ever runs.

## Caveats
- Not a general router: `TeamLeader` sits inside MetaGPT's software-company SOP, its prompt frames the team
  as a product/architect/engineer pipeline, and `RoleZero` wraps the decision in a multi-command planner.
  Any routing accuracy measured here is confounded by that SOP. Treat as appendix evidence only.
- Installation is not reproducible from the published metadata: `metagpt` 0.8.2 hard-pins `lancedb==0.4.0`,
  which has been yanked from PyPI, so `pip install metagpt==0.8.2` fails outright. Both the 0.8.2 and the
  GitHub install here are `--no-deps` plus a hand-assembled dependency set. See `requirements-frameworks/metagpt.txt`.
- MetaGPT loads a global config at import (`metagpt.config2.Config.default()`), so the venv carries a stub
  `metagpt_root/config/config2.yaml` and the worker sets `METAGPT_PROJECT_ROOT` before importing.

## Status: not implemented (`fw_metagpt.FwMetagpt.__init__` raises `NotImplementedError`)
The venv is built and `from metagpt.roles.di.team_leader import TeamLeader` succeeds, but the rival was
never validated end to end and so must not enter a grid. What remains is a testable interception: MetaGPT's
pick arrives as a `RoleZero` command list (```json `[{"command_name": "TeamLeader.publish_team_message",
"args": {"content": …, "send_to": <member name>}}]` ```), which `scripts/mock_openai_server.py` does not
produce, so there is no GPU-free way to check it today. The design, for whoever picks this up:

```python
os.environ["METAGPT_PROJECT_ROOT"] = f"{RTE_DATA}/env/fw_metagpt/metagpt_root"   # BEFORE importing metagpt
cfg = Config.from_llm_config({"api_type": "openai", "model": req["model"],
                              "base_url": req["base_url"], "api_key": req["api_key"], "temperature": 0})
ctx = Context(config=cfg)
tl  = TeamLeader(context=ctx)
env = Environment(context=ctx)
env.add_roles([tl] + [Role(name=c["name"], profile=c["description"], goal=c["description"], context=ctx)
                      for c in req["candidates"]])          # _get_team_info reads name / profile / goal
# the command dispatch table captures the BOUND method, so patch the map, not the attribute
raiser = lambda content, send_to: (_ for _ in ()).throw(Picked(send_to))
tl.tool_execution_map["TeamLeader.publish_team_message"] = raiser
tl.tool_execution_map["TeamLeader.publish_message"] = raiser
loop.run_until_complete(tl.run(Message(content=req["task"], send_to=tl.name)))
```

Two things to settle first: (1) a mock policy that emits a RoleZero command list, which is
MetaGPT-specific and therefore a poor fit for the shared generic mock — a live Qwen endpoint may be the
honest way to validate this one; (2) whether `RoleZero.max_react_loop = 3` lets a failed parse burn extra
supervisor calls, which would misprice the ledger.
