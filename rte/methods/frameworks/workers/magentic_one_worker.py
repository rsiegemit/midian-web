"""Magentic-One, SPEC §6A recipe 3 second half: the `MagenticOneGroupChat` orchestrator writes a JSON
progress ledger and dispatches `next_speaker.answer`, reasoning over a `{team}` roster built from each
participant's `description`. It announces the pick through `_log_message("Next Speaker: ...")` just before
dispatching, which is the only place to stop it with the pick in hand. Since 2026-09-03 (`robust=True`,
the default) a ledger the orchestrator rejects is read by name mention instead of being retried up to
`max_json_retries` times: the first participant named after "next_speaker" in the raw reply (else the first
named anywhere) is the pick; a rejected ledger naming nobody, or a ledger that declares the request satisfied
without any speaker (the orchestrator answered the task itself), is a FAILURE. See NOTES_magentic_one.md."""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
from _bridge import serve_worker  # noqa: E402
from _wk import run_async  # noqa: E402
from autogen_worker import first_speaker, team_parts  # noqa: E402

from autogen_agentchat.teams import MagenticOneGroupChat  # noqa: E402
from autogen_agentchat.teams._group_chat._magentic_one._magentic_one_orchestrator import (  # noqa: E402
    MagenticOneOrchestrator)

MARK = "Next Speaker: "
RETRY = ("Failed to parse ledger information", "Invalid ledger format")
DONE = "Task completed, preparing final answer"    # ledger says is_request_satisfied: the orchestrator answered itself
NAME = re.compile(r"agent_\d{6}")
LAST = {"content": "", "robust": True}          # last raw model reply (the orchestrator logs only some of them)


class Picked(Exception):
    pass


_log_message = MagenticOneOrchestrator._log_message


def by_mention(text):
    """The participant named right after "next_speaker" in a rejected ledger, else the first one named."""
    tail = text[text.find("next_speaker"):] if "next_speaker" in text else text
    return next(iter(NAME.findall(tail) or NAME.findall(text)), None)


async def _stop_at_pick(self, content):
    if content.startswith(MARK):
        raise Picked(content[len(MARK):].strip())
    if LAST["robust"] and content.startswith(RETRY):
        raise Picked(by_mention(LAST["content"]) or "")
    if content.startswith(DONE):
        raise Picked("")
    await _log_message(self, content)


MagenticOneOrchestrator._log_message = _stop_at_pick


async def _select(req):
    # structured_output: the ledger is a nested pydantic model, and free-form JSON costs a model call per
    # mis-shaped reply (`max_json_retries`, default 10).
    mc, agents = team_parts(req, structured_output=True)
    LAST["robust"], orig = req.get("params", {}).get("robust", True), mc.create

    async def create(*a, **k):                      # remember the raw reply for by-mention extraction
        r = await orig(*a, **k); LAST["content"] = str(r.content); return r
    mc.create = create
    team = MagenticOneGroupChat(agents, model_client=mc, max_turns=1, emit_team_events=True)
    try:
        return await first_speaker(team, mc, req["task"])
    except Picked as p:
        return (p.args[0] or None), (MARK + p.args[0] if p.args[0] else "FAILURE: orchestrator answered itself or named nobody")


if __name__ == "__main__":
    serve_worker(lambda req: run_async(_select(req)))
