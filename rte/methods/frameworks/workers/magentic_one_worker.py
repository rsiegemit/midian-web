"""Magentic-One, SPEC §6A recipe 3 second half: the `MagenticOneGroupChat` orchestrator writes a JSON
progress ledger and dispatches `next_speaker.answer`, reasoning over a `{team}` roster built from each
participant's `description`. It announces the pick through `_log_message("Next Speaker: ...")` just before
dispatching, which is the only place to stop it with the pick in hand. See NOTES_magentic_one.md."""
import os
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


class Picked(Exception):
    pass


_log_message = MagenticOneOrchestrator._log_message


async def _stop_at_pick(self, content):
    if content.startswith(MARK):
        raise Picked(content[len(MARK):].strip())
    await _log_message(self, content)


MagenticOneOrchestrator._log_message = _stop_at_pick


async def _select(req):
    # structured_output: the ledger is a nested pydantic model, and free-form JSON costs a model call per
    # mis-shaped reply (`max_json_retries`, default 10).
    mc, agents = team_parts(req, structured_output=True)
    team = MagenticOneGroupChat(agents, model_client=mc, max_turns=1, emit_team_events=True)
    try:
        return await first_speaker(team, mc, req["task"])
    except Picked as p:
        return p.args[0], MARK + p.args[0]


if __name__ == "__main__":
    serve_worker(lambda req: run_async(_select(req)))
