"""Magentic-One rival worker (SPEC §6A recipe 3, second half). Selection primitive: the
`MagenticOneGroupChat` orchestrator, which writes a JSON progress ledger and dispatches
`progress_ledger["next_speaker"]["answer"]`. The roster it reasons over is `{team}`, built from each
participant's `description`, so that is where the self-description goes.

The orchestrator announces the pick through `_log_message("Next Speaker: ...")` just before it publishes
the work request, so patching that call is the only place to stop the run with the pick in hand and
nothing dispatched. Falls back to the `SelectSpeakerEvent`. Shares the AutoGen venv. See
NOTES_magentic_one.md.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bridge import serve_worker  # noqa: E402
from autogen_worker import LOOP, MODEL_INFO, agents_for, client, first_speaker  # noqa: E402

from autogen_agentchat.teams import MagenticOneGroupChat  # noqa: E402
from autogen_agentchat.teams._group_chat._magentic_one._magentic_one_orchestrator import (  # noqa: E402
    MagenticOneOrchestrator)

# The ledger is a nested pydantic model (`LedgerEntry`); ask for structured output so a mis-shaped reply
# does not burn `max_json_retries` model calls per selection.
LEDGER_INFO = dict(MODEL_INFO, structured_output=True)
MARK = "Next Speaker: "


class Picked(Exception):
    def __init__(self, name):
        self.name = name


_log_message = MagenticOneOrchestrator._log_message


async def _stop_at_pick(self, content):
    if content.startswith(MARK):
        raise Picked(content[len(MARK):].strip())
    await _log_message(self, content)


MagenticOneOrchestrator._log_message = _stop_at_pick


async def _select(req):
    cands = req["candidates"]
    if len(cands) == 1:
        return cands[0]["name"], "single candidate"
    mc = client(req, LEDGER_INFO)
    team = MagenticOneGroupChat(agents_for(cands, mc), model_client=mc, max_turns=1, emit_team_events=True)
    try:
        return await first_speaker(team, req["task"])
    except Picked as p:
        return p.name, (MARK + p.name)[:500]
    finally:
        await mc.close()


if __name__ == "__main__":
    serve_worker(lambda req: LOOP.run_until_complete(_select(req)))
