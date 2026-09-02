"""MetaGPT rival (SPEC §6A appendix) — NOT IMPLEMENTED. The pin and the recipe are incompatible:
metagpt 0.8.2 has no TeamLeader and no publish_team_message, and is not installable as published
(hard pin lancedb==0.4.0, yanked). See NOTES_metagpt.md for the design and DEVIATIONS.md."""
from ._common import FrameworkMethod


class FwMetagpt(FrameworkMethod):
    name, env, worker = "fw_metagpt", "fw_metagpt", "metagpt_worker.py"

    def __init__(self, **params):
        raise NotImplementedError(self.__doc__)
