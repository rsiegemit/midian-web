"""AutoGen rival (SPEC §6A row 3): `SelectorGroupChat` asks the model to pick the next speaker from a
roster of `name: description` lines; the pick surfaces as a `SelectSpeakerEvent`. See NOTES_autogen.md."""
from ._common import FrameworkMethod


class FwAutogen(FrameworkMethod):
    name = "fw_autogen"
    env = "fw_autogen"
    worker = "autogen_worker.py"
