"""AutoGen rival (SPEC §6A row 3). See NOTES_autogen.md."""
from ._common import FrameworkMethod


class FwAutogen(FrameworkMethod):
    name = env = "fw_autogen"
    worker = "autogen_worker.py"
