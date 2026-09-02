"""Google ADK rival (SPEC §6A row 7). See NOTES_google_adk.md."""
from ._common import FrameworkMethod


class FwGoogleAdk(FrameworkMethod):
    name = env = "fw_google_adk"
    worker = "google_adk_worker.py"
