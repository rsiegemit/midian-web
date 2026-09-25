"""Google ADK rival. See docs/frameworks/NOTES_google_adk.md.

Ledger and params: as FrameworkMethod (midian.methods.frameworks._common)."""
from ._common import FrameworkMethod


class FwGoogleAdk(FrameworkMethod):
    name = env = "fw_google_adk"
    worker = "google_adk_worker.py"
