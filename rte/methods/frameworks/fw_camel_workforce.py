"""CAMEL Workforce rival. See docs/frameworks/NOTES_camel.md.

Ledger and params: as FrameworkMethod (rte.methods.frameworks._common)."""
from ._common import FrameworkMethod


class FwCamelWorkforce(FrameworkMethod):
    name, env, worker = "fw_camel_workforce", "fw_camel", "camel_worker.py"
