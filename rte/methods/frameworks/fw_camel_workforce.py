"""CAMEL Workforce rival (SPEC §6A row 10, recipe 9). See NOTES_camel.md."""
from ._common import FrameworkMethod


class FwCamelWorkforce(FrameworkMethod):
    name, env, worker = "fw_camel_workforce", "fw_camel", "camel_worker.py"
