"""AgentScope rival (framework appendix): no selection primitive, so a DIY router.
See docs/frameworks/NOTES_agentscope.md.

Ledger and params: as FrameworkMethod (rte.methods.frameworks._common)."""
from ._common import FrameworkMethod


class FwAgentscope(FrameworkMethod):
    name, env, worker = "fw_agentscope", "fw_agentscope", "agentscope_worker.py"
