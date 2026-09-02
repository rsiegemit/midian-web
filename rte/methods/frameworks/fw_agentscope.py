"""AgentScope rival (SPEC §6A appendix). No selection primitive: DIY router. See NOTES_agentscope.md."""
from ._common import FrameworkMethod


class FwAgentscope(FrameworkMethod):
    name, env, worker = "fw_agentscope", "fw_agentscope", "agentscope_worker.py"
