"""OpenAI Agents SDK rival. See docs/frameworks/NOTES_openai_agents.md.

Ledger and params: as FrameworkMethod (rte.methods.frameworks._common)."""
from ._common import FrameworkMethod


class FwOpenAIAgents(FrameworkMethod):
    name = env = "fw_openai_agents"
    worker = "openai_agents_worker.py"
