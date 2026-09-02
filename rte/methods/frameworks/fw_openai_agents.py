"""OpenAI Agents SDK rival (SPEC §6A row 6). See NOTES_openai_agents.md."""
from ._common import FrameworkMethod


class FwOpenAIAgents(FrameworkMethod):
    name = env = "fw_openai_agents"
    worker = "openai_agents_worker.py"
