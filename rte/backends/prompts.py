"""Turning (family, question, handicap, tool) into messages, and a reply back into an answer.

HANDICAP: no exemplar, no family description, no tool. The spec's "difficulty capped" cannot be
applied per agent — every agent must see the SAME instance for the paired stream and the shared
verifier to mean anything — and the generation-budget cap that once stood in for it measured
NON-MONOTONE (it HELPED a 0.5B on syllogism, 0.85 vs 0.30), so it is off by default.
See DEVIATIONS.md 2026-09-02 and scripts/calibrate_families.py.
"""
from __future__ import annotations

import re

from . import families, tools

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
TOOL_RE = {"calculator": re.compile(r"<calc>(.*?)</calc>", re.DOTALL | re.IGNORECASE),
           "python": re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)}
FORMAT = "Give your final answer inside <answer></answer> tags, with nothing else inside the tags."


def build(family: str, question: str, handicapped: bool, tool: str) -> list[dict]:
    """A handicapped agent keeps the answer-format instruction: withholding that would measure
    format compliance rather than skill, and all agents are scored by the same verifier."""
    if handicapped:
        parts = ["Solve the following problem."]
    else:
        q, a = families.exemplar(family)
        parts = [f"You are solving a '{family}' task.", families.describe(family),
                 f"Worked example.\nQuestion: {q}\nAnswer: {a}"]
    parts += [tools.HINT[tool]] if tool in tools.HINT else []
    return [{"role": "system", "content": "\n\n".join(parts + [FORMAT])},
            {"role": "user", "content": question}]


def follow_up(messages: list[dict], reply: str, result: str) -> list[dict]:
    return messages + [{"role": "assistant", "content": reply},
                       {"role": "user", "content": f"Tool result:\n{result}\n\nNow give the final "
                                                   f"answer. {FORMAT}"}]


def extract_answer(text: str) -> str:
    if m := ANSWER_RE.findall(text or ""):
        return m[-1].strip()
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def find_tool_call(text: str, tool: str) -> str | None:
    """The tool block the model emitted — but not once it has already answered."""
    if not text or tool not in TOOL_RE or ANSWER_RE.search(text):
        return None
    m = TOOL_RE[tool].search(text)
    return m.group(1) if m else None


def parse_rating(text: str) -> float:
    """A self-declared competence in [0,1]; 0.5 when the model says nothing parseable."""
    hay = (ANSWER_RE.findall(text or "") or [text or ""])[-1]
    if m := re.search(r"(\d{1,3})\s*%", hay):
        return min(1.0, float(m.group(1)) / 100)
    if m := re.search(r"(?<![\d.])(0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])", hay):
        return float(m.group(1))
    return 0.5


def rate_self(family: str, tool: str, example: str) -> list[dict]:
    return [{"role": "system", "content":
             f"Rate your own competence at a class of problems. Your tool: {tool}. Answer with a "
             "single number between 0 and 1 (0 = never solve one, 1 = always) inside "
             "<answer></answer> tags."},
            {"role": "user", "content": f"Task family: {family}.\n{families.describe(family)}\n\n"
                                        f"Example problem:\n{example}\n\nWhat fraction of problems "
                                        "like this do you solve correctly?"}]


def describe_self(model: str, tool: str, specialty: str) -> list[dict]:
    return [{"role": "system", "content":
             "Write one short paragraph (<=70 words) describing yourself as a service in an agent "
             "marketplace: what problems you handle well and what tools you have. No lists, no "
             "headings."},
            {"role": "user", "content":
             f"Your base model is {model}. Your tool is: {tool}. You are equipped for these task "
             f"families: {specialty or 'no particular area'}. Others you handle without examples "
             "or tools."}]
