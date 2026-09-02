"""Agent tools (SPEC §1: tools in {calculator, python, none}). Each is `run(payload) -> str`.

The `python` tool is containment, not a security boundary: an isolated interpreter, 5 s, CPU and
memory rlimits, empty environment, sockets disabled. It stops runaway loops and stray imports; it
is not hardened against a determined escape (the author is a 0.5-14B model doing arithmetic).
"""
from __future__ import annotations

import ast
import operator as op
import subprocess
import sys

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow,
        ast.USub: op.neg, ast.UAdd: op.pos}

# Byte-identical to commit 20304d0: this text is PREPENDED to model-written code, so changing it
# shifts every traceback line number the sandbox feeds back to the model.
_PY_PREAMBLE = (
    "import socket as _s\n"
    "def _blocked(*a, **k):\n"
    "    raise OSError('network disabled in the RTE python tool')\n"
    "_s.socket = _blocked; _s.create_connection = _blocked; _s.socket_socketpair = _blocked\n"
)


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 64:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError(f"unsupported expression element {type(node).__name__}")


def calculator(expr: str) -> str:
    """Literal-only arithmetic on a model-emitted expression. Nothing but numbers and operators."""
    expr = expr.strip().replace("^", "**").replace(",", "")
    if len(expr) > 400:
        return "ERROR: expression too long"          # no exception-type prefix (commit 20304d0)
    try:
        return str(_eval(ast.parse(expr, mode="eval")))
    except Exception as e:                                       # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {e}"


def python(code: str, timeout: float = 5.0) -> str:
    def limits():
        import resource
        for res, lim in ((resource.RLIMIT_CPU, int(timeout) + 1), (resource.RLIMIT_AS, 2 << 30),
                         (resource.RLIMIT_NPROC, 64), (resource.RLIMIT_FSIZE, 1 << 20)):
            resource.setrlimit(res, (lim, lim))

    if len(code) > 8000:
        return "ERROR: code too long"                # no exception-type prefix (commit 20304d0)
    try:
        r = subprocess.run([sys.executable, "-I", "-c", _PY_PREAMBLE + code],
                           capture_output=True, text=True, timeout=timeout, cwd="/tmp",
                           env={"PATH": "/usr/bin:/bin", "HOME": "/tmp", "PYTHONHASHSEED": "0"},
                           preexec_fn=limits)
    except subprocess.TimeoutExpired:
        return "ERROR: timed out after 5s"
    except Exception as e:                                       # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {e}"
    out = (r.stdout or "") + (("\nSTDERR: " + r.stderr) if r.returncode else "")
    return out.strip()[:2000] or "(no output)"


RUN = {"calculator": calculator, "python": python}
NAMES = ("calculator", "python", "none")

# What the system prompt tells an agent that has the tool. Keyed the same way, so adding a tool
# is one entry in RUN, one in HINT, one pattern in rte.backends.llm.TOOL_RE.
HINT = {
    "calculator": "You have a calculator. To use it, emit exactly one line "
                  "<calc>ARITHMETIC EXPRESSION</calc> and stop; the result will be given to you.",
    "python": "You have a Python sandbox (5 s, no network). To use it, emit exactly one "
              "```python ... ``` block that prints what you need, and stop; its stdout will be "
              "given to you.",
}
