"""The agent tool: `python` (or none).

The `python` tool is containment, not a security boundary: an isolated interpreter, 5 s, CPU and
memory rlimits, empty environment, sockets disabled. It stops runaway loops and stray imports; it
is not hardened against a determined escape (the author is a 0.5-14B model doing arithmetic).
"""
from __future__ import annotations

import hashlib
import subprocess
import sys

TIMEOUT = 5.0                                       # seconds; named in HINT and in the timeout error

# Byte-identical to commit 20304d0: this text is PREPENDED to model-written code, so changing it
# shifts every traceback line number the sandbox feeds back to the model.
_PY_PREAMBLE = (
    "import socket as _s\n"
    "def _blocked(*a, **k):\n"
    "    raise OSError('network disabled in the RTE python tool')\n"
    "_s.socket = _blocked; _s.create_connection = _blocked; _s.socket_socketpair = _blocked\n"
)


def python(code: str, memo: bool = True) -> str:
    if len(code) > 8000:
        return "ERROR: code too long"                # no exception-type prefix (commit 20304d0)
    if not memo:      # de-cloned populations: sampled code rarely repeats (midian.backends.llm)
        return _python(code)
    # runs are deterministic (PYTHONHASHSEED=0): memoise like a generation
    from ..llm_client import memo_call
    key = "tool:python:" + hashlib.blake2b(code.encode(), digest_size=16).hexdigest()
    return memo_call(key, lambda: _python(code))


def _python(code: str) -> str:
    def limits():
        import resource
        for res, lim in ((resource.RLIMIT_CPU, int(TIMEOUT) + 1), (resource.RLIMIT_AS, 2 << 30),
                         (resource.RLIMIT_NPROC, 64), (resource.RLIMIT_FSIZE, 1 << 20)):
            resource.setrlimit(res, (lim, lim))

    try:
        r = subprocess.run([sys.executable, "-I", "-c", _PY_PREAMBLE + code],
                           capture_output=True, text=True, timeout=TIMEOUT, cwd="/tmp",
                           env={"PATH": "/usr/bin:/bin", "HOME": "/tmp", "PYTHONHASHSEED": "0"},
                           preexec_fn=limits)
    except subprocess.TimeoutExpired:
        return "ERROR: timed out after 5s"
    except Exception as e:                                       # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {e}"
    out = (r.stdout or "") + (("\nSTDERR: " + r.stderr) if r.returncode else "")
    return out.strip()[:2000] or "(no output)"


# What the system prompt tells an agent that has the tool (midian.backends.prompts.build).
HINT = {
    "python": "You have a Python sandbox (5 s, no network). To use it, emit exactly one "
              "```python ... ``` block that prints what you need, and stop; its stdout will be "
              "given to you.",
}
