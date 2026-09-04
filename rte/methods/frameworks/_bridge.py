"""JSON-lines subprocess bridge to a framework worker running in its own venv.

Protocol (one JSON object per line, both directions):
  request : {"id": int, "task": str, "candidates": [{"name": str, "description": str}],
             "model": str, "base_url": str, "api_key": str, "params": {method params, e.g. "mode"}}
  response: {"id": int, "choice": str|null, "error": str|null, "raw": any}
`choice` must be one of the candidate names (the worker maps its framework's pick back to a name).
Workers live in rte/methods/frameworks/workers/<fw>_worker.py and must run under the venv's python.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading

RTE_DATA = os.environ.get("RTE_DATA", "/scratch/rte")
WORKERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workers")


def venv_python(env_name: str) -> str:
    p = os.path.join(RTE_DATA, "env", env_name, "bin", "python")
    if not os.path.exists(p):
        raise RuntimeError(f"framework venv missing: {p} (run scripts/04_build_fw_envs.sh {env_name})")
    return p


class Bridge:
    def __init__(self, env_name: str, worker: str, timeout: float = 120.0):
        self.env_name, self.worker, self.timeout = env_name, worker, timeout
        self._proc = None
        self._lock = threading.Lock()
        self._next = 0
        self.stats = {"calls": 0, "errors": 0, "restarts": 0}

    def _start(self):
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=WORKERS)
        for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
            env.pop(k, None)
        env.setdefault("OPENAI_API_KEY", "EMPTY")
        env["PYTHONNOUSERSITE"] = "1"                  # conda prefixes see ~/.local site-packages; never let it shadow a venv
        self._proc = subprocess.Popen([venv_python(self.env_name), os.path.join(WORKERS, self.worker)],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
                                      text=True, bufsize=1, env=env)

    def select(self, task: str, candidates: list[dict], model: str, base_url: str, api_key: str = "EMPTY", params: dict | None = None) -> dict:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                if self._proc is not None:
                    self.stats["restarts"] += 1
                self._start()
            self._next += 1
            req = {"id": self._next, "task": task, "candidates": candidates, "model": model,
                   "base_url": base_url, "api_key": api_key, "params": params or {}}
            self.stats["calls"] += 1
            try:
                self._proc.stdin.write(json.dumps(req) + "\n"); self._proc.stdin.flush()
                line = _readline_timeout(self._proc, self.timeout)
                resp = json.loads(line)
            except Exception as e:                       # worker died / timed out: restart next call
                self.stats["errors"] += 1
                try: self._proc.kill()
                except Exception: pass
                self._proc = None
                return {"id": self._next, "choice": None, "error": f"{type(e).__name__}: {e}", "raw": None}
            if resp.get("error"):
                self.stats["errors"] += 1
            return resp

    def close(self):
        if self._proc is not None:
            try: self._proc.stdin.close(); self._proc.wait(timeout=5)
            except Exception: self._proc.kill()
            self._proc = None


def _readline_timeout(proc, timeout):
    """Next JSON line from the worker; frameworks that print to stdout (Rich panels, warnings) are skipped."""
    out = {}
    def rd():
        while True:
            line = proc.stdout.readline()
            if not line or line.lstrip().startswith("{"):
                out["line"] = line; return
    t = threading.Thread(target=rd, daemon=True); t.start(); t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"worker did not answer within {timeout}s")
    if not out.get("line"):
        raise RuntimeError("worker closed stdout")
    return out["line"]


def serve_worker(select_fn):
    """Run inside the venv: `select_fn(req: dict) -> str|None` picks a candidate name. Loop forever on stdin."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        resp = {"id": req.get("id"), "choice": None, "error": None, "raw": None}
        try:
            out = select_fn(req)
            if isinstance(out, tuple):
                resp["choice"], resp["raw"] = out
            else:
                resp["choice"] = out
        except Exception as e:                           # never die on one bad request
            resp["error"] = f"{type(e).__name__}: {e}"
        sys.stdout.write(json.dumps(resp, default=str) + "\n"); sys.stdout.flush()
