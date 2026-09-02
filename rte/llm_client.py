"""The single place any LLM call goes through. OpenAI-compatible, temperature 0, disk-memoized.

Endpoints: one file per served model under `$RTE_DATA/endpoints.d/` (written lock-free by
scripts/_register_endpoint.py), merged into `$RTE_DATA/endpoints.json` for the contract.

NFS: `fcntl.flock` never returns on this cluster's netscratch mount (instant on $HOME and /tmp),
and the env's SQLite 3.50.3 picks a locking style that hits it, so the memo opens with `nolock=1`.
Sound here — one writer process, threads serialised by a per-database lock; `_claim` turns a second
live writer on the same host into an error instead of a corrupt cache. RTE_LLM_CACHE_NOLOCK=0
restores real locking. See DEVIATIONS.md 2026-09-02.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

RTE_DATA = Path(os.environ.get("RTE_DATA", "/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte"))
ENDPOINTS_PATH = Path(os.environ.get("RTE_ENDPOINTS", RTE_DATA / "endpoints.json"))
ENDPOINT_DIR = Path(os.environ.get("RTE_ENDPOINT_DIR", ENDPOINTS_PATH.parent / "endpoints.d"))
CACHE_DIR = Path(os.environ.get("RTE_LLM_CACHE", RTE_DATA / "cache"))
NOLOCK = os.environ.get("RTE_LLM_CACHE_NOLOCK", "1") == "1"
MAX_RETRIES = int(os.environ.get("RTE_LLM_RETRIES", "5"))

_STATS = {"hits": 0, "misses": 0, "generations": 0, "errors": 0, "retries": 0}
_LOCK = threading.Lock()
_dbs: dict[str, tuple] = {}
_clients: dict[str, object] = {}


class NoEndpointsError(RuntimeError):
    """No vLLM fleet configured — callers degrade with a clear message."""


def endpoints() -> dict[str, str]:
    """{model_id: base_url}. The per-model directory wins: it is what each server writes, so it is
    never a stale merge. Re-read every call — servers register as they warm up."""
    files = sorted(ENDPOINT_DIR.glob("*.json")) if ENDPOINT_DIR.is_dir() else []
    eps = {d["model"]: d["url"] for d in (json.loads(p.read_text()) for p in files)}
    if not eps and ENDPOINTS_PATH.exists():
        eps = json.loads(ENDPOINTS_PATH.read_text())
    if not eps and not (ENDPOINT_DIR.is_dir() or ENDPOINTS_PATH.exists()):
        raise NoEndpointsError(f"no vLLM endpoints at {ENDPOINT_DIR} or {ENDPOINTS_PATH}; launch "
                               f"scripts/serve_fleet.sbatch or set RTE_ENDPOINTS")
    return eps


def _claim(path: Path) -> None:
    """With nolock=1 SQLite cannot see a second writer, so make the common case fail loudly."""
    stamp, pid, host = path.with_suffix(".owner"), os.getpid(), socket.gethostname()
    prev = json.loads(stamp.read_text()) if stamp.exists() else {}
    if prev.get("host") == host and prev.get("pid") not in (None, pid):
        try:
            os.kill(prev["pid"], 0)
        except OSError:
            pass                                    # stale stamp: the owner is gone, take over
        else:
            raise RuntimeError(f"{path.name} is already open by pid {prev['pid']} on {host}; the "
                               f"memo runs without SQLite locking here, so two live writers would "
                               f"corrupt it.")
    stamp.write_text(json.dumps({"host": host, "pid": pid, "t": time.time()}))


def _db(model: str) -> tuple:
    key = model.replace("/", "__")
    with _LOCK:
        if key not in _dbs:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path = CACHE_DIR / f"{key}.sqlite"
            if NOLOCK:
                _claim(path)
            con = sqlite3.connect(f"file:{path}?nolock=1" if NOLOCK else str(path), uri=NOLOCK,
                                  check_same_thread=False, timeout=60.0)
            con.execute("PRAGMA journal_mode=TRUNCATE")
            con.execute("CREATE TABLE IF NOT EXISTS memo (k TEXT PRIMARY KEY, v TEXT)")
            con.commit()
            _dbs[key] = (con, threading.Lock())
        return _dbs[key]


def _bump(field: str, k: int = 1) -> None:
    with _LOCK:
        _STATS[field] += k


def content_key(model: str, messages: Sequence[dict], max_tokens: int,
                prefix: str | None = None) -> str:
    """The memo key: blake2b over the FULL request. Never a signature of the inputs that built the
    prompt -- reword a prompt while keeping such a key and the cache serves an answer that prompt
    never produced (it happened three times: the self-description, the self-rating and the tool
    follow-up). Hashing the content makes any wording change miss the cache automatically.
    `prefix` is decoration for readability only; it is never a substitute for the hash."""
    blob = json.dumps([model, list(messages), int(max_tokens)], sort_keys=True)
    h = hashlib.blake2b(blob.encode(), digest_size=16).hexdigest()
    return f"{prefix}:{h}" if prefix else h


_NO_SYSTEM: frozenset[str] | None = None


def _no_system() -> frozenset[str]:
    """Models whose chat template REJECTS a system message — Gemma-2 answers HTTP 400
    'System role not supported'. Declared by `system_role: false` in configs/models.yaml."""
    global _NO_SYSTEM
    if _NO_SYSTEM is None:
        from .backends.population import ladder
        _NO_SYSTEM = frozenset(m["id"] for m in ladder()["models"] if not m.get("system_role", True))
    return _NO_SYSTEM


def for_model(model: str, messages: Sequence[dict]) -> list[dict]:
    """Fold the system turn into the first user turn for models that reject a system role. The
    memo key is computed from the ORIGINAL messages: it is the same logical request either way."""
    msgs = list(messages)
    if model not in _no_system() or not msgs or msgs[0]["role"] != "system":
        return msgs
    head, rest = msgs[0], msgs[1:]
    if rest and rest[0]["role"] == "user":
        return [{"role": "user", "content": f"{head['content']}\n\n{rest[0]['content']}"}] + rest[1:]
    return [{"role": "user", "content": head["content"]}] + rest


def _generate(model: str, messages: Sequence[dict], max_tokens: int) -> str:
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            if model not in _clients:
                eps = endpoints()
                if model not in eps:
                    raise NoEndpointsError(f"model {model!r} not served; have {sorted(eps)}")
                from openai import OpenAI
                _clients[model] = OpenAI(base_url=eps[model], api_key="EMPTY", timeout=600.0,
                                         max_retries=0)
            r = _clients[model].chat.completions.create(
                model=model, messages=for_model(model, messages), max_tokens=int(max_tokens),
                temperature=0.0, seed=0)
            _bump("generations")
            return r.choices[0].message.content or ""
        except Exception as e:                                    # noqa: BLE001
            last = e
            _bump("retries")
            _clients.pop(model, None)                             # re-read endpoints on reconnect
            time.sleep(min(30.0, 1.5 * 2 ** attempt))
    _bump("errors")
    raise RuntimeError(f"generation failed for {model} after {MAX_RETRIES} attempts: {last}")


def complete(model: str, messages: Sequence[dict], max_tokens: int = 512,
             cache_key: str | None = None) -> str:
    """One deterministic completion, disk-memoized. `cache_key` is an optional readability PREFIX
    on the content hash, never a replacement for it -- see `content_key`."""
    return complete_batch(model, [messages], [cache_key] if cache_key else None, max_tokens, 1)[0]


def complete_batch(model: str, batch: Sequence[Sequence[dict]], keys: Sequence[str] | None = None,
                   max_tokens: int = 512, concurrency: int = 64) -> list[str]:
    """Fan out over a thread pool — vLLM batches server-side. Results follow `batch` order.

    Deduplicates WITHIN the batch as well as against the memo: agents that share a prompt
    signature emit a byte-identical prompt, so a measurement sweep hands us the same request
    hundreds of times in one call, and without this they would all miss in parallel and regenerate
    the identical prompt once per agent. `keys` are readability PREFIXES only -- the memo key is
    always the hash of the full request."""
    batch = list(batch)
    if keys is not None and len(keys) != len(batch):
        raise ValueError(f"keys/batch length mismatch: {len(keys)} vs {len(batch)}")
    prefixes = list(keys) if keys is not None else [None] * len(batch)
    keys = [content_key(model, m, max_tokens, pre) for m, pre in zip(batch, prefixes)]
    con, lock = _db(model)

    answer: dict[str, str] = {}
    todo: dict[str, int] = {}                     # unique key -> a representative index
    with lock:
        for i, k in enumerate(keys):
            if k in answer or k in todo:
                continue
            row = con.execute("SELECT v FROM memo WHERE k=?", (k,)).fetchone()
            (answer if row else todo).__setitem__(k, row[0] if row else i)
    _bump("hits", sum(1 for k in keys if k not in todo))
    _bump("misses", len(todo))                    # misses == unique generations, not positions

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(todo)))) as ex:
            texts = list(ex.map(lambda i: _generate(model, batch[i], max_tokens), todo.values()))
        answer.update(zip(todo, texts))
        with lock:
            con.executemany("INSERT OR REPLACE INTO memo VALUES (?, ?)", list(zip(todo, texts)))
            con.commit()
    return [answer[k] for k in keys]


def stats() -> dict:
    with _LOCK:
        s = dict(_STATS)
    tot = s["hits"] + s["misses"]
    return s | {"cache_hit_rate": s["hits"] / tot if tot else 0.0}


def reset_stats() -> None:
    with _LOCK:
        _STATS.update(dict.fromkeys(_STATS, 0))
