"""OpenAI-compatible client for the vLLM fleet, with a disk memo.

Endpoints live in `$RTE_DATA/endpoints.json`:

    {"Qwen/Qwen2.5-7B-Instruct": "http://holygpu8a11:8003/v1", ...}

written by `scripts/serve_fleet.sbatch` as each server passes /health.

Everything is temperature 0 / seed 0, so a completion is a pure function of
(model, messages, max_tokens).  That is what makes the live grid affordable:
the SAME (agent, instance) pair fetched by twenty different methods costs one
generation, once, for the whole grid (SPEC §1 "Memoization").  The memo is one
sqlite file per model under `$RTE_DATA/cache/`.

NFS note.  These files sit on netscratch, where file locking is broken: `fcntl.flock` never
returns (verified 2026-09-02 -- instant on $HOME and /tmp, hangs forever under $RTE_DATA), and
the env's SQLite 3.50.3 picks a locking style that hits it, so a plain `sqlite3.connect` there
HANGS on the first write.  (Base Python's SQLite 3.51.0 does not -- it is version-dependent.)

So the memo opens its databases with `nolock=1`, which tells SQLite to do no file locking at all.
That is sound here because the memo has exactly one writer process -- the runner -- whose threads
are already serialised by a per-database `threading.Lock`.  It is NOT safe to point two concurrent
processes at the same cache file; `owner_stamp` catches that on the same host and turns it into a
clear error instead of a corrupted database.  Set RTE_LLM_CACHE_NOLOCK=0 to restore real locking
on a filesystem where it works.
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
# Authoritative source: one file per served model, written lock-free by scripts/_register_endpoint.py.
# `flock` blocks forever on this cluster's /n/netscratch mount (verified 2026-09-02), so the fleet
# cannot guard a single shared file; see that script's docstring.
ENDPOINT_DIR = Path(os.environ.get("RTE_ENDPOINT_DIR", ENDPOINTS_PATH.parent / "endpoints.d"))
CACHE_DIR = Path(os.environ.get("RTE_LLM_CACHE", RTE_DATA / "cache"))

DEFAULT_CONCURRENCY = int(os.environ.get("RTE_LLM_CONCURRENCY", "64"))
MAX_RETRIES = int(os.environ.get("RTE_LLM_RETRIES", "5"))
# See the module docstring: SQLite locking hangs on this cluster's netscratch mount.
NOLOCK = os.environ.get("RTE_LLM_CACHE_NOLOCK", "1") == "1"

_STATS = {"hits": 0, "misses": 0, "generations": 0, "errors": 0, "retries": 0}
_STATS_LOCK = threading.Lock()


class NoEndpointsError(RuntimeError):
    """Raised when no vLLM fleet is configured -- callers degrade with a clear message."""


# --------------------------------------------------------------------------- endpoints
_endpoints_cache: dict | None = None
_endpoints_mtime: tuple | None = None


def _read_endpoint_dir() -> dict[str, str]:
    out: dict[str, str] = {}
    if not ENDPOINT_DIR.is_dir():
        return out
    for p in sorted(ENDPOINT_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("model") and d.get("url"):
            out[d["model"]] = d["url"]
    return out


def endpoints(refresh: bool = False) -> dict[str, str]:
    """{model_id: base_url}. Re-read when anything changes -- servers register as they warm up.

    The per-model directory wins over the merged endpoints.json: it is what each server writes
    directly, so it is never a stale merge."""
    global _endpoints_cache, _endpoints_mtime
    stamp = []
    if ENDPOINT_DIR.is_dir():
        stamp.append(ENDPOINT_DIR.stat().st_mtime)
    if ENDPOINTS_PATH.exists():
        stamp.append(ENDPOINTS_PATH.stat().st_mtime)
    if not stamp:
        raise NoEndpointsError(
            f"no vLLM endpoints at {ENDPOINT_DIR} or {ENDPOINTS_PATH}; launch "
            f"scripts/serve_fleet.sbatch (or serve_smoke.sbatch) first, or set RTE_ENDPOINTS")
    key = tuple(stamp)
    if refresh or _endpoints_cache is None or key != _endpoints_mtime:
        eps = _read_endpoint_dir()
        if not eps and ENDPOINTS_PATH.exists():
            try:
                eps = json.loads(ENDPOINTS_PATH.read_text())
            except json.JSONDecodeError:
                eps = {}
        _endpoints_cache = eps
        _endpoints_mtime = key
    return dict(_endpoints_cache)


def have_endpoints() -> bool:
    try:
        return bool(endpoints())
    except Exception:                                             # noqa: BLE001
        return False


# --------------------------------------------------------------------------- disk memo
_dbs: dict[str, sqlite3.Connection] = {}
_db_locks: dict[str, threading.Lock] = {}
_dbs_guard = threading.Lock()


def _slug(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def _claim(path: Path) -> None:
    """Record that this process owns `path`. With nolock=1 SQLite will not notice a second writer,
    so make the common case (two runs on one host) fail loudly rather than corrupt the memo."""
    stamp = path.with_suffix(".owner")
    host, pid = socket.gethostname(), os.getpid()
    try:
        prev = json.loads(stamp.read_text())
    except (OSError, json.JSONDecodeError):
        prev = None
    if prev and prev.get("host") == host and prev.get("pid") not in (None, pid):
        try:
            os.kill(int(prev["pid"]), 0)
        except (OSError, ValueError, TypeError):
            pass                                   # stale stamp: the owner is gone, take over
        else:
            raise RuntimeError(
                f"{path.name} is already open by pid {prev['pid']} on {host}. The memo runs "
                f"without SQLite locking on this filesystem (see rte.llm_client docstring), so two "
                f"live writers would corrupt it. Stop the other run, or point RTE_LLM_CACHE "
                f"somewhere else.")
    try:
        stamp.write_text(json.dumps({"host": host, "pid": pid, "t": time.time()}))
    except OSError:
        pass                                       # a read-only cache dir is not a reason to die


def _db(model: str) -> tuple[sqlite3.Connection, threading.Lock]:
    key = _slug(model)
    with _dbs_guard:
        if key not in _dbs:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path = CACHE_DIR / f"{key}.sqlite"
            if NOLOCK:
                _claim(path)
                con = sqlite3.connect(f"file:{path}?nolock=1", uri=True,
                                      check_same_thread=False, timeout=60.0)
            else:
                con = sqlite3.connect(path, check_same_thread=False, timeout=60.0)
            con.execute("PRAGMA journal_mode=TRUNCATE")
            con.execute("PRAGMA synchronous=NORMAL")
            con.execute("CREATE TABLE IF NOT EXISTS memo (k TEXT PRIMARY KEY, v TEXT)")
            con.commit()
            _dbs[key] = con
            _db_locks[key] = threading.Lock()
        return _dbs[key], _db_locks[key]


def content_key(model: str, messages: Sequence[dict], max_tokens: int) -> str:
    """Deterministic key for a completion request. blake2b, the repo's stable-digest primitive."""
    blob = json.dumps({"m": model, "msgs": list(messages), "mt": int(max_tokens)},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()


def _memo_get(model: str, key: str) -> str | None:
    con, lock = _db(model)
    with lock:
        row = con.execute("SELECT v FROM memo WHERE k=?", (key,)).fetchone()
    return None if row is None else row[0]


def _memo_put(model: str, key: str, value: str) -> None:
    con, lock = _db(model)
    with lock:
        con.execute("INSERT OR REPLACE INTO memo (k, v) VALUES (?, ?)", (key, value))
        con.commit()


# --------------------------------------------------------------------------- generation
_clients: dict[str, object] = {}


def _client(model: str):
    if model not in _clients:
        eps = endpoints()
        if model not in eps:
            raise NoEndpointsError(f"model {model!r} not served; endpoints.json has {sorted(eps)}")
        from openai import OpenAI
        _clients[model] = OpenAI(base_url=eps[model], api_key="EMPTY", timeout=600.0, max_retries=0)
    return _clients[model]


def _bump(field: str, k: int = 1) -> None:
    with _STATS_LOCK:
        _STATS[field] += k


def _generate(model: str, messages: Sequence[dict], max_tokens: int) -> str:
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            r = _client(model).chat.completions.create(
                model=model, messages=list(messages), max_tokens=int(max_tokens),
                temperature=0.0, seed=0)
            _bump("generations")
            return r.choices[0].message.content or ""
        except Exception as e:                                    # noqa: BLE001
            last = e
            _bump("retries")
            _clients.pop(model, None)                             # force endpoint re-read on reconnect
            time.sleep(min(30.0, 1.5 * 2 ** attempt))
    _bump("errors")
    raise RuntimeError(f"generation failed for {model} after {MAX_RETRIES} attempts: {last}") from last


def complete(model: str, messages: Sequence[dict], max_tokens: int = 512,
             cache_key: str | None = None) -> str:
    """One deterministic completion, disk-memoized. `cache_key` overrides the content hash."""
    key = cache_key or content_key(model, messages, max_tokens)
    hit = _memo_get(model, key)
    if hit is not None:
        _bump("hits")
        return hit
    _bump("misses")
    out = _generate(model, messages, max_tokens)
    _memo_put(model, key, out)
    return out


def complete_batch(model: str, batch: Sequence[Sequence[dict]], keys: Sequence[str] | None = None,
                   max_tokens: int = 512, concurrency: int = DEFAULT_CONCURRENCY) -> list[str]:
    """Fan out over a thread pool -- vLLM batches server-side. Order of results matches `batch`."""
    batch = list(batch)
    if keys is None:
        keys = [content_key(model, m, max_tokens) for m in batch]
    keys = list(keys)
    if len(keys) != len(batch):
        raise ValueError(f"keys/batch length mismatch: {len(keys)} vs {len(batch)}")

    out: list[str | None] = [None] * len(batch)
    # Dedupe WITHIN the batch as well as against the disk memo. The llm backend deliberately keys
    # on the prompt signature, so a measurement sweep hands us the same key hundreds of times in
    # one call; without this the whole batch would miss in parallel and generate each prompt once
    # per agent, which is exactly the cost the memo exists to avoid.
    todo: dict[str, int] = {}                    # unique key -> a representative index
    for i, k in enumerate(keys):
        if k in todo:
            continue
        hit = _memo_get(model, k)
        if hit is not None:
            out[i] = hit
        else:
            todo[k] = i
    if todo:
        def work(i: int) -> tuple[int, str]:
            return i, _generate(model, batch[i], max_tokens)
        with ThreadPoolExecutor(max_workers=max(1, min(int(concurrency), len(todo)))) as ex:
            for i, text in ex.map(work, list(todo.values())):
                out[i] = text
                _memo_put(model, keys[i], text)
    # fan the unique results back out over every position that asked for the same key
    first: dict[str, str] = {}
    for i, k in enumerate(keys):
        if out[i] is not None:
            first.setdefault(k, out[i])
    hits = misses = 0
    for i, k in enumerate(keys):
        if out[i] is None:
            out[i] = first.get(k, "")
        if k in todo and i == todo[k]:
            misses += 1
        else:
            hits += 1
    _bump("hits", hits)
    _bump("misses", misses)
    return [x if x is not None else "" for x in out]


def stats() -> dict:
    with _STATS_LOCK:
        s = dict(_STATS)
    tot = s["hits"] + s["misses"]
    s["cache_hit_rate"] = (s["hits"] / tot) if tot else 0.0
    return s


def reset_stats() -> None:
    with _STATS_LOCK:
        for k in _STATS:
            _STATS[k] = 0
