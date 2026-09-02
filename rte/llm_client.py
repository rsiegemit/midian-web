"""The single place any LLM call goes through. OpenAI-compatible, temperature 0, disk-memoized.

Endpoints: one file per served model under `$RTE_DATA/endpoints.d/` (written lock-free by
scripts/_register_endpoint.py), merged into `$RTE_DATA/endpoints.json` for the contract.

NFS: `fcntl.flock` never returns on this cluster's netscratch mount (instant on $HOME and /tmp),
and the env's SQLite 3.50.3 picks a locking style that hits it, so every database opens `nolock=1`.
The memo is therefore SHARDED PER PROCESS: this process writes only its own
`$RTE_DATA/cache/memo_<host>_<pid>.sqlite`, and at startup reads EVERY `*.sqlite` in that directory
into one in-memory dict. No two processes ever write the same file, so the live grid can run one
process per method against the shared cache; a process sees whatever finished before it started.
`python -m rte.llm_client compact` merges the shards between stages.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
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
_clients: dict[str, object] = {}
_mem: dict[str, str] | None = None      # every shard, read once at startup
_shard: sqlite3.Connection | None = None


class NoEndpointsError(RuntimeError):
    """No vLLM fleet configured — callers degrade with a clear message."""


def served_models() -> list[str]:
    """Model ids actually served (replica aliases "<model>#<job>" collapsed onto their model)."""
    return sorted({k.split("#")[0] for k in endpoints()})


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


def _open(path: Path, readonly: bool = False) -> sqlite3.Connection:
    uri = f"file:{path}?nolock=1" + ("&mode=ro" if readonly else "")
    con = sqlite3.connect(uri if NOLOCK else str(path), uri=NOLOCK, check_same_thread=False,
                          timeout=60.0)
    if not readonly:
        con.execute("PRAGMA journal_mode=TRUNCATE")
        con.execute("CREATE TABLE IF NOT EXISTS memo (k TEXT PRIMARY KEY, v TEXT)")
        con.commit()
    return con


def _memo() -> tuple[dict, sqlite3.Connection]:
    """(all cached answers, this process's write shard). Every `*.sqlite` in CACHE_DIR is read
    once; only `memo_<host>_<pid>.sqlite` is ever written, so processes never share a writer."""
    global _mem, _shard
    if _mem is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        mine = CACHE_DIR / f"memo_{socket.gethostname()}_{os.getpid()}.sqlite"
        _mem = {}
        _refresh(force=True)
        _shard = _open(mine)
    return _mem, _shard


_seen: dict = {}                              # shard file -> last rowid read
_last_refresh = 0.0
REFRESH_S = 30.0
ENDPOINT_TTL = 20.0
_lat: dict = {}                                # endpoint url -> EWMA request latency (this process)                          # seconds a process sticks to one endpoint before re-picking among replicas


def _refresh(force: bool = False) -> None:
    """Pull rows other processes have written since we last looked (rowid > last seen per shard), so
    concurrent grid jobs share generations live instead of only at startup. Cheap; rate-limited."""
    global _last_refresh
    if not force and time.time() - _last_refresh < REFRESH_S:
        return
    _last_refresh = time.time()
    mine = f"memo_{socket.gethostname()}_{os.getpid()}.sqlite"
    for f in sorted(CACHE_DIR.glob("*.sqlite")):
        if f.name == mine:
            continue
        try:
            con = _open(f, readonly=True)
            rows = con.execute("SELECT rowid, k, v FROM memo WHERE rowid > ?", (_seen.get(f, 0),)).fetchall()
            con.close()
        except sqlite3.Error:
            continue                            # not one of ours, or mid-write: next time
        if rows:
            _seen[f] = rows[-1][0]
            _mem.update((k, v) for _, k, v in rows)


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
            if model in _clients and time.time() - _clients[model][1] > ENDPOINT_TTL:
                _clients.pop(model)                               # re-pick: replicas may have joined since
            if model not in _clients:
                eps = endpoints()
                if model not in eps:
                    raise NoEndpointsError(f"model {model!r} not served; have {sorted(eps)}")
                from openai import OpenAI
                urls = [u for k, u in eps.items() if k == model or k.startswith(model + "#")]   # replicas: "<model>#<job>"
                # latency-aware: usually the endpoint with the lowest recent latency, sometimes a random one (explore)
                url = random.choice(urls) if random.random() < 0.2 else min(urls, key=lambda u: _lat.get(u, 0.0))
                _clients[model] = (OpenAI(base_url=url, api_key="EMPTY", timeout=600.0, max_retries=0), time.time())
            t0 = time.time()
            r = _clients[model][0].chat.completions.create(
                model=model, messages=for_model(model, messages), max_tokens=int(max_tokens),
                temperature=0.0, seed=0)
            u = str(_clients[model][0].base_url); _lat[u] = 0.7 * _lat.get(u, time.time() - t0) + 0.3 * (time.time() - t0)
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
    with _LOCK:
        mem, shard = _memo()
        answer = {k: mem[k] for k in keys if k in mem}
        todo = {k: i for i, k in enumerate(keys) if k not in mem}
    if todo:
        with _LOCK:                               # another job may have generated these since we started
            _refresh()
            answer.update({k: mem[k] for k in todo if k in mem})
            todo = {k: i for k, i in todo.items() if k not in mem}
    _bump("hits", sum(1 for k in keys if k not in todo))
    _bump("misses", len(todo))                    # misses == unique generations, not positions
    if todo:
        with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(todo)))) as ex:
            texts = list(ex.map(lambda i: _generate(model, batch[i], max_tokens), todo.values()))
        rows = list(zip(todo, texts))
        answer.update(rows)
        with _LOCK:
            mem.update(rows)
            shard.executemany("INSERT OR REPLACE INTO memo VALUES (?, ?)", rows)
            shard.commit()
    return [answer[k] for k in keys]


def stats() -> dict:
    with _LOCK:
        s = dict(_STATS)
    tot = s["hits"] + s["misses"]
    return s | {"cache_hit_rate": s["hits"] / tot if tot else 0.0}


def reset_stats() -> None:
    with _LOCK:
        _STATS.update(dict.fromkeys(_STATS, 0))


def compact() -> int:
    """Merge every shard into one file and delete the ones merged. Run BETWEEN stages: a shard
    still being written by a live process would lose whatever it writes after the merge."""
    mem, _ = _memo()
    out = CACHE_DIR / "memo_compact.sqlite"
    old = [f for f in sorted(CACHE_DIR.glob("*.sqlite")) if f != out]
    con = _open(out)
    con.executemany("INSERT OR REPLACE INTO memo VALUES (?, ?)", list(mem.items()))
    con.commit()
    con.close()
    for f in old:
        f.unlink()
    print(f"compacted {len(old)} shard(s) -> {out} ({len(mem)} rows)")
    return len(mem)


if __name__ == "__main__":                        # python -m rte.llm_client compact
    import sys
    if sys.argv[1:2] == ["compact"]:
        compact()
    else:
        print(json.dumps(endpoints(), indent=2))
