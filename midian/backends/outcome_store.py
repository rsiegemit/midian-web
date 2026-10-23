"""Scored outcomes of a de-cloned live population (LLMBackend declone), kept beside the population instead of its
sampled texts (why: midian.backends.llm, DE-CLONED POPULATIONS).

Per population: `<population dir>/outcomes/<host>_<pid>.sqlite`, one writer per process as in midian.llm_client (NFS:
`nolock`), every shard read at first use, this process's shard opened at its first write. The build
(scripts/data/build_declone.py) fills it; a method that probes past the build adds its outcomes, so every later reader
sees the same answer -- except under concurrency: two processes that miss the same key both sample it (vLLM sampling
is not batch-invariant, so their draws can differ) and each writes its own shard; a reader then gets the last shard it
loads. Measured at 0.07% of stored outcomes; rows reproduce to about +/-0.01 (docs/reproducing.md, Tier 4)."""
from __future__ import annotations

import os
import socket
import sqlite3
from pathlib import Path


def _open(path: Path, readonly: bool = False) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?nolock=1" + ("&mode=ro" if readonly else ""), uri=True,
                          check_same_thread=False, timeout=60.0)
    if not readonly:
        con.execute("PRAGMA journal_mode=TRUNCATE")
        con.execute("CREATE TABLE IF NOT EXISTS o (a INTEGER, f INTEGER, i INTEGER, v INTEGER, PRIMARY KEY (a, f, i))")
        con.commit()
    return con


class OutcomeStore:
    def __init__(self, directory: Path):
        self.dir, self._mem, self._shard = Path(directory) / "outcomes", None, None

    def _load(self) -> dict:
        if self._mem is None:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._mem = {}
            for f in sorted(self.dir.glob("*.sqlite")):
                try:
                    con = _open(f, readonly=True)
                    self._mem.update(((a, fa, i), v) for a, fa, i, v in con.execute("SELECT a, f, i, v FROM o"))
                    con.close()
                except sqlite3.Error:
                    continue                                    # mid-write: its rows come next process
        return self._mem

    def get(self, keys) -> list:
        """Outcome per (agent, family, instance) key, None where not stored."""
        mem = self._load()
        return [mem.get(k) for k in keys]

    def put(self, rows) -> None:
        """rows: ((agent, family, instance), outcome)."""
        mem, rows = self._load(), [(int(a), int(f), int(i), int(v)) for (a, f, i), v in rows]
        mem.update(((a, f, i), v) for a, f, i, v in rows)
        if self._shard is None:
            self._shard = _open(self.dir / f"{socket.gethostname()}_{os.getpid()}.sqlite")
        self._shard.executemany("INSERT OR REPLACE INTO o VALUES (?, ?, ?, ?)", rows)
        self._shard.commit()
