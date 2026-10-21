"""Process-stable seeding, because `hash()` on a string is not.

`PYTHONHASHSEED` randomises `str.__hash__` per process, and `tuple.__hash__` folds its elements' hashes in, so

    random.Random(hash((seed, n, name)) & 0xFFFFFFFF)      # name is a str

draws a DIFFERENT stream in every process: an experiment seeded that way is not reproducible from its own `seed`
argument. Every seed in the package therefore goes through `stable_seed_32`, integers included, so a reader of a
seeding line never has to know which argument types are hash-stable. The digest is blake2b.
"""
from __future__ import annotations

import hashlib

__all__ = ["stable_seed_32"]

_SEP = "\x1f"          # unit separator: cannot occur in an arm or protocol name


def stable_seed_32(*parts) -> int:
    """A deterministic non-negative 32-bit int derived from `parts`, identical in every process, on every platform, in
    every Python build. `parts` are rendered with `repr` and joined on a separator that cannot appear inside a name, so
    ("a", "b") and ("a\\x1fb",) cannot collide into the same seed; the 8-byte digest is masked to its low 32 bits."""
    key = _SEP.join(repr(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") & 0xFFFFFFFF
