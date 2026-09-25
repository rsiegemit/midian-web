"""Process-stable seeding, because `hash()` on a string is not.

`PYTHONHASHSEED` randomises `str.__hash__` per process, and `tuple.__hash__` folds its elements' hashes in, so

    random.Random(hash((seed, n, name)) & 0xFFFFFFFF)      # name is a str

draws a DIFFERENT stream in every process: an experiment seeded that way is not reproducible from its own `seed`
argument. Every seed in the package therefore goes through `stable_seed_32`, integers included, so a reader of a
seeding line never has to know which argument types are hash-stable. The digest is blake2b.
"""
from __future__ import annotations

import hashlib

__all__ = ["stable_seed", "stable_seed_32"]

_SEP = "\x1f"          # unit separator: cannot occur in an arm or protocol name


def stable_seed(*parts, bits: int = 64) -> int:
    """A deterministic non-negative int derived from `parts`.

    Identical in every process, on every platform, in every Python build --
    which is the whole point.  `parts` are rendered with `repr` and joined on a
    separator that cannot appear inside a name, so ("a", "b") and ("a\\x1fb",)
    cannot collide into the same seed.
    """
    if not 8 <= bits <= 512:
        raise ValueError(f"bits must be in [8, 512], got {bits}")
    key = _SEP.join(repr(p) for p in parts).encode("utf-8")
    n_bytes = max(1, bits // 8)
    digest = hashlib.blake2b(key, digest_size=n_bytes).digest()
    return int.from_bytes(digest, "big")


def stable_seed_32(*parts) -> int:
    """`stable_seed` masked to 32 bits, the width the call sites used."""
    return stable_seed(*parts, bits=64) & 0xFFFFFFFF
