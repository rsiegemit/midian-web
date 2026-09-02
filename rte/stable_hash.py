"""Process-stable seeding, because `hash()` on a string is not.

WHY THIS MODULE EXISTS.  `PYTHONHASHSEED` randomises `str.__hash__` per process,
and `tuple.__hash__` folds its elements' hashes in.  So

    random.Random(hash((seed, n, name)) & 0xFFFFFFFF)      # name is a str

draws a DIFFERENT stream in every process, and an experiment seeded that way is
not reproducible from its own `seed` argument.  This was not hypothetical: E8's
per-task protocol RNG was seeded exactly that way, and across `PYTHONHASHSEED`
0-5 the midian-vs-broadcast delta ranged -0.0055 to -0.0667 -- a 12x swing on a
family whose pre-registered bar is 0.05 (STATUS W132).

Integers are safe (`hash(3) == 3`), so the two int-only call sites were never
wrong.  They are routed through here anyway: the reader of a seeding line should
not have to know which argument types are hash-stable, and the next person to add
a string to one of those tuples must not silently reintroduce the defect.

`blake2b` is already this repo's stable-digest primitive (`HashingEmbedder`), so
this uses it rather than adding a second convention.
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
