"""Task families: one small adapter per data source (swappable by config).

Adapter protocol (duck-typed):
    generate(instance_seed) -> entry        deterministic; same seed == same problem
    question(entry) -> str                  what the agent is shown
    score(answer, entry) -> float           the programmatic verifier, 1.0 == correct

Reasoning Gym is ONE adapter. Adding a dataset = adding an adapter to SOURCES and listing its
family names; the backend, the methods and the world are untouched. A family name may be
qualified ("rg:gcd"); a bare name uses DEFAULT_SOURCE.
"""
from __future__ import annotations

from functools import lru_cache

from ..config import flag
from ..stable_hash import stable_seed_32

DEFAULT_SOURCE = "rg"

# 16 diverse generators with programmatic verifiers (K=16 default).
# `propositional_logic` and `graph_color` are deliberately absent: in reasoning-gym 0.1.19 both
# emit entries with answer=None and score their own gold answer 0.0, so no agent can ever be
# right on them. scripts/data/probe_families.py re-checks every name below.
# leg_counting / caesar_cipher / base_conversion / bitwise_arithmetic / spell_backward / word_sorting
# were demoted to the K=64 tail: measured <=0.20 on BOTH the 7B and the 14B, so they carry no signal
# about which agent to route to (docs/archive/DEVIATIONS.md). True/False families are avoided here on
# purpose -- a chance-level responder floors at 0.50, which no weak agent can score below.
FAMILIES_16 = ["basic_arithmetic", "chain_sum", "letter_counting", "syllogism",
               "family_relationships", "gcd", "lcm", "prime_factorization", "number_sorting",
               "simple_equations", "count_bits", "number_format", "time_intervals",
               "word_sequence_reversal", "binary_alternation", "calendar_arithmetic"]
FAMILIES_64 = FAMILIES_16 + [
    "leg_counting", "caesar_cipher", "base_conversion", "bitwise_arithmetic", "spell_backward",
    "word_sorting", "count_primes", "decimal_arithmetic", "decimal_chain_sum", "products",
    "fraction_simplification", "number_filtering", "number_sequence",
    "power_function", "polynomial_equations", "polynomial_multiplication", "complex_arithmetic",
    "simple_geometry", "advanced_geometry", "palindrome_generation",
    "aiw", "knights_knaves", "coin_flip", "dice", "jugs", "quantum_lock", "circuit_logic",
    "letter_jumble", "group_anagrams", "isomorphic_strings",
    "palindrome_partitioning", "sentence_reordering", "ransom_note",
    "string_insertion", "string_splitting", "string_manipulation", "string_synthesis",
    "binary_matrix", "rotate_matrix", "spiral_matrix", "manipulate_matrix",
    "pool_matrix", "rectangle_count", "largest_island", "shortest_path",
    "course_schedule", "tower_of_hanoi", "self_reference"]

SEED_KEY = "_rte_instance"      # generate() stamps the seed so score() can rebuild the verifier

# Per-family generator settings, passed straight to `create_dataset`. This is the difficulty knob:
# at stock settings `basic_arithmetic` (up to 6 terms, 4 digits) scored 0.25 on the 7B and 0.10 on
# the 14B, far below the 0.70-0.95 band wanted for a specialty family.
PARAMS: dict[str, dict] = {
    "basic_arithmetic": {"max_terms": 3, "max_digits": 2},
    "chain_sum": {"max_terms": 3, "max_digits": 2},
}


_RG_PATCHED = []


def _rg_speedup():
    """Opt-in (RTE_RG_CACHE=1), identical problems: letter_counting / word_sequence_reversal re-read and regex-scan a
    word corpus in every dataset __init__; cache the file read and the scan (a fresh list per call), ~27x per
    problem."""
    import functools
    import importlib
    import re as _re

    import reasoning_gym.data as rgd
    rgd.read_data_file = functools.lru_cache(maxsize=None)(rgd.read_data_file)
    scan = functools.lru_cache(maxsize=64)(lambda p, s, fl=0: tuple(_re.findall(p, s, fl)))

    class _Re:
        def __getattr__(self, k):
            return getattr(_re, k)
        @staticmethod
        def findall(p, s, flags=0):
            return list(scan(p, s, flags))
    for m in ("letter_counting", "word_sequence_reversal"):
        mod = importlib.import_module(f"reasoning_gym.algorithmic.{m}")
        mod.re = _Re()
        if hasattr(mod, "read_data_file"):
            mod.read_data_file = rgd.read_data_file
    _RG_PATCHED.append(True)


@lru_cache(maxsize=8192)
def _rg_dataset(name: str, seed: int, params: tuple = ()):
    if flag("RTE_RG_CACHE") and not _RG_PATCHED:
        _rg_speedup()
    from reasoning_gym.factory import create_dataset
    return create_dataset(name, size=1, seed=seed, **dict(params))


class ReasoningGym:
    """reasoning-gym adapter: the library seeds the problem from the dataset seed, so the
    instance seed IS the dataset seed and (family, instance) regenerates the identical problem."""

    def __init__(self, name: str, **params):
        self.name, self.params = name, tuple(sorted(params.items()))

    def generate(self, instance_seed: int) -> dict:
        seed = int(instance_seed) & 0x7FFFFFFF
        return dict(_rg_dataset(self.name, seed, self.params)[0], **{SEED_KEY: seed})

    def question(self, entry: dict) -> str:
        return entry["question"]

    def score(self, answer: str, entry: dict) -> float:
        # Some verifiers parse the answer (prime_factorization calls int()): a malformed model
        # answer is a wrong answer, never a crash of the run.
        try:
            ds = _rg_dataset(self.name, entry[SEED_KEY], self.params)
            return float(ds.score_answer(answer=answer, entry=entry))
        except Exception:                                       # noqa: BLE001
            return 0.0


SOURCES = {"rg": ReasoningGym}


@lru_cache(maxsize=1024)
def adapter(family: str):
    src, _, name = family.rpartition(":")
    return SOURCES[src or DEFAULT_SOURCE](name, **PARAMS.get(family, {}))


def names(K: int) -> list[str]:
    pool = FAMILIES_16 if K <= 16 else FAMILIES_64
    if len(pool) < K:
        raise ValueError(f"only {len(pool)} families listed, need K={K}")
    return pool[:K]


def entry(family: str, instance: int) -> dict:
    return adapter(family).generate(instance)


def question(family: str, instance: int) -> str:
    a = adapter(family)
    return a.question(a.generate(instance))


def correct(family: str, instance: int, answer: str) -> int:
    a = adapter(family)
    return int(a.score(answer, a.generate(instance)) >= 0.99)


def _aux(family: str, kind: str) -> int:
    """Seed for prompt furniture. Its own stable_seed namespace, so colliding with a live task
    instance is a ~1e-6 event that would leak at most one worked example."""
    return int(stable_seed_32("aux", kind, family)) & 0x7FFFFFFF


@lru_cache(maxsize=256)
def describe(family: str) -> str:
    """One sentence per family: the humanised name plus one example question. Deterministic --
    it is prompt text AND the retrieval key for llm_supervisor and the framework rivals."""
    q = " ".join(question(family, _aux(family, "desc")).split())[:240]
    return f"{family.replace('_', ' ')} problems. Example question: {q}"


@lru_cache(maxsize=256)
def exemplar(family: str) -> tuple[str, str]:
    """One worked (question, answer) pair, from a seed disjoint from any task instance."""
    e = entry(family, _aux(family, "exemplar"))
    return " ".join(adapter(family).question(e).split())[:600], str(e["answer"])[:200]
