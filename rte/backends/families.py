"""Task families: one small adapter per data source (CONTRACT "swappable by config").

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

from ..stable_hash import stable_seed_32

DEFAULT_SOURCE = "rg"

# 16 diverse generators with programmatic verifiers (SPEC §1, K=16 default).
# `propositional_logic` and `graph_color` are deliberately absent: in reasoning-gym 0.1.19 both
# emit entries with answer=None and score their own gold answer 0.0, so no agent can ever be
# right on them. scripts/probe_families.py re-checks every name below.
FAMILIES_16 = ["basic_arithmetic", "chain_sum", "leg_counting", "word_sorting", "letter_counting",
               "syllogism", "family_relationships", "gcd", "lcm", "prime_factorization",
               "base_conversion", "caesar_cipher", "spell_backward", "number_sorting",
               "bitwise_arithmetic", "simple_equations"]
FAMILIES_64 = FAMILIES_16 + [
    "count_bits", "count_primes", "decimal_arithmetic", "decimal_chain_sum", "products",
    "fraction_simplification", "number_format", "number_filtering", "number_sequence",
    "power_function", "polynomial_equations", "polynomial_multiplication", "complex_arithmetic",
    "simple_geometry", "advanced_geometry", "time_intervals", "calendar_arithmetic",
    "aiw", "knights_knaves", "coin_flip", "dice", "jugs", "quantum_lock", "circuit_logic",
    "letter_jumble", "group_anagrams", "isomorphic_strings", "palindrome_generation",
    "palindrome_partitioning", "word_sequence_reversal", "sentence_reordering", "ransom_note",
    "string_insertion", "string_splitting", "string_manipulation", "string_synthesis",
    "binary_alternation", "binary_matrix", "rotate_matrix", "spiral_matrix", "manipulate_matrix",
    "pool_matrix", "rectangle_count", "largest_island", "shortest_path",
    "course_schedule", "tower_of_hanoi", "self_reference"]

SEED_KEY = "_rte_instance"      # generate() stamps the seed so score() can rebuild the verifier


@lru_cache(maxsize=8192)
def _rg_dataset(name: str, seed: int):
    from reasoning_gym.factory import create_dataset
    return create_dataset(name, size=1, seed=seed)


class ReasoningGym:
    """reasoning-gym adapter: the library seeds the problem from the dataset seed, so the
    instance seed IS the dataset seed and (family, instance) regenerates the identical problem."""

    def __init__(self, name: str):
        self.name = name

    def generate(self, instance_seed: int) -> dict:
        seed = int(instance_seed) & 0x7FFFFFFF
        return dict(_rg_dataset(self.name, seed)[0], **{SEED_KEY: seed})

    def question(self, entry: dict) -> str:
        return entry["question"]

    def score(self, answer: str, entry: dict) -> float:
        # Some verifiers parse the answer (prime_factorization calls int()): a malformed model
        # answer is a wrong answer, never a crash of the run.
        try:
            return float(_rg_dataset(self.name, entry[SEED_KEY]).score_answer(answer=answer, entry=entry))
        except Exception:                                       # noqa: BLE001
            return 0.0


SOURCES = {"rg": ReasoningGym}


@lru_cache(maxsize=1024)
def adapter(family: str):
    src, _, name = family.rpartition(":")
    return SOURCES[src or DEFAULT_SOURCE](name)


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
