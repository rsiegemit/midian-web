"""The primary backend: real vLLM-served models as agents, Reasoning Gym as the task families.

An agent is a *profile*: (base model from the ladder, a set of specialty families served at
full capability, handicap flags on the rest, a tool in {calculator, python, none}).  Executing
a task is a real generation by that agent's model under that agent's prompt scaffolding; the
outcome is `reasoning_gym.score_answer(...) >= 0.99`.

Determinism.  A task is `(family f, instance)`; the concrete problem is regenerated with
`reasoning_gym.create_dataset(name, size=1, seed=instance)`, so every agent sees the identical
instance and the paired task stream is honoured.  Generations run at temperature 0, seed 0.

Memoization.  What determines an agent's answer is not its id but its *prompt signature*
`(model, handicapped_on_f, tool_on_f, max_tokens)` -- at most |ladder| x 2 x 3 distinct values.
Two agents with the same signature emit the same prompt and therefore the same answer, so the
disk memo in `rte.llm_client` is keyed on the signature, not the agent.  That is what makes
`true_skill()` (n x K x 200 probes) affordable: the unique generation count is
`#signatures x K x measure_probes`, independent of n.

Handicaps (SPEC §1 "exemplars withheld, difficulty capped, family tool removed"):
  * no worked exemplar,
  * no family description -- the handicapped agent is told only to solve the problem, so it has
    no framing for what kind of task this is,
  * the family tool is removed.

The spec's "difficulty capped" cannot be applied per agent: every agent must see the SAME instance
for the paired task stream and the shared verifier to mean anything. An earlier version substituted
a cap on the agent's generation budget (160 vs 512 tokens) and that turned out to be NON-MONOTONE
-- measured on Qwen2.5-0.5B, the "handicapped" configuration beat the unhandicapped one on
syllogism 0.85 vs 0.30, because a short budget forces a terse answer while a long one lets a small
model ramble past the answer format. A handicap that sometimes helps is worse than no handicap, so
the budget cap is off by default (`handicap_max_tokens=None`); pass a value to study it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..stable_hash import stable_seed, stable_seed_32
from ..world import Task

RTE_DATA = Path(os.environ.get("RTE_DATA", "/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte"))
POP_DIR = Path(os.environ.get("RTE_POPULATIONS", RTE_DATA / "populations"))

# --------------------------------------------------------------------------- the model ladder
LADDER = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
          "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct",
          "google/gemma-2-2b-it", "google/gemma-2-9b-it"]
SMALL = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]
BIG = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "google/gemma-2-9b-it"]
SUPERVISOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LARGE_MODELS = frozenset({"Qwen/Qwen2.5-14B-Instruct", "google/gemma-2-9b-it"})   # fewer measure probes
TOOLS = ("calculator", "python", "none")

# --------------------------------------------------------------------------- family lists
# 16 diverse generators with programmatic verifiers (SPEC §1, K=16 default).
# NOTE: `propositional_logic` and `graph_color` are excluded -- in reasoning-gym 0.1.19 both emit
# entries with answer=None and score their own gold answer 0.0, so they can never be solved.
# `syllogism` (logic) and `circuit_logic` take their places. Verified by scripts/probe_families.py.
FAMILIES_16 = ["basic_arithmetic", "chain_sum", "leg_counting", "word_sorting", "letter_counting",
               "syllogism", "family_relationships", "gcd", "lcm", "prime_factorization",
               "base_conversion", "caesar_cipher", "spell_backward", "number_sorting",
               "bitwise_arithmetic", "simple_equations"]
# 48 more for K=64. All are registered generators in reasoning-gym >= 0.1.19.
FAMILIES_EXTRA_48 = [
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
FAMILIES_64 = FAMILIES_16 + FAMILIES_EXTRA_48

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
CALC_RE = re.compile(r"<calc>(.*?)</calc>", re.DOTALL | re.IGNORECASE)
CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
RATING_RE = re.compile(r"(?<![\d.])(0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])")


# --------------------------------------------------------------------------- reasoning gym glue
def default_families(K: int) -> list[str]:
    """The K family names, filtered to what the installed reasoning-gym actually registers."""
    pool = FAMILIES_16 if K <= 16 else FAMILIES_64
    try:
        from reasoning_gym.factory import DATASETS
        avail = [f for f in pool if f in DATASETS]
        if len(avail) < K:                                  # top up from the rest of the registry
            avail += [f for f in sorted(DATASETS) if f not in avail]
    except Exception:                                       # noqa: BLE001 -- tests may run without rg
        avail = pool
    if len(avail) < K:
        raise RuntimeError(f"only {len(avail)} reasoning-gym families available, need K={K}")
    return list(avail[:K])


@lru_cache(maxsize=8192)
def _dataset(family: str, instance: int):
    from reasoning_gym.factory import create_dataset
    return create_dataset(family, size=1, seed=int(instance) & 0x7FFFFFFF)


def instance_entry(family: str, instance: int) -> dict:
    """The concrete problem for (family, instance). Deterministic; identical for every agent."""
    return _dataset(family, instance)[0]


def score(family: str, instance: int, answer: str) -> int:
    ds, entry = _dataset(family, instance), instance_entry(family, instance)
    try:
        s = ds.score_answer(answer=answer, entry=entry)
    except TypeError:
        try:
            s = ds.score_answer(answer, entry)
        except Exception:                                    # noqa: BLE001
            return 0
    except Exception:                                        # noqa: BLE001
        # e.g. prime_factorization does int() on the answer: a malformed model answer is
        # a wrong answer, never a crash of the whole run.
        return 0
    try:
        return int(float(s) >= 0.99)
    except (TypeError, ValueError):
        return 0


def _aux_instance(family: str, kind: str) -> int:
    """Instance seed for prompt furniture (exemplars, family descriptions). Drawn from a
    separate stable_seed namespace; a collision with a live task instance is a ~1e-6 event
    across the whole grid and would leak at most one worked example."""
    return int(stable_seed_32("aux", kind, family)) & 0x7FFFFFFF


@lru_cache(maxsize=256)
def family_description(family: str) -> str:
    """A short natural-language description: the humanised name plus a sample question.
    Used in prompts and by `llm_supervisor`'s TF-IDF retrieval, so it must be deterministic."""
    pretty = family.replace("_", " ")
    try:
        q = " ".join(instance_entry(family, _aux_instance(family, "desc"))["question"].split())[:240]
        return f"{pretty} problems. Example question: {q}"
    except Exception:                                        # noqa: BLE001
        return f"{pretty} problems."


@lru_cache(maxsize=256)
def family_exemplar(family: str) -> tuple[str, str]:
    """One worked (question, answer) pair, drawn from a seed disjoint from any task instance."""
    e = instance_entry(family, _aux_instance(family, "exemplar"))
    return " ".join(e["question"].split())[:600], str(e["answer"])[:200]


# --------------------------------------------------------------------------- tools
_CALC_OPS = set("0123456789.+-*/%()e ")


def run_calculator(expr: str) -> str:
    """Deterministic arithmetic on a model-emitted expression. Literal-only AST evaluation."""
    import ast
    import operator as op
    expr = expr.strip().replace("^", "**").replace(",", "")
    if len(expr) > 400:
        return "ERROR: expression too long"
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
           ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow, ast.USub: op.neg,
           ast.UAdd: op.pos}

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            if isinstance(node.op, ast.Pow):
                r = ev(node.right)
                if abs(r) > 64:
                    raise ValueError("exponent too large")
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expression element {type(node).__name__}")

    try:
        return str(ev(ast.parse(expr, mode="eval")))
    except Exception as e:                                   # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {e}"


_PY_PREAMBLE = (
    "import socket as _s\n"
    "def _blocked(*a, **k):\n"
    "    raise OSError('network disabled in the RTE python tool')\n"
    "_s.socket = _blocked; _s.create_connection = _blocked; _s.socket_socketpair = _blocked\n"
)


def run_python(code: str, timeout: float = 5.0) -> str:
    """Run a model-emitted code block in a throwaway subprocess: 5 s wall clock, CPU/address-space
    rlimits, no network, isolated interpreter (`-I`), empty environment. Returns stdout (truncated).

    This is containment, not a security boundary: it stops runaway loops, accidental network use
    and stray imports of our own modules, but a determined escape is out of scope (the code is
    written by a 0.5-14B model answering an arithmetic puzzle)."""
    if len(code) > 8000:
        return "ERROR: code too long"

    def limits():
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 1, int(timeout) + 1))
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))

    try:
        r = subprocess.run([sys.executable, "-I", "-c", _PY_PREAMBLE + code],
                           capture_output=True, text=True, timeout=timeout, cwd="/tmp",
                           env={"PATH": "/usr/bin:/bin", "HOME": "/tmp", "PYTHONHASHSEED": "0"},
                           preexec_fn=limits)
        out = (r.stdout or "") + (("\nSTDERR: " + r.stderr) if r.returncode else "")
    except subprocess.TimeoutExpired:
        return "ERROR: timed out after 5s"
    except Exception as e:                                   # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {e}"
    return out.strip()[:2000] or "(no output)"


# --------------------------------------------------------------------------- profiles
def draw_profiles(n: int, K: int, dist: str, seed: int) -> list[dict]:
    """The population. `dist` is realized by how profiles are drawn (SPEC §3, llm column);
    the resulting S is *measured*, never assumed."""
    rng = np.random.default_rng(stable_seed_32(seed, "profiles", n, K, dist))
    profiles = []
    for a in range(n):
        if dist == "specialist":
            model = LADDER[int(rng.integers(len(LADDER)))]
            spec = sorted(int(x) for x in rng.choice(K, size=min(3, K), replace=False))
            tool = TOOLS[int(rng.integers(3))]
        elif dist == "heavy_tail":
            expert = bool(rng.random() < 0.1)
            model = BIG[int(rng.integers(len(BIG)))] if expert else SMALL[int(rng.integers(len(SMALL)))]
            spec = list(range(K)) if expert else []
            tool = "python" if expert else "none"
        elif dist == "bimodal":
            good = bool(rng.random() < 0.2)
            model = "Qwen/Qwen2.5-7B-Instruct" if good else "Qwen/Qwen2.5-0.5B-Instruct"
            spec = list(range(K)) if good else []
            tool = "python" if good else "none"
        elif dist == "correlated":
            model = LADDER[int(rng.integers(len(LADDER)))]
            good_groups = {g for g in range(4) if rng.random() < 0.5}
            spec = [f for f in range(K) if (f % 4) in good_groups]
            tool = TOOLS[int(rng.integers(3))]
        elif dist == "iid_uniform":
            model = LADDER[int(rng.integers(len(LADDER)))]
            spec = [f for f in range(K) if rng.random() < 0.5]
            tool = TOOLS[int(rng.integers(3))]
        else:
            raise ValueError(f"unknown skill_dist {dist!r}")
        profiles.append({"id": a, "model": model, "specialty": list(spec), "tool": tool})
    return profiles


def signature(profile: dict, f: int, max_tokens: int, handicap_max_tokens: int) -> tuple:
    """Everything that determines the prompt (and hence the answer) for (agent, family f).
    Agents that share a signature share cache entries -- see the module docstring."""
    hand = f not in profile["specialty"]
    tool = "none" if hand else profile["tool"]
    return (profile["model"], bool(hand), tool, handicap_max_tokens if hand else max_tokens)


def _sig_hash(sig: tuple) -> str:
    return f"{stable_seed(*sig, bits=96):024x}"


# --------------------------------------------------------------------------- prompts
def build_prompt(family: str, question: str, handicapped: bool, tool: str) -> list[dict]:
    """A handicapped agent loses the family description, the worked exemplar and the tool.
    It keeps the answer-format instruction: withholding that would measure format compliance
    rather than skill, and every agent must be scored by the same verifier."""
    if handicapped:
        parts = ["Solve the following problem."]
    else:
        parts = [f"You are solving a '{family}' task.", family_description(family)]
        q, a = family_exemplar(family)
        parts.append(f"Worked example.\nQuestion: {q}\nAnswer: {a}")
    if tool == "calculator":
        parts.append("You have a calculator. To use it, emit exactly one line "
                     "<calc>ARITHMETIC EXPRESSION</calc> and stop; the result will be given to you.")
    elif tool == "python":
        parts.append("You have a Python sandbox (5 s, no network). To use it, emit exactly one "
                     "```python ... ``` block that prints what you need, and stop; "
                     "its stdout will be given to you.")
    parts.append("Give your final answer inside <answer></answer> tags, with nothing else inside "
                 "the tags.")
    return [{"role": "system", "content": "\n\n".join(parts)},
            {"role": "user", "content": question}]


def extract_answer(text: str) -> str:
    m = ANSWER_RE.findall(text or "")
    if m:
        return m[-1].strip()
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def find_tool_call(text: str, tool: str) -> str | None:
    """The tool block the model emitted, if any -- but not when it already answered."""
    if not text or ANSWER_RE.search(text):
        return None
    if tool == "calculator":
        m = CALC_RE.search(text)
        return m.group(1) if m else None
    if tool == "python":
        m = CODE_RE.search(text)
        return m.group(1) if m else None
    return None


def run_tool(tool: str, payload: str) -> str:
    return run_calculator(payload) if tool == "calculator" else run_python(payload)


# --------------------------------------------------------------------------- the backend
class LLMBackend:
    def __init__(self, n: int, K: int = 16, dist: str = "specialist", seed: int = 0,
                 rng: np.random.Generator | None = None, families: list[str] | None = None,
                 measure_probes: int = 200, measure_probes_large: int = 60,
                 max_tokens: int = 512, handicap_max_tokens: int | None = None,
                 declared_noise: float = 0.05, population_dir: str | None = None,
                 concurrency: int = 64, **_):
        self.n, self.K = int(n), int(K)
        self.seed = int(seed)
        self.dist = dist
        self.families = list(families) if families else default_families(self.K)
        if len(self.families) != self.K:
            raise ValueError(f"got {len(self.families)} families for K={self.K}")
        self.measure_probes = int(measure_probes)
        self.measure_probes_large = int(measure_probes_large)
        self.max_tokens = int(max_tokens)
        # None => handicapped agents get the same budget; the handicap is prompt-side only.
        self.handicap_max_tokens = int(handicap_max_tokens) if handicap_max_tokens is not None \
            else self.max_tokens
        self.declared_noise = float(declared_noise)
        self.concurrency = int(concurrency)
        self.profiles = draw_profiles(self.n, self.K, dist, self.seed)
        self.dir = Path(population_dir) if population_dir else \
            POP_DIR / f"{dist}_n{self.n}_K{self.K}_seed{self.seed}"
        self._S: np.ndarray | None = None
        self._descriptions: list[str] | None = None
        self._counts = {"executions": 0, "tool_calls": 0, "batches": 0}
        self._lock = threading.Lock()
        _set_current(self)

    # ---- ids ------------------------------------------------------------
    def _sig(self, a: int, f: int) -> tuple:
        return signature(self.profiles[a], f, self.max_tokens, self.handicap_max_tokens)

    def _key(self, a: int, f: int, instance: int, stage: str = "a") -> str:
        return f"{_sig_hash(self._sig(a, f))}:{f}:{int(instance)}:{stage}"

    # ---- one generation, possibly with one tool round --------------------
    def _answer_batch(self, items: list[tuple[int, int, int]]) -> list[str]:
        """items = [(agent, family_idx, instance)] -> extracted answers, in order.
        Grouped by model so each group is one server-side batch."""
        from .. import llm_client
        out: list[str | None] = [None] * len(items)
        # Group by (model, max_tokens): the token budget IS the handicap, so a capped agent must
        # never ride along on an uncapped agent's batch -- that would both undo the handicap and
        # make the memo key (which encodes the budget) disagree with the generation behind it.
        groups: dict[tuple[str, int], list[int]] = {}
        for i, (a, f, _inst) in enumerate(items):
            model_i, _h, _t, mt_i = self._sig(a, f)
            groups.setdefault((model_i, mt_i), []).append(i)

        for (model, mt), idxs in groups.items():
            msgs, keys = [], []
            for i in idxs:
                a, f, inst = items[i]
                _m, hand, tool, _mt = self._sig(a, f)
                q = instance_entry(self.families[f], inst)["question"]
                msgs.append(build_prompt(self.families[f], q, hand, tool))
                keys.append(self._key(a, f, inst, "a"))
            texts = llm_client.complete_batch(model, msgs, keys, max_tokens=mt,
                                              concurrency=self.concurrency)
            with self._lock:
                self._counts["batches"] += 1

            # one tool round for whoever asked for it
            follow_idx, follow_msgs, follow_keys = [], [], []
            for j, i in enumerate(idxs):
                a, f, inst = items[i]
                _m, _hand, tool, _mt = self._sig(a, f)
                call = find_tool_call(texts[j], tool) if tool != "none" else None
                if call is None:
                    out[i] = extract_answer(texts[j])
                    continue
                result = run_tool(tool, call)
                with self._lock:
                    self._counts["tool_calls"] += 1
                follow_idx.append(i)
                follow_msgs.append(msgs[j] + [{"role": "assistant", "content": texts[j]},
                                              {"role": "user",
                                               "content": f"Tool result:\n{result}\n\n"
                                                          f"Now give the final answer inside "
                                                          f"<answer></answer> tags."}])
                follow_keys.append(self._key(a, f, inst, "b"))
            if follow_idx:
                ftexts = llm_client.complete_batch(model, follow_msgs, follow_keys, max_tokens=mt,
                                                   concurrency=self.concurrency)
                for j, i in enumerate(follow_idx):
                    out[i] = extract_answer(ftexts[j])
        with self._lock:
            self._counts["executions"] += len(items)
        return [x if x is not None else "" for x in out]

    # ---- backend protocol ------------------------------------------------
    def execute(self, a: int, task: Task) -> int:
        ans = self._answer_batch([(int(a), int(task.family), int(task.instance))])[0]
        return score(self.families[task.family], int(task.instance), ans)

    def execute_many(self, agents, families, reps: int, rng) -> np.ndarray:
        agents = np.asarray(agents).ravel().astype(int)
        families = np.asarray(families).ravel().astype(int)
        if families.size == 1 and agents.size > 1:
            families = np.repeat(families, agents.size)
        reps = int(reps)
        insts = rng.integers(0, 2**31 - 1, size=(agents.size, reps))
        items = [(int(agents[i]), int(families[i]), int(insts[i, r]))
                 for i in range(agents.size) for r in range(reps)]
        answers = self._answer_batch(items)
        out = np.empty((agents.size, reps), dtype=np.int8)
        for t, (a, f, inst) in enumerate(items):
            out[t // reps, t % reps] = score(self.families[f], inst, answers[t])
        return out

    def true_skill(self) -> np.ndarray:
        """Measured S[n,K]: `measure_probes` fresh instances per (agent, family), cached to disk.
        Instances are shared across agents (deterministic from the population seed) so that agents
        with the same prompt signature hit the same memo entries."""
        if self._S is not None:
            return self._S
        cache = self.dir / "S.npy"
        if cache.exists():
            S = np.load(cache)
            if S.shape == (self.n, self.K):
                self._S = S.astype(np.float32)
                self._load_descriptions()
                return self._S
        S = np.zeros((self.n, self.K), dtype=np.float32)
        chunk = max(1, int(os.environ.get("RTE_MEASURE_AGENT_CHUNK", "64")))
        for f in range(self.K):
            fam = self.families[f]
            # Measurement instances are shared across agents (deterministic from the population
            # seed), which is what lets agents with the same prompt signature share memo entries.
            reps_full = [int(stable_seed_32(self.seed, "measure", fam, r))
                         for r in range(self.measure_probes)]
            for lo in range(0, self.n, chunk):
                hi = min(self.n, lo + chunk)
                items, meta = [], []
                for a in range(lo, hi):
                    big = self.profiles[a]["model"] in LARGE_MODELS
                    r_a = self.measure_probes_large if big else self.measure_probes
                    items.extend((a, f, inst) for inst in reps_full[:r_a])
                    meta.append(r_a)
                answers = self._answer_batch(items)
                t = 0
                for j, a in enumerate(range(lo, hi)):
                    hits = sum(score(fam, items[t + u][2], answers[t + u]) for u in range(meta[j]))
                    t += meta[j]
                    S[a, f] = hits / meta[j]
            print(f"[llm-backend] measured family {f + 1}/{self.K} {fam}: "
                  f"mean={S[:, f].mean():.3f} max={S[:, f].max():.3f}", flush=True)
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(cache, S)
        (self.dir / "profiles.json").write_text(json.dumps(
            {"dist": self.dist, "n": self.n, "K": self.K, "seed": self.seed,
             "families": self.families, "profiles": self.profiles}, indent=2))
        self._S = S
        return S

    def declared(self, source: str = "programmatic") -> np.ndarray:
        if source == "programmatic":
            S = self.true_skill()
            rng = np.random.default_rng(stable_seed_32(self.seed, "declared"))
            return np.clip(S + rng.normal(0, self.declared_noise, S.shape), 0, 1).astype(np.float32)
        if source == "self_described":
            return self._self_described()
        raise ValueError(source)

    def _self_described(self) -> np.ndarray:
        cache = self.dir / "D_self_described.npy"
        if cache.exists():
            D = np.load(cache)
            if D.shape == (self.n, self.K):
                self._load_descriptions()
                return D.astype(np.float32)
        from .. import llm_client
        D = np.zeros((self.n, self.K), dtype=np.float32)
        for f in range(self.K):
            fam = self.families[f]
            q, _a = family_exemplar(fam)
            by_model: dict[str, list[int]] = {}
            for a in range(self.n):
                by_model.setdefault(self.profiles[a]["model"], []).append(a)
            for model, agents in by_model.items():
                msgs, keys = [], []
                for a in agents:
                    _m, hand, tool, _mt = self._sig(a, f)
                    tool_txt = {"calculator": "You have a calculator tool.",
                                "python": "You have a Python sandbox tool.",
                                "none": "You have no tools."}[tool]
                    msgs.append([
                        {"role": "system", "content":
                            "Rate your own competence at a class of problems. "
                            f"{tool_txt} Answer with a single number between 0 and 1 (0 = never "
                            "solve one, 1 = always solve one) inside <answer></answer> tags."},
                        {"role": "user", "content":
                            f"Task family: {fam}.\n{family_description(fam)}\n\n"
                            f"Here is one example problem:\n{q}\n\n"
                            "What fraction of problems like this do you solve correctly?"}])
                    keys.append(f"selfrate:{_sig_hash(self._sig(a, f))}:{f}")
                texts = llm_client.complete_batch(model, msgs, keys, max_tokens=64,
                                                  concurrency=self.concurrency)
                for a, txt in zip(agents, texts):
                    D[a, f] = _parse_rating(txt)
            print(f"[llm-backend] self-rated family {f + 1}/{self.K} {fam}: "
                  f"mean={D[:, f].mean():.3f}", flush=True)
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(cache, D)
        self.descriptions()
        return D

    def descriptions(self) -> list[str]:
        """One natural-language self-description per agent (for `llm_supervisor`)."""
        if self._descriptions is not None:
            return self._descriptions
        path = self.dir / "descriptions.json"
        if path.exists():
            self._descriptions = json.loads(path.read_text())
            return self._descriptions
        from .. import llm_client
        out = [""] * self.n
        by_model: dict[str, list[int]] = {}
        for a in range(self.n):
            by_model.setdefault(self.profiles[a]["model"], []).append(a)
        for model, agents in by_model.items():
            msgs, keys = [], []
            for a in agents:
                p = self.profiles[a]
                spec = ", ".join(self.families[f] for f in p["specialty"]) or "no particular area"
                msgs.append([
                    {"role": "system", "content":
                        "Write a single short paragraph (<=70 words) describing yourself as a "
                        "service in an agent marketplace: what kinds of problems you handle well "
                        "and what tools you have. No lists, no headings."},
                    {"role": "user", "content":
                        f"Your base model is {model}. Your tool is: {p['tool']}. You are trained "
                        f"and equipped for these task families: {spec}. Other families you handle "
                        f"without examples or tools. Describe yourself."}])
                keys.append(f"selfdesc:{stable_seed(model, p['tool'], tuple(p['specialty']), bits=96):024x}")
            texts = llm_client.complete_batch(model, msgs, keys, max_tokens=160,
                                              concurrency=self.concurrency)
            for a, txt in zip(agents, texts):
                p = self.profiles[a]
                spec = ", ".join(self.families[f] for f in p["specialty"]) or "general tasks"
                out[a] = f"Agent {a} ({model}, tool={p['tool']}). {' '.join((txt or '').split())} " \
                         f"Declared areas: {spec}."
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2))
        self._descriptions = out
        return out

    def _load_descriptions(self) -> None:
        path = self.dir / "descriptions.json"
        if self._descriptions is None and path.exists():
            self._descriptions = json.loads(path.read_text())

    def stats(self) -> dict:
        from .. import llm_client
        try:
            s = llm_client.stats()
        except Exception:                                    # noqa: BLE001
            s = {}
        return {"llm_executions": self._counts["executions"],
                "llm_tool_calls": self._counts["tool_calls"],
                "llm_batches": self._counts["batches"],
                "llm_cache_hit_rate": s.get("cache_hit_rate", float("nan")),
                "llm_generations": s.get("generations", 0)}


def _parse_rating(text: str) -> float:
    inner = ANSWER_RE.findall(text or "")
    hay = inner[-1] if inner else (text or "")
    m = RATING_RE.search(hay)
    if m:
        return float(np.clip(float(m.group(1)), 0.0, 1.0))
    m = re.search(r"(\d{1,3})\s*%", hay)
    if m:
        return float(np.clip(float(m.group(1)) / 100.0, 0.0, 1.0))
    return 0.5


# --------------------------------------------------------------------------- module accessor
# `llm_supervisor` needs the per-agent descriptions but only ever sees a `View`, which exposes
# no backend handle. The most recently constructed LLMBackend registers itself here; the method
# reads it lazily at build() time. Documented in the method file and in the final report.
_CURRENT: LLMBackend | None = None


def _set_current(b: LLMBackend) -> None:
    global _CURRENT
    _CURRENT = b


def current_backend() -> LLMBackend | None:
    return _CURRENT


def current_descriptions() -> list[str] | None:
    return None if _CURRENT is None else _CURRENT.descriptions()


# --------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    from ..world import skill_summary
    ap = argparse.ArgumentParser(prog="python -m rte.backends.llm")
    ap.add_argument("--dist", default="specialist")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--measure-probes", type=int, default=200)
    ap.add_argument("--measure-probes-large", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--skip-self-described", action="store_true")
    ap.add_argument("--measure", action="store_true",
                    help="accepted for symmetry; measuring IS what this entry point does")
    a = ap.parse_args(argv)

    b = LLMBackend(n=a.n, K=a.K, dist=a.dist, seed=a.seed, rng=np.random.default_rng(a.seed),
                   measure_probes=a.measure_probes, measure_probes_large=a.measure_probes_large,
                   concurrency=a.concurrency)
    S = b.true_skill()
    summary = skill_summary(S, a.measure_probes)
    np.save(b.dir / "D_programmatic.npy", b.declared("programmatic"))
    if not a.skip_self_described:
        b.declared("self_described")
        b.descriptions()
    (b.dir / "S_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n===== measured S =====")
    for k, v in summary.items():
        print(f"  {k:26s} {v:.4f}")
    gate = summary["skill_excess_ratio_family"]
    print(f"\ngate skill_excess_ratio_family >= 1.5 : {gate:.3f} -> "
          f"{'PASS' if gate >= 1.5 else 'FAIL'}")
    print(f"population dir: {b.dir}")
    print(f"backend stats: {b.stats()}")
    return 0 if gate >= 1.5 else 3


if __name__ == "__main__":
    sys.exit(main())
