"""Primary backend: real vLLM-served models as agents, verified by a family adapter.

Population and prompts live next door (`population.py`, `prompts.py`); families and their
verifiers in `families.py`; every LLM call goes through `midian.llm_client`. This file is the
execution engine: batch, memoize, run one tool round, score, measure S.

MEMOIZATION — what makes the grid affordable. An answer is determined not by the agent id but by
its PROMPT SIGNATURE `(model, handicapped_on_f, tool_on_f, max_tokens)`, of which there are at
most |ladder| x 2 per family. Equal signatures emit an identical prompt and, at temperature 0, the
identical answer; the memo (midian.llm_client) is keyed by a hash of the full request, so they share one entry and
`true_skill()` costs `#signatures x K x measure_probes` generations, independent of n.

DE-CLONED POPULATIONS (`declone=V`, 2026-10-16). The plain population has at most 14 behaviours per family, so most
agents are exact clones. With declone each agent also holds one of V prompt variants (prompts.VARIANTS: persona,
reasoning style, sampling temperature), which joins its signature (at most |ladder| x 2 x V per family: distinct
skills), and every request an agent makes is SAMPLED at its variant's temperature with a seed of (population, agent,
request) (distinct answers: no two agents share a generation). Sampled texts are never memoised: one long answer per
(agent, family, instance) would grow the memo by tens of GB that every process loads at startup. Only the 0/1 outcome
matters, so it is kept in the population's outcome store (outcome_store.py). S is still measured per signature (same
signature, same expected skill), sampled with a seed of (signature, family, instance), so every de-cloned population
shares one measurement, kept in `<populations>/declone<V>_signature_S.json`. The population lives in its own directory
(`..._dc<V>`); a plain population is unchanged.
"""
from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .. import llm_client
from ..config import RTE_DATA, count, flag
from ..stable_hash import stable_seed_32
from . import families, noisy_declared, prompts, tools
from .outcome_store import OutcomeStore
from .population import bands, draw_profiles, ladder, signature

_OUTCOMES: dict = {}                           # RTE_OUTCOME_CACHE: (population, agent, family, instance) -> outcome
POP_DIR = Path(os.environ.get("RTE_POPULATIONS", RTE_DATA / "populations"))


class LLMBackend:
    def __init__(self, n: int, K: int = 16, dist: str = "specialist", seed: int = 0, rng=None,
                 families: list[str] | None = None, measure_probes: int = 200,
                 measure_probes_large: int = 60, max_tokens: int = 512,
                 population_dir: str | None = None, concurrency: int | None = None, declone: int = 0, **_):
        from . import families as fam  # the module; `families` is the kwarg here
        self.n, self.K, self.seed, self.dist = int(n), int(K), int(seed), dist
        self.cfg = ladder()
        self._large = bands(self.cfg)[3]
        self.families = list(families) if families else fam.names(self.K)
        assert len(self.families) == self.K, f"{len(self.families)} families for K={self.K}"
        self.measure_probes, self.measure_probes_large = int(measure_probes), int(measure_probes_large)
        self.max_tokens = int(max_tokens)
        # in-flight requests per batch; RTE_CONCURRENCY lets a dedicated warm-up job raise it (was a hard 64 here,
        # which silently bypassed the same knob on llm_client.complete_batch)
        self.concurrency = int(concurrency or llm_client.CONCURRENCY)
        self.declone = int(declone)
        assert self.declone <= len(prompts.VARIANTS), f"declone={declone} > {len(prompts.VARIANTS)} variants"
        self.profiles = draw_profiles(self.n, self.K, dist, self.seed, self.cfg, self.declone)
        self._set_dir(Path(population_dir) if population_dir else POP_DIR / (
            f"{dist}_n{self.n}_K{self.K}_seed{self.seed}" + (f"_dc{self.declone}" if self.declone else "")))
        self._S = self._desc = None
        self._dir0 = self.dir
        self._counts = {"executions": 0, "tool_calls": 0}
        self._lock = threading.Lock()
        global _CURRENT
        _CURRENT = self

    def _set_dir(self, d: Path) -> None:
        """The population directory, and beside it a de-cloned population's outcome store (outcomes, not texts)."""
        self.dir = d
        self._store = OutcomeStore(d) if self.declone else None

    def _sig(self, a: int, f: int) -> tuple:
        return signature(self.profiles[a], f, self.max_tokens)

    def _variant(self, a: int) -> dict | None:
        return prompts.VARIANTS[self.profiles[a]["variant"]] if self.declone else None

    def _sample(self, a: int, *what, measure: bool = False) -> tuple[float, int]:
        """(temperature, seed) of one request by agent a: greedy in a plain population; in a de-cloned one sampled at
        the agent's variant temperature with a seed of (population, agent, *what), or, for the S measurement
        (`measure`, what = (family, instance)), of (signature, *what): every de-cloned population shares it."""
        if not self.declone:
            return (0.0, 0)
        seed = stable_seed_32("declone_measure", *self._sig(a, what[0]), *what) if measure else \
            stable_seed_32("declone", self.dir.name, a, *what)
        return (self._variant(a)["temperature"], int(seed))

    def _sampling(self, samples) -> list | None:
        """complete_batch's `sampling`: each request's (temperature, seed) in a de-cloned population, None (greedy,
        memoised) in a plain one."""
        return list(samples) if self.declone else None

    # ---- churn: fresh profiles for `ids`; S / self-ratings / descriptions are recomputed lazily under a per-event
    # population dir (every signature is already measured, so only the new agents' descriptions cost generations)
    def snapshot(self):
        return ([dict(p) for p in self.profiles], self.dir, self._S, self._desc)
    def restore(self, snap):
        self.profiles, self.dir, self._S, self._desc = [dict(p) for p in snap[0]], snap[1], snap[2], snap[3]
    def redraw(self, ids, rng):
        tag = int(rng.integers(2 ** 31))
        new = draw_profiles(len(ids), self.K, self.dist, tag, self.cfg, self.declone)
        for i, prof in zip(ids, new):
            self.profiles[int(i)] = {**prof, "id": int(i)}
        self._set_dir(self._dir0 / f"churn_{tag}")
        self._S, self._desc = None, None

    def _ask(self, jobs: dict, prompt, max_tokens: int, sample=None) -> dict:
        """`jobs` maps a job key to the model that must answer it; returns {key: text}, batched
        per model. Self-rating keys on the SIGNATURE (the answer depends only on model, tool and
        family), self-description keys on the agent (it names the agent's own specialty list).
        `sample(key)` -> (temperature, seed) for the de-cloned populations (default greedy)."""
        by_model: dict[str, list] = {}
        for k, model in jobs.items():
            by_model.setdefault(model, []).append(k)
        out = {}
        for model, ks in by_model.items():
            for k, t in zip(ks, llm_client.complete_batch(
                    model, [prompt(k) for k in ks], None, max_tokens, self.concurrency,
                    self._sampling(sample(k) for k in ks))):
                out[k] = t
        return out

    def _answers(self, items: list[tuple[int, int, int]], measure: bool = False) -> list[str]:
        """items = [(agent, family_idx, instance)] -> answers, in order, batched per model. `measure`: the S
        measurement, sampled per SIGNATURE in a de-cloned population (`_sample`)."""
        out: list[str] = [""] * len(items)
        groups: dict[str, list[int]] = {}
        for i, (a, _, _) in enumerate(items):
            groups.setdefault(self.profiles[a]["model"], []).append(i)

        def run(group) -> None:
            model, idx = group
            msgs, samp = [], []
            for i in idx:
                a, f, inst = items[i]
                _m, hand, tool = self._sig(a, f)[:3]
                fam = self.families[f]
                msgs.append(prompts.build(fam, families.question(fam, inst), hand, tool, self._variant(a)))
                samp.append(self._sample(a, f, inst, measure=measure))
            # No caller-supplied keys anywhere: midian.llm_client hashes the full request, so two
            # agents with the same prompt still share one entry and a reworded prompt cannot
            # collide with a stale one.
            texts = llm_client.complete_batch(model, msgs, None, self.max_tokens, self.concurrency,
                                              self._sampling(samp))

            calls = []                                      # one tool round for whoever asked
            for j, i in enumerate(idx):
                a, f, _ = items[i]
                tool = self._sig(a, f)[2]
                call = prompts.find_tool_call(texts[j], tool)
                if call is None:
                    out[i] = prompts.extract_answer(texts[j])
                    continue
                calls.append((j, i, tool, call))
            with self._lock:
                self._counts["tool_calls"] += len(calls)
            # de-cloned: sampled code rarely repeats, so run the sandbox directly (no memo) and in parallel
            with ThreadPoolExecutor(max_workers=16 if self.declone else 1) as ex:
                results = list(ex.map(lambda c: tools.python(c[3], memo=not self.declone), calls))
            follow = [(i, prompts.follow_up(msgs[j], texts[j], r), samp[j]) for (j, i, _, _), r in zip(calls, results)]
            if follow:
                for (i, _, _), t in zip(follow, llm_client.complete_batch(
                        model, [m for _, m, _ in follow], None, self.max_tokens, self.concurrency,
                        self._sampling(x for _, _, x in follow))):
                    out[i] = prompts.extract_answer(t)

        # One thread per model group, each keeping its own client-side concurrency: the
        # groups hit DIFFERENT vLLM servers, so serialising them left most of the fleet idle and
        # capped a measurement sweep at the slowest model's rate. Writes are to distinct `out`
        # slots; only the shared counter needs the lock.
        if len(groups) == 1:
            run(next(iter(groups.items())))
        else:
            with ThreadPoolExecutor(max_workers=len(groups)) as ex:
                list(ex.map(run, list(groups.items())))
        with self._lock:
            self._counts["executions"] += len(items)
        return out

    def _outcomes(self, items, measure: bool = False) -> np.ndarray:
        if self._store is not None and not measure:     # de-cloned: the population's outcome store, misses added
            have = self._store.get(items)
            miss = list(dict.fromkeys(it for it, v in zip(items, have) if v is None))
            if miss:
                self._store.put(zip(miss, self._score(miss, self._answers(miss))))
                have = self._store.get(items)
            return np.array(have, dtype=np.int8)
        if flag("RTE_OUTCOME_CACHE"):          # opt-in, exact: an (agent, family, instance) outcome is fixed
            # (memoised answer, deterministic scoring), so score it once per process
            pop = str(self.dir)
            miss = list(dict.fromkeys(it for it in items if (pop,) + it not in _OUTCOMES))
            res = self._score(miss, self._answers(miss, measure) if miss else [])
            _OUTCOMES.update({(pop,) + it: r for it, r in zip(miss, res)})
            return np.array([_OUTCOMES[(pop,) + it] for it in items], dtype=np.int8)
        return np.array(self._score(items, self._answers(items, measure)), dtype=np.int8)

    def _score(self, items, answers) -> list[int]:
        """0/1 per (agent, family, instance) answer. Scoring rebuilds each instance: past 20k answers it forks
        RTE_TEXT_PROCS workers (pure function, same result)."""
        args = [(self.families[f], i, x) for (_, f, i), x in zip(items, answers)]
        n = count("RTE_TEXT_PROCS")
        if n > 1 and len(args) > 20000:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(n) as pool:
                return pool.starmap(families.correct, args, chunksize=5000)
        return [families.correct(*a) for a in args]

    # ---- backend protocol -------------------------------------------------
    def execute(self, a: int, task) -> int:
        return int(self._outcomes([(int(a), int(task.family), int(task.instance))])[0])

    def execute_many(self, agents, fam, inst) -> np.ndarray:
        agents, fam, inst = np.broadcast_arrays(np.asarray(agents), np.asarray(fam), np.asarray(inst))
        items = [(int(a), int(f), int(i)) for a, f, i in zip(agents.ravel(), fam.ravel(), inst.ravel())]
        return self._outcomes(items).reshape(inst.shape)

    def true_skill(self) -> np.ndarray:
        """Measured S[n,K], cached to disk. Both the generation and the scoring happen once per
        prompt SIGNATURE, from a project-wide probe set -- see the comments below."""
        if self._S is not None:
            return self._S
        cache = self.dir / "S.npy"
        if cache.exists() and (S := np.load(cache)).shape == (self.n, self.K):
            self._S = S.astype(np.float32)
            return self._S
        S = np.zeros((self.n, self.K), dtype=np.float32)
        # de-cloned: one measurement for every population beside this one
        shared = self.dir.parent / f"declone{self.declone}_signature_S.json"
        known = json.loads(shared.read_text()) if self.declone and shared.exists() else {}
        for f, fam in enumerate(self.families):
            # Instances are NOT seeded by the population: S is a property of the prompt SIGNATURE,
            # not of the population that happens to contain it. A fixed project-wide probe set
            # means the measurement generations are produced ONCE and every population, at every
            # grid seed and every n, is served from the memo.
            seeds = [int(stable_seed_32("measure", fam, r)) for r in range(self.measure_probes)]
            # ... and we GENERATE AND SCORE once per signature too, not once per agent. There are
            # at most 2 signatures per model on a given family (specialty / handicapped), so this
            # is ~14 cells whatever n is; scoring per agent would run 32M verifier calls at n=1e4
            # to learn those same 14 numbers.
            rep, S_sig = {}, {}
            for a in range(self.n):
                rep.setdefault(self._sig(a, f), a)
            key = lambda sig: f"{fam}|{json.dumps(list(sig))}"                          # noqa: E731
            S_sig.update({sig: known[key(sig)] for sig in rep if key(sig) in known})
            rep = {sig: a for sig, a in rep.items() if sig not in S_sig}
            n_probe = {sig: (self.measure_probes_large if sig[0] in self._large
                             else self.measure_probes) for sig in rep}
            items = [(a, f, s) for sig, a in rep.items() for s in seeds[:n_probe[sig]]]
            o, t = (self._outcomes(items, measure=True) if items else []), 0
            for sig, a in rep.items():
                S_sig[sig] = float(o[t:t + n_probe[sig]].mean())
                t += n_probe[sig]
            if self.declone and rep:                    # re-read first: another population may have added signatures
                known = {**(json.loads(shared.read_text()) if shared.exists() else {}),
                         **{key(sig): S_sig[sig] for sig in rep}}
                tmp = shared.with_suffix(f".tmp{os.getpid()}")
                tmp.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(json.dumps(known))
                os.replace(tmp, shared)
            for a in range(self.n):
                S[a, f] = S_sig[self._sig(a, f)]
            print(f"[llm] family {f+1}/{self.K} {fam}: {len(S_sig)} signatures ({len(rep)} measured now), "
                  f"mean={S[:, f].mean():.3f} max={S[:, f].max():.3f}", flush=True)
        self._write("S.npy", S)
        (self.dir / "profiles.json").write_text(json.dumps(
            {"dist": self.dist, "n": self.n, "K": self.K, "seed": self.seed,
             "families": self.families, "profiles": self.profiles}, indent=2))
        self._S = S
        return S

    def _write(self, name: str, arr: np.ndarray) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(self.dir / name, arr)

    def declared(self, source: str = "programmatic") -> np.ndarray:
        if source == "programmatic":
            return noisy_declared(self.true_skill(), self.seed)
        if source != "self_described":
            raise ValueError(source)
        path = self.dir / "D_self_described.npy"
        if path.exists() and (D := np.load(path)).shape == (self.n, self.K):
            return D.astype(np.float32)
        D = np.zeros((self.n, self.K), dtype=np.float32)
        for f, fam in enumerate(self.families):             # the agent's OWN model rates itself
            q, _ = families.exemplar(fam)
            if self.declone:                                # per agent: its own variant, its own sample
                text = self._ask({a: self.profiles[a]["model"] for a in range(self.n)},
                                 lambda a, fam=fam, q=q: prompts.rate_self(fam, self._sig(a, f)[2], q, self._variant(a)),
                                 64, lambda a, f=f: self._sample(a, f, "rate_self"))
                D[:, f] = [prompts.parse_rating(text[a]) for a in range(self.n)]
                print(f"[llm] self-rated {f+1}/{self.K} {fam}: per agent, mean={D[:, f].mean():.3f}", flush=True)
                continue
            sigs = {self._sig(a, f) for a in range(self.n)}
            text = self._ask({sig: sig[0] for sig in sigs},
                             lambda sig, fam=fam, q=q: prompts.rate_self(fam, sig[2], q), 64)
            rate = {sig: prompts.parse_rating(t) for sig, t in text.items()}
            D[:, f] = [rate[self._sig(a, f)] for a in range(self.n)]
            print(f"[llm] self-rated {f+1}/{self.K} {fam}: {len(sigs)} signatures, "
                  f"mean={D[:, f].mean():.3f}", flush=True)
        self._write("D_self_described.npy", D)
        return D

    # ---- text accessors for llm_supervisor and the framework rivals --------
    def descriptions(self) -> list[str]:
        """One self-description per agent, written by the agent's own model. Produced whatever
        `declared_source` is; cached in descriptions.json."""
        if self._desc is not None:
            return self._desc
        path = self.dir / "descriptions.json"
        if path.exists():
            self._desc = json.loads(path.read_text())
            return self._desc
        spec = lambda a: ", ".join(self.families[f] for f in self.profiles[a]["specialty"])  # noqa: E731
        # Per AGENT, not per signature: the prompt names this agent's own specialty list, which two
        # agents sharing a signature need not share. Identical prompts still cost one generation.
        texts = self._ask({a: self.profiles[a]["model"] for a in range(self.n)},
                          lambda a: prompts.describe_self(self.profiles[a]["model"],
                                                          self.profiles[a]["tool"], spec(a), self._variant(a)), 160,
                          lambda a: self._sample(a, "describe"))
        self._desc = [f"{' '.join(texts[a].split())} Declared areas: {spec(a) or 'general tasks'}."
                      for a in range(self.n)]
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._desc, indent=2))
        return self._desc

    def confidence(self, agents, f: int, inst: int, again: bool = False) -> list[str]:
        """Each agent's raw verbal confidence on instance `inst` of family f, asked by its own model in its own solve
        prompt (prompts.rate_task); `again` continues that conversation once (prompts.rate_again). Plain population:
        memoised by content hash, so agents sharing a signature share one generation and the first reply of `again` is
        a memo hit; de-cloned: sampled per agent (seed of agent, family, instance), never memoised."""
        agents = [int(a) for a in agents]
        fam, sig = self.families[f], [self._sig(a, f) for a in agents]
        q = families.question(fam, int(inst))
        first = lambda i: prompts.rate_task(fam, q, sig[i][1], sig[i][2], self._variant(agents[i]))  # noqa: E731
        jobs = {i: s[0] for i, s in enumerate(sig)}
        conf = lambda i: self._sample(agents[i], f, int(inst), "confidence")                       # noqa: E731
        text = self._ask(jobs, first, 32, conf)
        if again:
            text = self._ask(jobs, lambda i: prompts.rate_again(first(i), text[i]), 32, conf)
        return [text[i] for i in range(len(sig))]

    def family_descriptions(self) -> list[str]:
        return [families.describe(f) for f in self.families]

    def task_text(self, task) -> str:
        return self.text(task.family, task.instance)

    def text(self, f: int, inst: int, probe: bool = False) -> str:
        return families.question(self.families[int(f)], int(inst))

    def stats(self) -> dict:
        s = llm_client.stats()
        return {"llm_executions": self._counts["executions"],
                "llm_tool_calls": self._counts["tool_calls"],
                "llm_cache_hit_rate": s["cache_hit_rate"], "llm_generations": s["generations"]}


_CURRENT: LLMBackend | None = None


def current_backend() -> LLMBackend | None:
    """The most recently constructed LLMBackend, or None. A Method only ever sees a View, which
    exposes no backend handle; this is how llm_supervisor and the framework rivals reach the agent
    descriptions, the family descriptions and the concrete task text."""
    return _CURRENT
