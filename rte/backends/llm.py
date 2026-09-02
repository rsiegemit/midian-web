"""Primary backend: real vLLM-served models as agents, verified by a family adapter.

Population and prompts live next door (`population.py`, `prompts.py`); families and their
verifiers in `families.py`; every LLM call goes through `rte.llm_client`. This file is the
execution engine: batch, memoize, run one tool round, score, measure S.

MEMOIZATION — what makes the grid affordable. An answer is determined not by the agent id but by
its PROMPT SIGNATURE `(model, handicapped_on_f, tool_on_f, max_tokens)`, of which there are at
most |ladder| x 2 x 3. Equal signatures emit an identical prompt and, at temperature 0, the
identical answer, so the memo is keyed on the signature: `true_skill()` costs
`#signatures x K x measure_probes` generations, independent of n.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..stable_hash import stable_seed_32
from . import families, prompts, tools
from .population import bands, draw_profiles, ladder, signature
from .prompts import build as build_prompt, extract_answer, find_tool_call, parse_rating  # re-export

RTE_DATA = Path(os.environ.get("RTE_DATA", "/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte"))
POP_DIR = Path(os.environ.get("RTE_POPULATIONS", RTE_DATA / "populations"))


class LLMBackend:
    def __init__(self, n: int, K: int = 16, dist: str = "specialist", seed: int = 0, rng=None,
                 families: list[str] | None = None, measure_probes: int = 200,
                 measure_probes_large: int = 60, max_tokens: int = 512,
                 handicap_max_tokens: int | None = None, declared_noise: float = 0.05,
                 population_dir: str | None = None, concurrency: int = 64, **_):
        from . import families as fam                  # the module; `families` is the kwarg here
        self.n, self.K, self.seed, self.dist = int(n), int(K), int(seed), dist
        self.cfg = ladder()
        self._large = bands(self.cfg)[3]
        self.families = list(families) if families else fam.names(self.K)
        assert len(self.families) == self.K, f"{len(self.families)} families for K={self.K}"
        self.measure_probes, self.measure_probes_large = int(measure_probes), int(measure_probes_large)
        self.max_tokens = int(max_tokens)
        self.handicap_max_tokens = self.max_tokens if handicap_max_tokens is None \
            else int(handicap_max_tokens)
        self.declared_noise, self.concurrency = float(declared_noise), int(concurrency)
        self.profiles = draw_profiles(self.n, self.K, dist, self.seed, self.cfg)
        self.dir = Path(population_dir) if population_dir else \
            POP_DIR / f"{dist}_n{self.n}_K{self.K}_seed{self.seed}"
        self._S = self._desc = None
        self._counts = {"executions": 0, "tool_calls": 0}
        self._lock = threading.Lock()
        global _CURRENT
        _CURRENT = self

    def _sig(self, a: int, f: int) -> tuple:
        return signature(self.profiles[a], f, self.max_tokens, self.handicap_max_tokens)

    def _by_model(self) -> dict[str, list[int]]:
        g: dict[str, list[int]] = {}
        for a, p in enumerate(self.profiles):
            g.setdefault(p["model"], []).append(a)
        return g

    def _ask(self, prompt, max_tokens: int) -> list[str]:
        """Ask every agent's OWN model one question, batched per model. The self-rating and
        self-description passes differ only in their prompt, so they share this."""
        from .. import llm_client
        out = [""] * self.n
        for model, agents in self._by_model().items():
            for a, t in zip(agents, llm_client.complete_batch(
                    model, [prompt(a) for a in agents], None, max_tokens, self.concurrency)):
                out[a] = t
        return out

    def _answers(self, items: list[tuple[int, int, int]]) -> list[str]:
        """items = [(agent, family_idx, instance)] -> answers, in order. Grouped by
        (model, max_tokens): the budget is part of the memo key, so a capped agent must never
        ride on an uncapped agent's batch."""
        from .. import llm_client
        out: list[str] = [""] * len(items)
        groups: dict[tuple, list[int]] = {}
        for i, (a, f, _) in enumerate(items):
            model, _h, _t, mt = self._sig(a, f)
            groups.setdefault((model, mt), []).append(i)

        def run(group) -> None:
            (model, mt), idx = group
            msgs = []
            for i in idx:
                a, f, _ = items[i]
                _m, hand, tool, _ = self._sig(a, f)
                fam = self.families[f]
                msgs.append(prompts.build(fam, families.question(fam, items[i][2]), hand, tool))
            # No caller-supplied keys anywhere: rte.llm_client hashes the full request, so two
            # agents with the same prompt still share one entry and a reworded prompt cannot
            # collide with a stale one.
            texts = llm_client.complete_batch(model, msgs, None, mt, self.concurrency)

            follow = []                                     # one tool round for whoever asked
            for j, i in enumerate(idx):
                a, f, _ = items[i]
                tool = self._sig(a, f)[2]
                call = prompts.find_tool_call(texts[j], tool)
                if call is None:
                    out[i] = prompts.extract_answer(texts[j])
                    continue
                with self._lock:
                    self._counts["tool_calls"] += 1
                follow.append((i, prompts.follow_up(msgs[j], texts[j], tools.RUN[tool](call))))
            if follow:
                for (i, _), t in zip(follow, llm_client.complete_batch(
                        model, [m for _, m in follow], None, mt, self.concurrency)):
                    out[i] = prompts.extract_answer(t)

        # One thread per (model, budget) group, each keeping its own client-side concurrency: the
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

    def _outcomes(self, items) -> np.ndarray:
        return np.array([families.correct(self.families[f], i, x)
                         for (_, f, i), x in zip(items, self._answers(items))], dtype=np.int8)

    # ---- backend protocol -------------------------------------------------
    def execute(self, a: int, task) -> int:
        return int(self._outcomes([(int(a), int(task.family), int(task.instance))])[0])

    def execute_many(self, agents, fam, reps: int, rng) -> np.ndarray:
        agents, fam, reps = np.ravel(agents).astype(int), np.ravel(fam).astype(int), int(reps)
        fam = np.repeat(fam, agents.size) if fam.size == 1 < agents.size else fam
        inst = rng.integers(0, 2**31 - 1, size=(agents.size, reps))
        items = [(int(agents[i]), int(fam[i]), int(inst[i, r]))
                 for i in range(agents.size) for r in range(reps)]
        return self._outcomes(items).reshape(agents.size, reps)

    def true_skill(self) -> np.ndarray:
        """Measured S[n,K], cached to disk. Measurement instances are shared across agents
        (deterministic from the population seed) so equal signatures share memo entries."""
        if self._S is not None:
            return self._S
        cache = self.dir / "S.npy"
        if cache.exists() and (S := np.load(cache)).shape == (self.n, self.K):
            self._S = S.astype(np.float32)
            return self._S
        S = np.zeros((self.n, self.K), dtype=np.float32)
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
            rep = {}
            for a in range(self.n):
                rep.setdefault(self._sig(a, f), a)
            n_probe = {sig: (self.measure_probes_large if sig[0] in self._large
                             else self.measure_probes) for sig in rep}
            o, t, S_sig = self._outcomes([(a, f, s) for sig, a in rep.items()
                                          for s in seeds[:n_probe[sig]]]), 0, {}
            for sig, a in rep.items():
                S_sig[sig] = o[t:t + n_probe[sig]].mean()
                t += n_probe[sig]
            for a in range(self.n):
                S[a, f] = S_sig[self._sig(a, f)]
            print(f"[llm] family {f+1}/{self.K} {fam}: {len(rep)} signatures, "
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
            S = self.true_skill()
            rng = np.random.default_rng(stable_seed_32(self.seed, "declared"))
            return np.clip(S + rng.normal(0, self.declared_noise, S.shape), 0, 1).astype(np.float32)
        if source != "self_described":
            raise ValueError(source)
        path = self.dir / "D_self_described.npy"
        if path.exists() and (D := np.load(path)).shape == (self.n, self.K):
            return D.astype(np.float32)
        D = np.zeros((self.n, self.K), dtype=np.float32)
        for f, fam in enumerate(self.families):             # the agent's OWN model rates itself
            q, _ = families.exemplar(fam)
            D[:, f] = [prompts.parse_rating(t) for t in self._ask(
                lambda a, f=f, fam=fam, q=q: prompts.rate_self(fam, self._sig(a, f)[2], q), 64)]
            print(f"[llm] self-rated {f+1}/{self.K} {fam}: mean={D[:, f].mean():.3f}", flush=True)
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
        texts = self._ask(
            lambda a: prompts.describe_self(self.profiles[a]["model"], self.profiles[a]["tool"],
                                            spec(a)), 160)
        self._desc = [f"{' '.join(t.split())} Declared areas: {spec(a) or 'general tasks'}."
                      for a, t in enumerate(texts)]
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._desc, indent=2))
        return self._desc

    def family_descriptions(self) -> list[str]:
        return [families.describe(f) for f in self.families]

    def task_text(self, task) -> str:
        return families.question(self.families[int(task.family)], int(task.instance))

    def stats(self) -> dict:
        from .. import llm_client
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


if __name__ == "__main__":                    # python -m rte.backends.llm --measure --dist ...
    from ..measure import main
    sys.exit(main())
