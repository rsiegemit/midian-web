"""The world: agents, tasks, true skill S, declared skill D, probes, reports.

One `World`, three backends (rte.backends.{bernoulli,replay,llm}). A backend
supplies: n, K, family names, true skill S[n,K], honest declared skill D,
task instances, and `execute(agent, task) -> 0/1`. The World layers on top:
liar selection, the lying model (`apply_lying`), the report channel, the
ledger, the paired task stream, and the access-controlled `View` that methods
see. Methods NEVER see S or the liar set.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .ledger import Ledger
from .stable_hash import stable_seed_32

NEEDS = frozenset({"declared", "probe", "reports", "bus"})
SKILL_DISTS = ("specialist", "heavy_tail", "bimodal", "correlated", "iid_uniform")
DELTA_INFLATE = 0.4
REPORT_LIE_TOP_FRAC = 0.2


@dataclass(frozen=True)
class Task:
    id: int
    family: int
    instance: int          # instance seed; the backend regenerates the concrete instance from (family, instance)


class AccessError(RuntimeError):
    pass


# --------------------------------------------------------------------------- skill distributions (bernoulli)
def sample_skill(dist: str, n: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """S[n,K] in [0,1] for the bernoulli backend. Same shapes the llm backend realizes via profiles."""
    from .backends._profiles import pick_k_per_agent, group_of
    if dist == "specialist":
        good = pick_k_per_agent(n, K, 3, rng)
        return np.where(good, rng.uniform(0.70, 0.95, size=(n, K)), rng.uniform(0.05, 0.30, size=(n, K)))
    if dist == "heavy_tail":
        return 0.05 + 0.90 * rng.beta(0.5, 3.0, size=(n, K))
    if dist == "bimodal":
        good = rng.random(n) < 0.2
        S = rng.uniform(0.20, 0.40, size=(n, K))
        S[good] = rng.uniform(0.75, 0.95, size=(int(good.sum()), K))
        return S
    if dist == "correlated":
        S = rng.uniform(0.15, 0.90, size=(n, 4))[:, group_of(K, 4)] + rng.normal(0, 0.05, size=(n, K))
        return np.clip(S, 0.0, 1.0)
    if dist == "iid_uniform":
        return rng.uniform(0.20, 0.90, size=(n, K))
    raise ValueError(f"unknown skill_dist {dist!r}; choose from {SKILL_DISTS}")


def skill_excess_ratio(S: np.ndarray, n_probes: int) -> float:
    """Between-agent variance of mean skill over the binomial noise floor at n_probes.
    Same definition as the old repo's premise gate (>= 1.5 passes)."""
    per_agent = np.asarray(S, float).mean(axis=1)
    p = per_agent.mean()
    var_obs = per_agent.var(ddof=1) if per_agent.size > 1 else 0.0
    var_noise = p * (1 - p) / max(int(n_probes), 1)
    return float(var_obs / var_noise) if var_noise > 0 else float("nan")


def skill_excess_ratio_family(S: np.ndarray, n_probes: int) -> float:
    """Median over families of the per-family excess ratio. `specialist` populations have a flat
    per-agent MEAN (everyone has 3 good families) so the agent-level ratio is structurally ~0 there;
    the per-family version is the one that says whether there is expertise to discover."""
    S = np.asarray(S, float)
    r = []
    for f in range(S.shape[1]):
        col = S[:, f]; p = col.mean(); vn = p * (1 - p) / max(int(n_probes), 1)
        r.append(col.var(ddof=1) / vn if vn > 0 else np.nan)
    return float(np.nanmedian(r))


def skill_summary(S: np.ndarray, n_probes: int = 200) -> dict:
    pa = S.mean(axis=1)
    return {"mean": float(S.mean()), "per_agent_std": float(pa.std()),
            "p90_p10": float(np.percentile(pa, 90) - np.percentile(pa, 10)),
            "family_max_mean": float(S.max(axis=0).mean()),
            "skill_excess_ratio": skill_excess_ratio(S, n_probes),
            "skill_excess_ratio_family": skill_excess_ratio_family(S, n_probes)}


# --------------------------------------------------------------------------- lying
def select_liars(S: np.ndarray, beta: float, how: str, rng: np.random.Generator) -> np.ndarray:
    n = S.shape[0]
    m = int(round(beta * n))
    liars = np.zeros(n, dtype=bool)
    if m == 0:
        return liars
    if how == "random":
        liars[rng.choice(n, size=m, replace=False)] = True
    elif how == "low_skill_first":
        liars[np.argsort(S.mean(axis=1), kind="stable")[:m]] = True
    else:
        raise ValueError(f"liar_select must be random|low_skill_first, got {how!r}")
    return liars


def apply_lying(D_honest: np.ndarray, liars: np.ndarray, mode: str = "inflate",
                demand: np.ndarray | None = None, delta: float = DELTA_INFLATE) -> np.ndarray:
    """Declared-channel lie. `inflate`: D[a] = clip(D_honest[a] + delta). `squat`: D[a, f*] = 1
    on the top-3 highest-demand families. Liars still execute at true skill."""
    D = D_honest.copy()
    if not liars.any():
        return D
    if mode == "inflate":
        D[liars] = np.clip(D[liars] + delta, 0.0, 1.0)
    elif mode == "squat":
        if demand is None:
            raise ValueError("squat needs the family demand vector")
        top = np.argsort(-demand, kind="stable")[:3]
        D[np.ix_(liars, top)] = 1.0
    else:
        raise ValueError(f"lie_mode must be inflate|squat, got {mode!r}")
    return D


# --------------------------------------------------------------------------- bus
class Bus:
    """Message counter for decentralized methods. Delivers nothing; it only charges."""
    def __init__(self, ledger: Ledger, n: int):
        self._ledger, self._n = ledger, n

    def send(self, src: int, dst: int, payload: Any = None) -> Any:
        self._ledger.message(1)
        return payload

    def send_many(self, k: int) -> None:
        self._ledger.message(int(k))

    def broadcast(self, src: int, payload: Any = None) -> Any:
        self._ledger.message(self._n)
        return payload


# --------------------------------------------------------------------------- view
class View:
    """What a method may touch. Anything outside `needs` raises AccessError.
    Always available: n, K, families, ledger, rng, dist (the *name* of the skill distribution)."""

    def __init__(self, world: "World", needs: Iterable[str], seed: int):
        needs = frozenset(needs)
        bad = needs - NEEDS
        if bad:
            raise ValueError(f"unknown needs {sorted(bad)}; allowed {sorted(NEEDS)}")
        self._w = world
        self.needs = needs
        self.n, self.K = world.n, world.K
        self.families = world.families
        self.ledger = world.ledger
        self.rng = np.random.default_rng(stable_seed_32(seed, "view", sorted(needs)))
        self._bus = Bus(world.ledger, world.n)

    def _require(self, what: str):
        if what not in self.needs:
            raise AccessError(f"method declared needs={sorted(self.needs)} but touched {what!r}")

    @property
    def declared(self) -> np.ndarray:
        self._require("declared")
        return self._w.D_view                     # read-only copy

    @property
    def bus(self) -> Bus:
        self._require("bus")
        return self._bus

    def probe(self, a: int, f: int) -> int:
        self._require("probe")
        return self._w.probe(int(a), int(f))

    def probe_many(self, agents, families, reps: int = 1) -> np.ndarray:
        """Vectorized probes: returns outcomes of shape (len(agents), reps). families broadcast with agents."""
        self._require("probe")
        return self._w.probe_many(np.asarray(agents), np.asarray(families), int(reps))

    def report_channel(self, j: int, a: int, outcome: int) -> int:
        """Peer j reports the outcome it observed for agent a. May be corrupted if j lies."""
        self._require("reports")
        return self._w.report(int(j), int(a), int(outcome))

    def report_many(self, reporters, agents, outcomes) -> np.ndarray:
        """Vectorized reports (one row per report). Liars' top-20% rule uses the means within this batch."""
        self._require("reports")
        return self._w.report_many(np.asarray(reporters), np.asarray(agents), np.asarray(outcomes))

    def __getattr__(self, item):            # any other attribute is a bug
        raise AccessError(f"View has no attribute {item!r} (S and liars are never exposed)")


# --------------------------------------------------------------------------- world
class World:
    def __init__(self, n: int, K: int, dist: str, beta: float, liar_select: str = "random",
                 collude: bool = True, seed: int = 0, backend: str = "bernoulli",
                 lie_mode: str = "inflate", declared_source: str = "programmatic",
                 demand: str = "uniform", backend_kwargs: dict | None = None):
        self.n, self.K, self.dist, self.beta = int(n), int(K), dist, float(beta)
        self.liar_select, self.collude, self.seed = liar_select, bool(collude), int(seed)
        self.lie_mode, self.declared_source, self.demand_kind = lie_mode, declared_source, demand
        self.backend_name = backend
        self.ledger = Ledger()
        self.rng = np.random.default_rng(stable_seed_32(seed, "world", n, K, dist, backend))

        from . import backends
        self.backend = backends.make(backend, n=self.n, K=self.K, dist=dist, seed=self.seed,
                                     rng=self.rng, **(backend_kwargs or {}))
        self.n, self.K = self.backend.n, self.backend.K        # a backend may round n (replay shards)
        self.families = list(self.backend.families)
        self.S = np.asarray(self.backend.true_skill(), dtype=np.float32)
        assert self.S.shape == (self.n, self.K), self.S.shape

        self.demand = self._demand_vector()
        self.liars = select_liars(self.S, self.beta, liar_select,
                                  np.random.default_rng(stable_seed_32(seed, "liars")))
        D_honest = np.asarray(self.backend.declared(declared_source), dtype=np.float32)
        self.D = apply_lying(D_honest, self.liars, lie_mode, self.demand)
        self.D_view = self.D.copy(); self.D_view.setflags(write=False)
        self._obs: dict[int, dict[int, list]] = {}    # reporter j -> {a: [sum, cnt]} (scalar report path)
        self._probe_rng = np.random.default_rng(stable_seed_32(seed, "probes"))

    # ---- demand / tasks
    def _demand_vector(self) -> np.ndarray:
        if self.demand_kind == "uniform":
            return np.full(self.K, 1.0 / self.K)
        if self.demand_kind == "skewed":                       # Zipf(1) over a fixed family order
            w = 1.0 / np.arange(1, self.K + 1)
            return w / w.sum()
        raise ValueError(self.demand_kind)

    def tasks(self, Q: int, stream_seed: int | None = None) -> list[Task]:
        """The paired task stream: identical for every method at a given (world seed, stream_seed)."""
        rng = np.random.default_rng(stable_seed_32(self.seed if stream_seed is None else stream_seed,
                                                   "stream", self.K, self.demand_kind))
        fams = rng.choice(self.K, size=int(Q), p=self.demand)
        return [Task(i, int(f), int(stable_seed_32(self.seed, "inst", i, int(f)))) for i, f in enumerate(fams)]

    # ---- execution
    def execute(self, a: int, task: Task) -> int:
        self.ledger.task(1)
        return int(self.backend.execute(int(a), task))

    def oracle(self, task: Task) -> int:
        return int(np.argmax(self.S[:, task.family]))

    def oracle_all(self) -> np.ndarray:
        return np.argmax(self.S, axis=0)

    # ---- probes: fresh instance each time, charged once each
    def probe(self, a: int, f: int) -> int:
        self.ledger.probe(1)
        inst = int(self._probe_rng.integers(0, 2**31 - 1))
        return int(self.backend.execute(a, Task(-1, f, inst)))

    def probe_many(self, agents: np.ndarray, families: np.ndarray, reps: int) -> np.ndarray:
        agents, families = np.broadcast_arrays(agents.astype(np.int64), families.astype(np.int64))
        self.ledger.probe(agents.size * reps)
        return self.backend.execute_many(agents, families, reps, self._probe_rng)

    # ---- reports: the only channel decentralized methods learn through
    def _lie_report(self, j: int, a: int, outcome: int, observed_mean_of: dict) -> int:
        """Report-channel lie for liar j about agent a. observed_mean_of: {agent: mean outcome j has seen}."""
        if self.liars[a]:
            return 1
        honest = [(m, -x) for x, m in observed_mean_of.items() if not self.liars[x]]
        if honest:
            k = max(1, math.ceil(REPORT_LIE_TOP_FRAC * len(honest)))
            honest.sort(reverse=True)
            top = {-x for _, x in honest[:k]}
            if a in top:
                return 0
        return int(outcome)

    def report(self, j: int, a: int, outcome: int) -> int:
        self.ledger.report(1)
        st = self._obs.setdefault(j, {})
        s = st.setdefault(a, [0, 0]); s[0] += outcome; s[1] += 1
        if not (self.collude and self.liars[j]):
            return int(outcome)
        return self._lie_report(j, a, outcome, {x: v[0] / v[1] for x, v in st.items()})

    def report_many(self, reporters: np.ndarray, agents: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
        reporters, agents, outcomes = np.broadcast_arrays(reporters, agents, outcomes)
        out = np.array(outcomes, dtype=np.float32 if outcomes.dtype.kind == "f" else np.int8, order="C")   # contiguous copy: the lie edits it in place
        self.ledger.report(out.size)
        if not self.collude or not self.liars.any():
            return out
        rj, ra, ro = reporters.ravel(), agents.ravel(), out.ravel()
        lying_rows = self.liars[rj]
        if not lying_rows.any():
            return out
        # rule 1: liar j vouches for liar a
        ro[lying_rows & self.liars[ra]] = 1
        # rule 2: liar j zeroes the top-20% honest agents by j's observed mean in this batch
        idx = np.flatnonzero(lying_rows & ~self.liars[ra])
        if idx.size:
            key = rj[idx].astype(np.int64) * self.n + ra[idx]
            uk, inv = np.unique(key, return_inverse=True)
            means = np.bincount(inv, weights=outcomes.ravel()[idx]) / np.bincount(inv)
            uj = uk // self.n
            order = np.lexsort((uk % self.n, -means, uj))            # by reporter, then mean desc, then agent id
            ujs = uj[order]
            start = np.searchsorted(ujs, ujs, "left"); size = np.searchsorted(ujs, ujs, "right") - start
            zero_sorted = (np.arange(ujs.size) - start) < np.ceil(REPORT_LIE_TOP_FRAC * size)   # first ceil(20%) per reporter
            zero_pairs = np.empty(uk.size, dtype=bool); zero_pairs[order] = zero_sorted
            ro[idx[zero_pairs[inv]]] = 0
        return out

    def reset(self, tag: str = "") -> None:
        """Start a method: zero the ledger, forget reporters' observations, reseed probes by `tag` so a method's
        build never depends on which methods ran before it (pairing is through the task stream + deterministic execute)."""
        self.ledger.reset(); self._obs.clear()
        self._probe_rng = np.random.default_rng(stable_seed_32(self.seed, "probes", tag))

    # ---- view
    def view(self, needs: Iterable[str], seed: int | None = None) -> View:
        return View(self, needs, self.seed if seed is None else seed)

    def stats(self) -> dict:
        d = {"n": self.n, "K": self.K, "dist": self.dist, "beta": self.beta, "backend": self.backend_name,
             "n_liars": int(self.liars.sum()), **skill_summary(self.S)}
        d.update(getattr(self.backend, "stats", lambda: {})())
        return d
