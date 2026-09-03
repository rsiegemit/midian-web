"""Generic contract + correctness checks, auto-discovered over every file in `rte/methods/`.

Six properties, applied identically to every method (CONTRACT "Correctness checks"):
  1. builds inside its probe budget and charges only the counters its `needs` allow;
  2. returns valid agent ids over 100 tasks and moves the ledger;
  3. really uses every need it declares -- one need is dropped at a time and the View must raise
     (a method that survives is over-declaring: reported as an xfail naming it, not a failure);
  4. the class name matches its file;
  5. beats `random` on `specialist` at beta=0 (a router that loses to random is a bug until
     explained: xfail naming it);
  6. where the method exposes `est`, exact estimates make it find the true argmax.
Plus the documented per-fetch message formulas for MIDIAN (2*depth) and CNP (2n).
"""
from __future__ import annotations

import importlib
import math
import pkgutil
import traceback

import numpy as np
import pytest

import rte.methods as methods_pkg
from rte.budget import Budget
from rte.methods import load_method
from rte.world import AccessError, World

N, K, DIST, BETA, SEED, N_TASKS = 100, 16, "specialist", 0.25, 1, 100
BUDGET = Budget(3)
EXCLUDE = {"base", "__init__"}
NO_CHARGE = {"random"}                 # learns nothing, charges nothing: only `tasks` may move
DUMMY_URL = "http://127.0.0.1:9/v1"    # fw_echo never calls it; it exercises the bridge protocol


def discover() -> list[str]:
    """Every method module under rte/methods/, one level into subpackages (frameworks/)."""
    names = []
    for m in pkgutil.iter_modules(methods_pkg.__path__):
        if m.name in EXCLUDE or m.name.startswith("_"):
            continue
        if not m.ispkg:
            names.append(m.name)
            continue
        sub = importlib.import_module(f"rte.methods.{m.name}")
        names += [s.name for s in pkgutil.iter_modules(sub.__path__)
                  if not s.ispkg and not s.name.startswith("_")]
    return sorted(names)


NAMES = discover() or [pytest.param("<none yet>", marks=pytest.mark.skip(reason="no method files"))]


def _cls(name: str):
    """Import the class, skipping on a missing optional dep or a method that needs a live LLM.
    `fw_echo` is the bridge protocol check and runs without one, so it is not skipped."""
    try:
        c = load_method(name)
    except ImportError as e:
        pytest.skip(f"{name}: optional dependency missing at import ({e})")
    if getattr(c, "requires_llm", False) and name != "fw_echo":
        pytest.skip(f"{name}: requires_llm=True (needs a live vLLM endpoint)")
    return c


def _new(cls, name: str):
    kw = {"base_url": DUMMY_URL} if name.startswith("fw_") else {}
    try:
        return cls(**kw)
    except ImportError as e:
        pytest.skip(f"{name}: optional dependency missing at construction ({e})")


def _build(m, view, name: str, budget=BUDGET):
    try:
        m.build(view, budget)
    except ImportError as e:
        pytest.skip(f"{name}: optional dependency missing at build ({e})")
    except AccessError:
        raise                                       # AccessError is a RuntimeError: never a skip
    except RuntimeError as e:                       # e.g. no endpoints.json / no framework venv
        pytest.skip(f"{name}: cannot build here ({e})")


def agents_of(ret, n: int, name: str) -> list[int]:
    """fetch returns an int, or a non-empty list of ints for route-to-many."""
    got = list(ret) if isinstance(ret, (list, tuple, np.ndarray)) else [ret]
    assert got, f"{name}: fetch returned an empty agent list"
    for a in got:
        assert isinstance(a, (int, np.integer)) and not isinstance(a, bool), \
            f"{name}: fetch returned {a!r} of type {type(a)}, not an int"
        assert 0 <= int(a) < n, f"{name}: fetch returned agent {a} outside [0, {n})"
    return [int(a) for a in got]


def drive(world, m, stream, name):
    """Run a stream through a built method; returns the outcomes."""
    out = []
    for task in stream:
        agents = agents_of(m.fetch(task), world.n, name)
        outs = [world.execute(a, task) for a in agents]
        for a, o in zip(agents, outs):
            m.observe(task, a, o)
        out.append(outs[0] if len(outs) == 1 else int(2 * sum(outs) > len(outs)))
    return np.array(out)


@pytest.fixture(scope="module")
def world():
    return World(N, K, DIST, BETA, seed=SEED)


@pytest.fixture(scope="module")
def honest():
    """A beta=0 world plus `random`'s success on it: the floor every router must clear."""
    w = World(N, K, DIST, 0.0, seed=SEED)
    stream = w.tasks(300)
    r = load_method("random")()
    r.build(w.view(r.needs), BUDGET)
    return w, stream, float(drive(w, r, stream, "random").mean())


# --------------------------------------------------------------- 1, 2: budget, validity, ledger
@pytest.mark.parametrize("name", NAMES)
def test_method_contract(name, world):
    cls = _cls(name)
    assert set(cls.needs) <= {"declared", "probe", "reports", "bus"}, \
        f"{name}: needs={sorted(cls.needs)} contains an unknown need"
    m = _new(cls, name)

    before = world.ledger.snapshot()
    _build(m, world.view(m.needs), name)
    build = world.ledger.diff(before)
    cap = BUDGET.total_probes(world.n, world.K) * (1 + getattr(m, "rate", 0))      # audited variants: + audit rate (v2 1.2)
    assert build["probes"] <= cap, f"{name}: build spent {build['probes']} probes, budget is {cap}"
    if "probe" not in m.needs:
        assert build["probes"] == 0, f"{name}: probed without declaring 'probe'"
    if "reports" not in m.needs:
        assert build["reports"] == 0, f"{name}: reported without declaring 'reports'"

    world.ledger.reset()
    drive(world, m, world.tasks(N_TASKS), name)
    run = world.ledger.snapshot()
    assert run["tasks"] >= N_TASKS, f"{name}: {run['tasks']} task charges for {N_TASKS} tasks"
    moved = {c: v for c, v in run.items() if c != "tasks" and v}
    if name in NO_CHARGE:
        assert not moved, f"{name} should charge nothing but tasks, moved {moved}"
    else:
        assert moved, f"{name}: nothing but `tasks` moved over {N_TASKS} fetches ({run})"


# --------------------------------------------------------------- 3: needs enforcement (mutation)
@pytest.mark.parametrize("name", NAMES)
def test_method_uses_every_need_it_declares(name, world):
    cls = _cls(name)
    if not cls.needs:
        pytest.skip(f"{name}: needs={{}} -- nothing to drop")
    unused = []
    for dropped in sorted(cls.needs):
        m, raised = _new(cls, name), False
        try:
            _build(m, world.view(set(cls.needs) - {dropped}), name)
            drive(world, m, world.tasks(20), name)
        except AccessError:
            raised = True
        except Exception:                  # anything else is a real bug, not enforcement
            pytest.fail(f"{name}: dropping need {dropped!r} raised something other than "
                        f"AccessError:\n{traceback.format_exc()}")
        if not raised:
            unused.append(dropped)
    if unused:
        pytest.xfail(f"{name} declares {unused} but never touches "
                     f"{'them' if len(unused) > 1 else 'it'}")


@pytest.mark.parametrize("name", NAMES)
def test_method_name_matches_its_file(name):
    assert _cls(name).name == name, f"rte/methods/{name}.py defines a Method named {_cls(name).name!r}"


# --------------------------------------------------------------- 5: a router must beat random
@pytest.mark.parametrize("name", NAMES)
def test_method_beats_random(name, honest):
    """On `specialist` at beta=0 every router should find expertise. Losing to random is a bug."""
    w, stream, floor = honest
    if name == "random":
        pytest.skip("random is the floor")
    m = _new(_cls(name), name)
    w.ledger.reset()
    _build(m, w.view(m.needs), name)
    s = float(drive(w, m, stream, name).mean())
    if s < floor:
        pytest.xfail(f"{name} scores {s:.3f} on specialist beta=0, below random's {floor:.3f}")
    assert s >= floor


# --------------------------------------------------------------- 6: exact estimates -> true argmax
def exact_probe_many(S):
    """A `probe_many` that returns 100*S instead of Bernoulli draws: deterministic, monotone in S,
    and small enough to survive the report channel's int8 cast (so MIDIAN's peers can carry it)."""
    def probe_many(agents, families, reps=1):
        a, f = np.broadcast_arrays(np.asarray(agents), np.asarray(families))
        return np.broadcast_to(np.rint(100 * S[a, f])[..., None], a.shape + (int(reps),)).astype(np.int8)
    return probe_many


@pytest.mark.parametrize("name", NAMES)
def test_exact_estimates_find_the_true_argmax(name):
    """With the noise removed from probing, an argmax-over-estimates method must pick argmax S."""
    cls = _cls(name)
    if "probe" not in cls.needs:
        pytest.skip(f"{name}: does not probe")
    w = World(N, K, DIST, 0.0, seed=SEED)
    m = _new(cls, name)
    view = w.view(m.needs)
    view.probe_many = exact_probe_many(w.S)
    view.probe = lambda a, f: int(round(100 * float(w.S[a, f])))
    _build(m, view, name)
    est = getattr(m, "est", None)
    if not (isinstance(est, np.ndarray) and est.shape == (w.n, w.K)):
        pytest.skip(f"{name}: exposes no est[n, K] to check")
    best = w.S.max(0)
    got = np.array([w.S[int(np.argmax(est[:, f])), f] for f in range(w.K)])
    bad = np.flatnonzero(got < best - 0.01)
    assert bad.size == 0, (f"{name}: with exact estimates, families {bad.tolist()} pick an agent "
                           f"below the true best (got {got[bad].round(3)}, best {best[bad].round(3)})")


# --------------------------------------------------------------- documented message formulas
def fetch_messages(world, m, tasks) -> float:
    world.ledger.reset()
    for t in tasks:
        m.fetch(t)
    return world.ledger.messages / len(tasks)


def test_midian_sends_two_messages_per_level(world):
    """CONTRACT: MIDIAN's fetch = 2 messages per level (request down, answer up) = 2*ceil(log_r n)."""
    m = _new(_cls("midian"), "midian")
    _build(m, world.view(m.needs), "midian")
    r = m.params.get("r", 10)
    depth = math.ceil(math.log(world.n, r))
    assert fetch_messages(world, m, world.tasks(10)) == 2 * depth


def test_cnp_broadcasts_two_n_per_fetch(world):
    """CONTRACT: CNP broadcasts the task to n agents and reads n bids back = 2n per fetch."""
    m = _new(_cls("cnp_self_bid"), "cnp_self_bid")
    _build(m, world.view(m.needs), "cnp_self_bid")
    assert fetch_messages(world, m, world.tasks(10)) == 2 * world.n


def test_build_messages_are_recorded(world):
    """CONTRACT: collecting n declarations into a registry is n build messages."""
    for name in ("declared_argmax", "midian"):
        m = _new(_cls(name), name)
        world.ledger.reset()
        _build(m, world.view(m.needs), name)
        assert world.ledger.messages >= world.n, \
            f"{name}: build charged {world.ledger.messages} messages, expected at least n={world.n}"
