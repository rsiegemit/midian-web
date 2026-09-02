"""Generic contract test, auto-discovered over every file in `rte/methods/`.

For each method: it builds inside its probe budget, returns valid agents on 100 tasks,
charges the ledger, and -- the point of the exercise -- never touches anything outside the
`needs` it declares. The mutation test runs each method against a View with one declared
need removed and requires AccessError: a method that survives that is declaring a need it
never uses, which is reported as an xfail naming the method.
"""
from __future__ import annotations

import importlib
import pkgutil
import traceback

import numpy as np
import pytest

import rte.methods as methods_pkg
from rte.budget import Budget
from rte.methods import load_method
from rte.world import AccessError, World

N, K, DIST, BETA, SEED = 100, 16, "specialist", 0.25, 1
N_TASKS = 100
BUDGET = Budget(3)
EXCLUDE = {"base", "__init__"}
# `random` learns nothing and charges nothing: the only counter its run may move is `tasks`.
NO_CHARGE_METHODS = {"random"}


def discover() -> list[str]:
    """Every method module under rte/methods/, one level into subpackages (e.g. frameworks/)."""
    names = []
    for m in pkgutil.iter_modules(methods_pkg.__path__):
        if m.name in EXCLUDE or m.name.startswith("_"):
            continue
        if not m.ispkg:
            names.append(m.name)
            continue
        sub = importlib.import_module(f"rte.methods.{m.name}")
        names += [f"{m.name}.{s.name}" for s in pkgutil.iter_modules(sub.__path__)
                  if not s.ispkg and not s.name.startswith("_")]
    return sorted(names)


METHOD_NAMES = discover()
if not METHOD_NAMES:
    METHOD_NAMES = [pytest.param("<no methods present yet>", marks=pytest.mark.skip(
        reason="rte/methods/ contains no method files yet"))]


def _load(name: str):
    """Import the method class, skipping on a missing optional dependency or an LLM-only method."""
    try:
        cls = load_method(name)
    except ImportError as e:
        pytest.skip(f"{name}: optional dependency missing at import ({e})")
    if getattr(cls, "requires_llm", False):
        pytest.skip(f"{name}: requires_llm=True (needs a live vLLM endpoint)")
    if getattr(cls, "runner_only", False):
        pytest.skip(f"{name}: runner_only=True (reported as the ceiling line, not a method)")
    return cls


def _instantiate(cls, name: str):
    try:
        return cls()
    except ImportError as e:
        pytest.skip(f"{name}: optional dependency missing at construction ({e})")


def _build(m, view, name: str):
    try:
        m.build(view, BUDGET)
    except ImportError as e:
        pytest.skip(f"{name}: optional dependency missing at build ({e})")


def _agents_of(ret, n: int, name: str, where: str) -> list[int]:
    """fetch may return an int OR a non-empty list of ints (route-to-many)."""
    if isinstance(ret, (list, tuple, np.ndarray)):
        got = list(ret)
        assert got, f"{name}: {where} returned an empty agent list"
        for a in got:
            assert isinstance(a, (int, np.integer)) and not isinstance(a, bool), \
                f"{name}: {where} returned {a!r} of type {type(a)}, not an int"
            assert 0 <= int(a) < n, f"{name}: {where} returned agent {a} outside [0, {n})"
        return [int(a) for a in got]
    assert isinstance(ret, (int, np.integer)) and not isinstance(ret, bool), \
        f"{name}: {where} returned {ret!r} of type {type(ret)}, not an int or list of ints"
    assert 0 <= int(ret) < n, f"{name}: {where} returned agent {ret} outside [0, {n})"
    return [int(ret)]


@pytest.fixture(scope="module")
def world():
    return World(N, K, DIST, BETA, seed=SEED)


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_method_contract(name, world):
    """Build within budget, 100 valid fetches, ledger charged, no undeclared access."""
    cls = _load(name)
    assert set(cls.needs) <= {"declared", "probe", "reports", "bus"}, \
        f"{name}: needs={sorted(cls.needs)} contains an unknown need"
    m = _instantiate(cls, name)

    view = world.view(m.needs)
    before = world.ledger.snapshot()
    _build(m, view, name)
    build_cost = world.ledger.diff(before)

    cap = BUDGET.total_probes(world.n, world.K)
    assert build_cost["probes"] <= cap, \
        f"{name}: build spent {build_cost['probes']} probes, budget is {cap}"
    if "probe" not in m.needs:
        assert build_cost["probes"] == 0, f"{name}: probed without declaring 'probe'"
    if "reports" not in m.needs:
        assert build_cost["reports"] == 0, f"{name}: reported without declaring 'reports'"

    stream = world.tasks(N_TASKS)
    world.ledger.reset()
    run_before = world.ledger.snapshot()
    for task in stream:
        agents = _agents_of(m.fetch(task), world.n, name, "fetch")
        outcomes = [world.execute(a, task) for a in agents]
        for o in outcomes:
            assert o in (0, 1), f"{name}: execute returned {o!r}"
        if len(agents) == 1:
            m.observe(task, agents[0], outcomes[0])
        else:
            for a, o in zip(agents, outcomes):
                m.observe(task, a, o)
    run_cost = world.ledger.diff(run_before)

    assert run_cost["tasks"] >= N_TASKS, f"{name}: {run_cost['tasks']} task charges for {N_TASKS} tasks"
    moved = {c: v for c, v in run_cost.items() if c != "tasks" and v != 0}
    if name in NO_CHARGE_METHODS:
        assert not moved, f"{name} should charge nothing but tasks, moved {moved}"
    else:
        assert moved, f"{name}: nothing but `tasks` moved over {N_TASKS} fetches ({run_cost})"


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_method_uses_every_need_it_declares(name, world):
    """Mutation test: drop one declared need at a time; the View must raise.

    A method that runs fine without a need it declared is over-declaring -- reported as
    an xfail naming the method, not a hard failure, so the suite still says who they are.
    """
    cls = _load(name)
    if not cls.needs:
        pytest.skip(f"{name}: needs={{}} -- nothing to drop")
    unused = []
    for dropped in sorted(cls.needs):
        m = _instantiate(cls, name)
        view = world.view(set(cls.needs) - {dropped})
        raised = False
        try:
            m.build(view, BUDGET)
            for task in world.tasks(20):
                agents = _agents_of(m.fetch(task), world.n, name, "fetch")
                for a in agents:
                    m.observe(task, a, world.execute(a, task))
        except AccessError:
            raised = True
        except ImportError as e:
            pytest.skip(f"{name}: optional dependency missing ({e})")
        except Exception:                       # any other blow-up is a real bug, not enforcement
            pytest.fail(f"{name}: dropping need {dropped!r} raised something other than "
                        f"AccessError:\n{traceback.format_exc()}")
        if not raised:
            unused.append(dropped)
    if unused:
        pytest.xfail(f"{name} declares {unused} but never touches {'them' if len(unused) > 1 else 'it'}")


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_method_name_matches_its_file(name):
    cls = _load(name)
    assert cls.name == name.rsplit(".", 1)[-1], \
        f"rte/methods/{name.replace('.', '/')}.py defines a Method named {cls.name!r}"
