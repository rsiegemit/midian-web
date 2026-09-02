"""Method discovery: `load_method("midian")` imports `rte.methods.midian` and
returns its single Method subclass. Each method file is self-contained."""
from __future__ import annotations
import importlib
import inspect
from .base import Method


def load_method(name: str) -> type[Method]:
    try:
        mod = importlib.import_module(f"rte.methods.{name}")
    except ModuleNotFoundError as e:
        if e.name != f"rte.methods.{name}":
            raise
        mod = importlib.import_module(f"rte.methods.frameworks.{name}")   # fw_* rivals live in the subpackage
    cands = [c for _, c in inspect.getmembers(mod, inspect.isclass)
             if issubclass(c, Method) and c is not Method and c.__module__ == mod.__name__]
    if len(cands) != 1:
        raise ImportError(f"rte.methods.{name} must define exactly one Method subclass, found {cands}")
    return cands[0]
