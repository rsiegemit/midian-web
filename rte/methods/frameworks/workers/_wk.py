"""Boilerplate shared by every framework worker. Pure: no imports of any framework, no side effects."""
import asyncio
import re

_LOOP = asyncio.new_event_loop()
_UNSAFE = re.compile(r"[^A-Za-z0-9_]")


def openai_kwargs(req):
    """`model` / `base_url` / `api_key` as every framework's OpenAI-compatible client takes them."""
    return {"model": req["model"], "base_url": req["base_url"], "api_key": req["api_key"] or "EMPTY"}


def run_async(coro):
    """One event loop for the life of the worker process."""
    return _LOOP.run_until_complete(coro)


def sanitize(names):
    """(framework-safe names, safe -> original) for frameworks that restrict identifier characters."""
    safe = [_UNSAFE.sub("_", n) for n in names]
    return safe, dict(zip(safe, names))
