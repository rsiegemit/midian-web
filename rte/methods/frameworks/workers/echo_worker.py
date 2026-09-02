"""Reference worker: picks the first candidate. Validates the bridge protocol without any framework."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bridge import serve_worker   # noqa: E402  (workers import _bridge via PYTHONPATH=workers/.. or this insert)

if __name__ == "__main__":
    serve_worker(lambda req: req["candidates"][0]["name"])
