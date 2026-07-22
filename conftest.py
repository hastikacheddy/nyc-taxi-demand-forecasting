"""Root pytest conftest.

Ensures the repository root is on sys.path so top-level packages that are used
only from tests (e.g. `benchmarks`) import cleanly under a bare `pytest`
invocation in CI. CI installs the library with `pip install -e .`, which only
packages `src*` — `benchmarks/` is a dev/bench tool, not part of the shipped
distribution, so it is made importable here rather than installed.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
