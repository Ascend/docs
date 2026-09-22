"""Tests package marker.

Single responsibility: inject the repo's ``tests/`` into ``sys.path``
so that ``from doc_test.base import ...`` can resolve.

Framework deps (mistune) are installed by the common quick-start workflow
template, not at import time here.

Why it lives here:
* unittest treats ``tests/`` as a package; the parent ``__init__.py``
  executes before any submodule import.
* It runs before ``tests/test_*.py`` import, which is the earliest
  opportunity to inject ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# sys.path bootstrap: make ``doc_test.*`` resolvable.
# Layout: tests/peft/__init__.py -> parents[0]=tests/peft,
# parents[1]=tests, parents[2]=repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TESTS_ROOT = _REPO_ROOT / 'tests'
for _p in (_TESTS_ROOT, _REPO_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)
