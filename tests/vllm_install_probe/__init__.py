"""Tests package marker.

Single responsibility: inject the repo's ``tests/`` into ``sys.path``
so that ``from doc_test.base import ...`` can resolve.

Framework deps (mistune) are installed by the workflow, not at import time.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Layout: tests/vllm_install_probe/__init__.py -> parents[2] = repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TESTS_ROOT = _REPO_ROOT / 'tests'
for _p in (_TESTS_ROOT, _REPO_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)
