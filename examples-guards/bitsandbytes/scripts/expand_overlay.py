#!/usr/bin/env python3
"""Print OVERLAY_ARGS as a shell-quoted token list for run_example.sh."""
from __future__ import annotations

import json
import os
import shlex
import sys


def main() -> int:
    raw = os.environ.get('OVERLAY_ARGS', '').strip()
    if not raw or raw in ('null', '""'):
        return 0
    items = json.loads(raw)
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        print('OVERLAY_ARGS must be a JSON array of strings', file=sys.stderr)
        return 1
    tokens: list[str] = []
    for item in items:
        tokens.extend(shlex.split(os.path.expandvars(item), posix=True))
    print(' '.join(shlex.quote(token) for token in tokens))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
