#!/usr/bin/env python3
"""Print OVERLAY_ARGS as a shell-quoted token list.

The workflow passes manifest.overlay_args as a JSON array. Each item is
expanded against this job's environment, so ${LLAMA_CI_MODEL} and
${CI_OUTPUT_DIR} resolve here and nowhere else.
"""
from __future__ import annotations

import json
import os
import shlex
import sys


def main() -> int:
    raw = os.environ.get('OVERLAY_ARGS', '').strip()
    if not raw or raw in ('null', '""'):
        return 0
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f'OVERLAY_ARGS is not valid JSON: {exc}', file=sys.stderr)
        return 1
    if items in (None, ''):
        return 0
    if not isinstance(items, list):
        print(f'OVERLAY_ARGS must be a JSON array, got {type(items).__name__}',
              file=sys.stderr)
        return 1
    tokens: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            print('OVERLAY_ARGS items must be non-empty strings', file=sys.stderr)
            return 1
        tokens.extend(shlex.split(os.path.expandvars(item), posix=True))
    print(' '.join(shlex.quote(token) for token in tokens))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
