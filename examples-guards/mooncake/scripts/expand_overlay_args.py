#!/usr/bin/env python3
"""Print OVERLAY_ARGS as one argv token per line.

The manifest stores overlay_args as a JSON array of CLI strings. The
engine forwards that array in OVERLAY_ARGS. Empty, null, and "" mean
no extra args. Each string is split with POSIX shell rules so a quoted
token stays one argv element.
"""

from __future__ import annotations

import json
import os
import shlex


def main() -> None:
    raw = os.environ.get('OVERLAY_ARGS', '').strip()
    if not raw or raw in ('null', '""'):
        return
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f'OVERLAY_ARGS is not valid JSON: {exc}') from exc
    if not isinstance(items, list):
        raise SystemExit(
            f'OVERLAY_ARGS must be a JSON array, got {type(items).__name__}')
    for item in items:
        if not isinstance(item, str):
            raise SystemExit(
                'OVERLAY_ARGS items must be strings, '
                f'got {type(item).__name__}')
        for token in shlex.split(os.path.expandvars(item), posix=True):
            if '\n' in token or '\r' in token:
                raise SystemExit('OVERLAY_ARGS token contains a newline')
            print(token)


if __name__ == '__main__':
    main()
