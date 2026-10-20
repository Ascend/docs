#!/usr/bin/env python3
"""Print OVERLAY_ARGS as one line of shell-quoted tokens.

Empty, null, and missing values print nothing. Each JSON string is
expanded with os.path.expandvars and then split with POSIX shell rules,
matching the manifest values that contain ${EXAMPLES_ROOT}.
"""
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
    if items in (None, ''):
        return
    if not isinstance(items, list):
        raise SystemExit(
            f'OVERLAY_ARGS must be a JSON array, got {type(items).__name__}')
    tokens: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise SystemExit(
                f'OVERLAY_ARGS items must be strings, got {type(item).__name__}')
        tokens.extend(shlex.split(os.path.expandvars(item), posix=True))
    print(' '.join(shlex.quote(token) for token in tokens))


if __name__ == '__main__':
    main()
