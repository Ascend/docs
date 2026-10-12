#!/usr/bin/env python3
"""Print OVERLAY_ARGS as a shlex-quoted word list for bash eval.

OVERLAY_ARGS is a JSON array of strings, possibly empty or unset.
Each item is expandvars + shlex.split so overlay entries may contain
spaces. Invalid JSON exits non-zero. Empty / null prints nothing.
"""
from __future__ import annotations

import json
import os
import shlex


def tokens_from_env(raw: str) -> list[str]:
    text = raw.strip()
    if not text or text in ('null', '""'):
        return []
    try:
        items = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f'OVERLAY_ARGS is not valid JSON: {exc}') from exc
    if items in (None, ''):
        return []
    if not isinstance(items, list):
        raise SystemExit(
            f'OVERLAY_ARGS must be a JSON array, got {type(items).__name__}')
    tokens: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise SystemExit('OVERLAY_ARGS items must be non-empty strings')
        tokens.extend(shlex.split(os.path.expandvars(item), posix=True))
    return tokens


def main() -> None:
    tokens = tokens_from_env(os.environ.get('OVERLAY_ARGS', ''))
    if tokens:
        print(' '.join(shlex.quote(token) for token in tokens))


if __name__ == '__main__':
    main()
