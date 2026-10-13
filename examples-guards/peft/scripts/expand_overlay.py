"""Expand the manifest's overlay_args into one shell-quoted line.

The workflow serializes manifest.overlay_args as JSON into OVERLAY_ARGS.
Each item is shell-split with its quoting intact, then CI paths such as
${CI_OUTPUT_DIR} / ${SFT_MODEL_PATH} resolve only in this job's
environment. The tokens are printed space-joined, each shell-quoted, so
the caller can eval them into a bash array. Nothing is printed when
there are no overlay args.
"""
import json
import os
import shlex


def main():
    raw = os.environ.get('OVERLAY_ARGS', '').strip()
    if not raw or raw in ('null', '""'):
        raise SystemExit(0)
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f'OVERLAY_ARGS is not valid JSON: {exc}') from exc
    if items in (None, ''):
        raise SystemExit(0)
    if not isinstance(items, list):
        raise SystemExit(
            f'OVERLAY_ARGS must be a JSON array, got {type(items).__name__}')
    tokens = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise SystemExit('OVERLAY_ARGS items must be non-empty strings')
        tokens.extend(shlex.split(os.path.expandvars(item), posix=True))
    print(' '.join(shlex.quote(token) for token in tokens))


if __name__ == '__main__':
    main()
