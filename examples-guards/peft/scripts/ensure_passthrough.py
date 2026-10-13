"""Attach "$@" to the last command line of a shell example.

Usage: ensure_passthrough.py <script.sh>

Shell examples that do not forward "$@" cannot take the overlay args.
This appends ' "$@"' to the last non-comment, non-blank line (the tail of
the example's main command) in the CI working copy only. Never commit it
back to the target repo.
"""
from pathlib import Path
import sys


def main():
    path = Path(sys.argv[1])
    lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped and not stripped.startswith('#'):
            raw = lines[i]
            newline = ''
            if raw.endswith('\r\n'):
                newline = '\r\n'
                raw = raw[:-2]
            elif raw.endswith('\n'):
                newline = '\n'
                raw = raw[:-1]
            lines[i] = raw.rstrip() + ' "$@"' + newline
            path.write_text(''.join(lines), encoding='utf-8')
            print(f'patched {path} to pass "$@" on last command line')
            raise SystemExit(0)
    raise SystemExit(f'{path}: cannot find a command line to attach "$@"')


if __name__ == '__main__':
    main()
