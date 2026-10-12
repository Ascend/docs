#!/usr/bin/env python3
"""Working-copy patches for one speculators example script.

Only the edits that PATH shims cannot do: make config vars env-overridable,
replace the unbounded health wait, and append "$@" so overlay_args reach
the CLI. vLLM serve flags and `hf download` are interposed at runtime.
Never git add/commit/push the target tree.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

CONFIG_VARS = (
    'VLLM_GPUS',
    'TRAIN_GPUS',
    'NUM_TRAIN_GPUS',
    'VLLM_GPU',
    'TRAIN_GPU',
    'GPUS',
    'NUM_GPUS',
    'MODEL',
    'MAX_SAMPLES',
    'SEQ_LENGTH',
    'EPOCHS',
    'CONCURRENCY',
    'VLLM_PORT',
    'MAX_ANCHORS',
)

FRAGMENT_DIR = Path(__file__).resolve().parent / 'fragments'
HEALTH_WAIT = FRAGMENT_DIR / 'health_wait.sh'
HEALTH_URL_PLACEHOLDER = '__HEALTH_URL__'

HEALTH_WAIT_PATTERN = re.compile(
    r'echo "Waiting for vLLM server to be ready\.\.\."\n'
    r'until curl -sf "([^"]+)" > /dev/null 2>&1; do\n'
    r'    sleep 2\n'
    r'done\n'
    r'echo "vLLM server ready\."',
    re.M,
)


def make_env_overridable(text: str) -> str:
    names = '|'.join(CONFIG_VARS)

    def repl(match: re.Match[str]) -> str:
        var = match.group(1)
        raw = match.group(2)
        inner = raw[1:-1] if raw.startswith('"') else raw
        return f'{var}="${{{var}:-{inner}}}"'

    return re.sub(
        rf'^({names})=("[^"]*"|[0-9]+)\s*$',
        repl,
        text,
        flags=re.M,
    )


def bound_health_wait(text: str) -> str:
    matches = list(HEALTH_WAIT_PATTERN.finditer(text))
    if len(matches) != 1:
        raise SystemExit(f'health wait loop not patched (matches={len(matches)})')
    url = matches[0].group(1)
    snippet = HEALTH_WAIT.read_text()
    if HEALTH_URL_PLACEHOLDER not in snippet:
        raise SystemExit(f'{HEALTH_WAIT} missing {HEALTH_URL_PLACEHOLDER}')
    replacement = snippet.replace(HEALTH_URL_PLACEHOLDER, url).rstrip('\n')
    return HEALTH_WAIT_PATTERN.sub(replacement, text, count=1)


def append_overlay_receiver(text: str) -> str:
    if '"$@"' in text:
        return text
    if 'evaluate.py' in text:
        text = text.replace(
            '    --max-requests 80\n',
            '    --max-requests 80 \\\n    "$@"\n',
        )
        if '"$@"' not in text:
            raise SystemExit('failed to append "$@" to evaluate.py')
        return text
    if '-m speculators.train' not in text:
        raise SystemExit('no train/evaluate command to receive overlay args')
    lines = text.splitlines(keepends=True)
    start = next(
        (i for i, line in enumerate(lines) if '-m speculators.train' in line),
        None,
    )
    if start is None:
        raise SystemExit('could not find train flag block')
    insert_at = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].strip() == '' or not lines[i].startswith((' ', '\t')):
            insert_at = i
            break
    prev = insert_at - 1
    if not lines[prev].rstrip().endswith('\\'):
        lines[prev] = lines[prev].rstrip('\n') + ' \\\n'
    lines.insert(insert_at, '    "$@"\n')
    return ''.join(lines)


def patch(path: Path) -> None:
    text = path.read_text()
    text = make_env_overridable(text)
    text = bound_health_wait(text)
    text = append_overlay_receiver(text)
    path.write_text(text)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: patch_example.py <script>')
    target = Path(sys.argv[1])
    if not target.is_file():
        raise SystemExit(f'not a file: {target}')
    patch(target)
    print(f'patched {target}')
