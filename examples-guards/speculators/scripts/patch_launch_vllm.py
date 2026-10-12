#!/usr/bin/env python3
"""Working-copy patch for scripts/launch_vllm.py.

Train examples exec `python -m vllm.entrypoints.cli.main serve` and never
hit the PATH shim named vllm. Append the NPU serve flags onto `cmd`
after it is built and before it is printed / exec'd.
Never git add/commit/push the target tree.
"""
from __future__ import annotations

import sys
from pathlib import Path

from vllm_npu_flags import VLLM_NPU_FLAGS

HELPER = (
    'def _append_ci_npu_flags(cmd):\n'
    f'    extra = {list(VLLM_NPU_FLAGS)!r}\n'
    '    if extra[0] in cmd:\n'
    '        return cmd\n'
    '    return [*cmd, *extra]\n\n\n'
)

PRINT_MARKERS = (
    '    print("Running command:")',
    "    print('Running command:')",
)


def patch(path: Path) -> None:
    text = path.read_text()
    if '_append_ci_npu_flags' in text:
        print(f'already patched {path}')
        return
    if 'def main():' not in text:
        raise SystemExit(f'no def main() in {path}')
    text = text.replace('def main():', HELPER + 'def main():', 1)
    inserted = False
    for marker in PRINT_MARKERS:
        if marker in text:
            text = text.replace(
                marker,
                '    cmd = _append_ci_npu_flags(cmd)\n' + marker,
                1,
            )
            inserted = True
            break
    if not inserted:
        raise SystemExit(f'no Running command print in {path}')
    path.write_text(text)
    print(f'patched {path}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: patch_launch_vllm.py <launch_vllm.py>')
    target = Path(sys.argv[1])
    if not target.is_file():
        raise SystemExit(f'not a file: {target}')
    patch(target)
