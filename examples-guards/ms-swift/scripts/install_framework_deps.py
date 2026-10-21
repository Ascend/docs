#!/usr/bin/env python3
"""Install ms-swift's framework requirements for an NPU runner.

usage: install_framework_deps.py <target-root>

The target checkout itself is installed separately with --no-deps. This
installs requirements/framework.txt except gradio, fastapi and uvicorn,
which serve the web UI and swift deploy and which no supported example
uses. It adds qwen_vl_utils, which the Qwen3.5 multimodal template
imports, and math_verify, which the GRPO accuracy reward imports. The
audio and omni packages are left out: only the Qwen3-Omni examples need
them, and those are unsupported. The torch stack is held by PIP_CONSTRAINT
(constraints-npu.txt), so none of these installs can replace it.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

SKIPPED = {'gradio', 'fastapi', 'uvicorn'}
EXTRAS = [
    'decorator',
    'qwen_vl_utils>=0.0.14',
    'math_verify',
]


def framework_requirements(target_root: Path) -> list[str]:
    requirements = []
    lines = (target_root / 'requirements' / 'framework.txt').read_text(
        encoding='utf-8').splitlines()
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith('#') or line.startswith('-'):
            continue
        name = re.split(r'[<>=!~;\[]', line, maxsplit=1)[0].strip().lower()
        if name in SKIPPED:
            print(f'skip {line}', flush=True)
            continue
        requirements.append(line)
    return requirements


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f'usage: {sys.argv[0]} <target-root>')
    requirements = framework_requirements(Path(sys.argv[1])) + EXTRAS
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *requirements])


if __name__ == '__main__':
    main()
