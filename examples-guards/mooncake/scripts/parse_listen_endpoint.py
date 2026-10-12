#!/usr/bin/env python3
"""Print the last `listening on host:port` line from a target log.

Transfer Engine logs `Transfer Engine RPC using <protocol>, listening on
<host>:<port>` once the target is ready. The initiator needs that
endpoint as --segment_id. Exit 1 when the log does not have one yet.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f'usage: {sys.argv[0]} <log>')
    text = Path(sys.argv[1]).read_text(encoding='utf-8', errors='replace')
    matches = re.findall(r'listening on (\S+:\d+)', text)
    if not matches:
        raise SystemExit(1)
    print(matches[-1])


if __name__ == '__main__':
    main()
