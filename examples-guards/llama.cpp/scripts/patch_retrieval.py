#!/usr/bin/env python3
"""Stop examples/retrieval when stdin ends.

The upstream loop reads a query and then keeps going on an empty line,
so one CI prompt hangs until the job timeout. This edits the CI working
copy only. It does not change the retrieval math, and it must not be
committed back to the upstream checkout.
"""
from __future__ import annotations

import sys
from pathlib import Path

OLD = """        std::getline(std::cin, query);
        std::vector<int32_t> query_tokens = common_tokenize(ctx, query, true);
"""
NEW = """        std::getline(std::cin, query);
        if (query.empty() || std::cin.eof()) {
            break;
        }
        std::vector<int32_t> query_tokens = common_tokenize(ctx, query, true);
"""


def main() -> int:
    if len(sys.argv) != 2:
        print(f'usage: {sys.argv[0]} <retrieval.cpp>', file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    text = path.read_text(encoding='utf-8')
    if NEW in text:
        print(f'retrieval already exits on EOF: {path}', flush=True)
        return 0
    if OLD not in text:
        print(f'retrieval getline loop not found in {path}', file=sys.stderr)
        return 1
    path.write_text(text.replace(OLD, NEW, 1), encoding='utf-8')
    print(f'patched {path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
