#!/usr/bin/env python3
"""Fail if bitsandbytes fell back to its mock native library."""
from __future__ import annotations

import sys

import bitsandbytes
import bitsandbytes.cextension as ce

MOCK = 'ErrorHandlerMockBNBNativeLibrary'


def main() -> int:
    name = type(ce.lib).__name__
    print('bitsandbytes', bitsandbytes.__version__)
    print('BNB_BACKEND', ce.BNB_BACKEND)
    print('lib', name)
    if name == MOCK:
        print('native library missing; CPU .so did not load', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
