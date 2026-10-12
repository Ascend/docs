#!/usr/bin/env python3
"""huggingface_hub stand-in for `hf download REPO FILE --local-dir DIR`.

The evaluate/train PATH shim calls this instead of the real hf CLI.
Only the download subcommand is implemented.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser(prog='hf')
    parser.add_argument('subcommand')
    parser.add_argument('repo_id')
    parser.add_argument('filename')
    parser.add_argument('--repo-type', default='model')
    parser.add_argument('--local-dir', required=True)
    args = parser.parse_args()
    if args.subcommand != 'download':
        raise SystemExit('CI shim 只实现 download')
    dest = Path(args.local_dir)
    dest.mkdir(parents=True, exist_ok=True)
    got = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        repo_type=args.repo_type,
    )
    target = dest / args.filename
    src = Path(got)
    if src.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    print(target)


if __name__ == '__main__':
    main()
