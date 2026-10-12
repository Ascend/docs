#!/usr/bin/env python3
"""Download one GGUF into the runner cache.

Hugging Face first. ModelScope only if that download fails. Bytes land in
a private .part file and are renamed into the cache only after curl
finishes and the GGUF header checks out, so parallel matrix legs do not
append to one shared file.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from nfs_lock import download_lock, sweep_staging

CACHE = Path(os.environ.get(
    'LLAMA_CPP_CACHE', os.path.expanduser('~/.cache/llama.cpp')))
MAGIC = b'GGUF'


def gguf_ok(path: Path, min_bytes: int) -> bool:
    if not path.is_file():
        return False
    try:
        if path.stat().st_size < min_bytes:
            return False
        with path.open('rb') as handle:
            return handle.read(4) == MAGIC
    except OSError:
        return False


def curl_to(url: str, part: Path) -> None:
    part.parent.mkdir(parents=True, exist_ok=True)
    if part.is_file():
        with part.open('rb') as handle:
            if handle.read(4) != MAGIC:
                part.unlink()
    subprocess.run(
        [
            'curl', '-fL', '--retry', '5', '--retry-delay', '5',
            '--connect-timeout', '30', '-C', '-', url, '-o', str(part),
        ],
        check=True,
    )


def publish(name: str, min_bytes: int, hf_url: str, ms_url: str) -> Path:
    dest = CACHE / name
    if dest.exists() and not gguf_ok(dest, min_bytes):
        dest.unlink()
    staging = CACHE / 'staging' / f'{name}.{os.getpid()}'
    staging.mkdir(parents=True, exist_ok=True)
    part = staging / name
    try:
        print(f'[download] {name} <- HuggingFace', flush=True)
        try:
            curl_to(hf_url, part)
        except subprocess.CalledProcessError as exc:
            print(f'[fallback] HuggingFace failed ({exc}); trying ModelScope',
                  flush=True)
            part.unlink(missing_ok=True)
            try:
                curl_to(ms_url, part)
            except subprocess.CalledProcessError as ms_exc:
                print(f'{name}: neither HuggingFace nor ModelScope delivered '
                      f'the file ({ms_exc}) - download noise, not an example '
                      'failure', file=sys.stderr)
                raise SystemExit(1) from ms_exc
        if not gguf_ok(part, min_bytes):
            print(f'{name}: downloaded file is not a complete GGUF',
                  file=sys.stderr)
            raise SystemExit(1)
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(part, dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(f'model ready: {dest}', flush=True)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--min-bytes', type=int, required=True)
    parser.add_argument('--hf-url', required=True)
    parser.add_argument('--ms-url', required=True)
    parser.add_argument('--env-var', default='LLAMA_CI_MODEL')
    args = parser.parse_args()
    os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
    sweep_staging(CACHE / 'staging')
    dest = CACHE / args.name
    with download_lock(CACHE / 'locks', args.name):
        if gguf_ok(dest, args.min_bytes):
            print(f'reusing {dest}', flush=True)
        else:
            dest = publish(args.name, args.min_bytes, args.hf_url, args.ms_url)
    github_env = os.environ.get('GITHUB_ENV')
    if not github_env:
        print('GITHUB_ENV is required', file=sys.stderr)
        return 1
    with open(github_env, 'a', encoding='utf-8') as handle:
        handle.write(f'{args.env_var}={dest}\n')
    print(f'{args.env_var}={dest}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
