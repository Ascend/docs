#!/usr/bin/env python3
"""Build Dream-v0-Instruct-7B.Q8_0.gguf on the runner cache.

examples/diffusion needs this GGUF. Hugging Face publishes the official
safetensors, not the GGUF, so this downloads those weights and runs the
checkout's convert_hf_to_gguf.py. ModelScope is only the fallback for the
weights. A finished GGUF is reused; a failed convert leaves the weights
in place for the next attempt.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from modelscope import snapshot_download as ms_snapshot_download
from safetensors import SafetensorError, safe_open

from nfs_lock import download_lock, sweep_staging

HF_ID = 'Dream-org/Dream-v0-Instruct-7B'
MS_ID = 'Dream-org/Dream-v0-Instruct-7B'
PATTERNS = ('*.json', '*.txt', '*.py', '*.safetensors')
SKIP_NAMES = {'.mdl', '.msc', '.mv', '.gitattributes'}
MIN_GGUF_BYTES = 1_000_000_000

CACHE = Path(os.environ.get(
    'LLAMA_CPP_CACHE', os.path.expanduser('~/.cache/llama.cpp')))
WEIGHTS = CACHE / 'Dream-v0-Instruct-7B-hf'
GGUF = CACHE / 'Dream-v0-Instruct-7B.Q8_0.gguf'
MAGIC = b'GGUF'


def gguf_ok(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        if path.stat().st_size < MIN_GGUF_BYTES:
            return False
        with path.open('rb') as handle:
            return handle.read(4) == MAGIC
    except OSError:
        return False


def header_ok(path: Path) -> bool | None:
    """True when the safetensors header parses, False when it is corrupt.

    None means the file could not be read. A shared-volume read error is
    not evidence of corruption, so the caller must not delete it.
    """
    if not path.is_file():
        return False
    try:
        with safe_open(str(path), framework='numpy') as handle:
            list(handle.keys())
    except SafetensorError:
        return False
    except Exception:  # noqa: BLE001 - OSError and the like: cannot judge
        return None
    return True


def purge_hub_shards() -> None:
    """Drop corrupt Dream shards from the Hugging Face hub cache.

    snapshot_download copies a blob the cache already holds without
    checking it again, so a damaged blob is handed forward until removed.
    """
    root = Path(os.path.expanduser(
        '~/.cache/huggingface/hub')) / f"models--{HF_ID.replace('/', '--')}"
    snapshots = root / 'snapshots'
    if not snapshots.is_dir():
        return
    for snap in snapshots.iterdir():
        for shard in snap.rglob('*.safetensors'):
            if header_ok(shard) is not False:
                continue
            blob = Path(os.path.realpath(shard)) if shard.is_symlink() else None
            print(f'[purge] {shard}', flush=True)
            shard.unlink(missing_ok=True)
            if blob is not None and blob.is_file() and blob != shard:
                blob.unlink(missing_ok=True)


def weights_ready(root: Path) -> bool:
    if not (root / 'config.json').is_file():
        return False
    if not (root / 'vocab.json').is_file() and not (root / 'tokenizer.json').is_file():
        return False
    index = root / 'model.safetensors.index.json'
    if not index.is_file():
        return False
    try:
        names = set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values())
    except (OSError, json.JSONDecodeError, KeyError, AttributeError):
        return False
    if not names:
        return False
    return all(header_ok(root / name) is True for name in names)


def download_weights(staging: Path) -> Path:
    hf_dir = staging / 'hf'
    print(f'[download] {HF_ID} <- HuggingFace', flush=True)
    try:
        snapshot_download(
            HF_ID, allow_patterns=list(PATTERNS), local_dir=str(hf_dir))
    except Exception as exc:  # noqa: BLE001 - fall back to ModelScope
        print(f'[fallback] HuggingFace failed ({type(exc).__name__}: {exc}); '
              f'trying ModelScope {MS_ID}', flush=True)
        ms_dir = staging / 'ms'
        ms_dir.mkdir(parents=True, exist_ok=True)
        try:
            return Path(ms_snapshot_download(
                MS_ID, cache_dir=str(ms_dir), allow_patterns=list(PATTERNS)))
        except Exception as ms_exc:  # noqa: BLE001 - reported as download noise
            print('Dream weights: neither HuggingFace nor ModelScope delivered '
                  f'them ({type(ms_exc).__name__}: {ms_exc}) - download noise, '
                  'not an example failure', file=sys.stderr)
            raise SystemExit(1) from ms_exc
    return hf_dir


def place_file(src: Path, dest: Path) -> None:
    if src.is_symlink():
        src = src.resolve()
    if not src.is_file():
        raise RuntimeError(f'missing downloaded file: {src}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    same_size = dest.is_file() and dest.stat().st_size == src.stat().st_size
    if same_size and not dest.name.endswith('.safetensors'):
        return
    if same_size and header_ok(dest) is not False:
        return
    tmp = dest.with_name(f'{dest.name}.tmp-{os.getpid()}')
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)


def materialize(src: Path, dest: Path) -> None:
    for item in sorted(src.rglob('*')):
        rel = item.relative_to(src)
        if not item.is_file() and not item.is_symlink():
            continue
        if rel.parts[0] in {'.cache', '.git'} or rel.name in SKIP_NAMES:
            continue
        place_file(item, dest / rel)


def convert(weights: Path, target_root: Path) -> None:
    script = target_root / 'convert_hf_to_gguf.py'
    if not script.is_file():
        print(f'convert_hf_to_gguf.py not found in {target_root}', file=sys.stderr)
        raise SystemExit(1)
    part = GGUF.parent / f'{GGUF.name}.part'
    part.unlink(missing_ok=True)
    print(f'converting {weights} -> {GGUF}', flush=True)
    try:
        subprocess.run(
            [sys.executable, str(script), str(weights),
             '--outtype', 'q8_0', '--outfile', str(part)],
            check=True,
            cwd=target_root,
        )
    except subprocess.CalledProcessError as exc:
        print(f'Dream GGUF convert failed ({exc})', file=sys.stderr)
        raise SystemExit(1) from exc
    if not gguf_ok(part):
        print(f'convert produced no complete GGUF: {part}', file=sys.stderr)
        raise SystemExit(1)
    os.replace(part, GGUF)
    print(f'Dream GGUF ready: {GGUF}', flush=True)


def write_env(path: Path) -> None:
    github_env = os.environ.get('GITHUB_ENV')
    if not github_env:
        print('GITHUB_ENV is required', file=sys.stderr)
        raise SystemExit(1)
    with open(github_env, 'a', encoding='utf-8') as handle:
        handle.write(f'LLAMA_CI_MODEL={path}\n')
    print(f'LLAMA_CI_MODEL={path}', flush=True)


def main() -> int:
    os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
    target_root = Path(os.environ['TARGET_ROOT'])
    sweep_staging(CACHE / 'staging')
    with download_lock(CACHE / 'locks', 'dream-v0-instruct-7b'):
        if gguf_ok(GGUF):
            print(f'reusing {GGUF}', flush=True)
            write_env(GGUF)
            return 0
        if GGUF.exists():
            GGUF.unlink()
        if not weights_ready(WEIGHTS):
            purge_hub_shards()
            staging = CACHE / 'staging' / f'dream.{os.getpid()}'
            staging.mkdir(parents=True, exist_ok=True)
            try:
                src = download_weights(staging)
                materialize(src, WEIGHTS)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        if not weights_ready(WEIGHTS):
            print(f'Dream weights incomplete: {WEIGHTS}', file=sys.stderr)
            return 1
        convert(WEIGHTS, target_root)
    write_env(GGUF)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
