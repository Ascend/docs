#!/usr/bin/env python3
"""Resolve peft's overlay model paths through the shared HF hub cache.

setup_example.sh passes the env var names its profile needs, for example
`hub_cache.py SFT_MODEL_PATH ROBERTA_BASE_PATH`. For each one this
appends VAR=<snapshot dir> to $GITHUB_ENV for the manifest's
overlay_args:

- a complete cached snapshot is used as is, without network access;
- otherwise the asset is downloaded into the cache, HuggingFace first
  and ModelScope as the fallback, so the next run hits the cache.

The cache root is HF_HOME (default ~/.cache/huggingface), which on the
self-hosted runners lives on the runner pool's persistent volume.
"""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from modelscope import snapshot_download as ms_snapshot_download

HUB_ROOT = Path(os.environ.get(
    'HF_HOME', os.path.expanduser('~/.cache/huggingface'))) / 'hub'
# ModelScope's own snapshot bookkeeping, not repository content.
MODELSCOPE_METADATA = {'.mdl', '.msc', '.mv'}


@dataclass(frozen=True)
class Asset:
    hf_id: str
    ms_id: str
    # Files to fetch; skips the duplicate bin / h5 / onnx formats.
    patterns: tuple[str, ...]
    # Globs that must each match a snapshot file for a cache hit, so an
    # interrupted download is refetched instead of used.
    required: tuple[str, ...]


TEXT_MODEL = ('*.json', '*.txt', '*.safetensors')
TEXT_REQUIRED = ('config.json', '*.safetensors')

ASSETS = {
    'SFT_MODEL_PATH': Asset(
        'Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B', TEXT_MODEL, TEXT_REQUIRED),
    'ROBERTA_BASE_PATH': Asset(
        'roberta-base', 'AI-ModelScope/roberta-base', TEXT_MODEL, TEXT_REQUIRED),
    'BERT_BASE_UNCASED_PATH': Asset(
        'bert-base-uncased', 'AI-ModelScope/bert-base-uncased', TEXT_MODEL,
        TEXT_REQUIRED),
    'SD_MODEL_PATH': Asset(
        'stable-diffusion-v1-5/stable-diffusion-v1-5',
        'AI-ModelScope/stable-diffusion-v1-5',
        ('model_index.json', '*/*.json', '*/model.safetensors',
         '*/diffusion_pytorch_model.safetensors', 'tokenizer/*',
         'feature_extractor/*'),
        ('model_index.json', 'unet/diffusion_pytorch_model.safetensors',
         'vae/diffusion_pytorch_model.safetensors',
         'text_encoder/model.safetensors')),
}


def repo_dir(asset: Asset) -> Path:
    return HUB_ROOT / f"models--{asset.hf_id.replace('/', '--')}"


def cached_snapshot(asset: Asset) -> Path | None:
    """The snapshot refs/main points at, if it holds every required file."""
    refs = repo_dir(asset) / 'refs' / 'main'
    if not refs.is_file():
        return None
    snap = repo_dir(asset) / 'snapshots' / refs.read_text().strip()
    if not snap.is_dir():
        return None
    if all(any(p.is_file() for p in snap.glob(g)) for g in asset.required):
        return snap
    return None


def from_huggingface(asset: Asset) -> Path:
    # No revision: the hub writes refs/main only when it resolves a
    # branch name, and cached_snapshot reads refs/main.
    return Path(snapshot_download(
        repo_id=asset.hf_id, allow_patterns=list(asset.patterns),
        cache_dir=str(HUB_ROOT)))


def from_modelscope(asset: Asset) -> Path:
    """Copy the ModelScope snapshot into the HF cache layout."""
    try:
        sha = HfApi().model_info(asset.hf_id).sha
    except Exception:  # noqa: BLE001 - HuggingFace may be the reason we are here
        refs = repo_dir(asset) / 'refs' / 'main'
        sha = refs.read_text().strip() if refs.is_file() else 'modelscope'
    model_cache = Path(os.environ.get(
        'MODELSCOPE_CACHE', os.path.expanduser('~/.cache/modelscope')))
    # snapshot_download does not create its cache root, and a fresh
    # runner container has none: it then fails mid-transfer.
    model_cache.mkdir(parents=True, exist_ok=True)
    src = Path(ms_snapshot_download(
        asset.ms_id, cache_dir=str(model_cache),
        allow_patterns=list(asset.patterns)))
    snap = repo_dir(asset) / 'snapshots' / sha
    for item in src.rglob('*'):
        rel = item.relative_to(src)
        if not item.is_file() or rel.as_posix() in MODELSCOPE_METADATA:
            continue
        dest = snap / rel
        if dest.is_symlink():
            dest.unlink()
        elif dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + '.part')
        shutil.copy2(item, part)
        os.replace(part, dest)
    (repo_dir(asset) / 'refs').mkdir(parents=True, exist_ok=True)
    # No trailing newline: the hub compares this string with the
    # snapshot folder name without stripping it.
    (repo_dir(asset) / 'refs' / 'main').write_text(sha)
    return snap


def resolve(var: str) -> Path:
    asset = ASSETS[var]
    snap = cached_snapshot(asset)
    if snap is not None:
        print(f'[cache] {asset.hf_id} -> {snap}', flush=True)
        return snap
    print(f'[download] {asset.hf_id} <- HuggingFace', flush=True)
    try:
        snap = from_huggingface(asset)
    except Exception as exc:  # noqa: BLE001 - falls back below
        print(f'[fallback] {asset.hf_id}: HuggingFace failed '
              f'({type(exc).__name__}: {exc}); trying ModelScope {asset.ms_id}',
              flush=True)
        snap = from_modelscope(asset)
    if cached_snapshot(asset) != snap:
        raise RuntimeError(
            f'{asset.hf_id}: downloaded snapshot {snap} lacks {asset.required}')
    print(f'[cached] {asset.hf_id} -> {snap}', flush=True)
    return snap


def main(argv: list[str]) -> int:
    unknown = [var for var in argv if var not in ASSETS]
    if not argv or unknown:
        print(f'usage: hub_cache.py VAR... (known: {sorted(ASSETS)}; '
              f'unknown: {unknown})', file=sys.stderr)
        return 2
    for var in argv:
        try:
            snap = resolve(var)
        except Exception as exc:  # noqa: BLE001 - reported as guard noise
            print(f'{ASSETS[var].hf_id}: not in the shared cache and neither '
                  f'HuggingFace nor ModelScope delivered it ({exc}) - guard '
                  'noise, not an example failure', file=sys.stderr)
            return 1
        with open(os.environ['GITHUB_ENV'], 'a', encoding='utf-8') as fh:
            fh.write(f'{var}={snap}\n')
        print(f'{var}={snap}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
