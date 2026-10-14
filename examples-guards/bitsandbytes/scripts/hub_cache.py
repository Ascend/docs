#!/usr/bin/env python3
"""Put bitsandbytes example assets into the shared Hugging Face cache.

setup_example.sh passes asset names, for example
`hub_cache.py LLAMA_MODEL ALPACA_DATASET`. Each name is resolved to the
hub id the example script already uses:

- a cached snapshot that holds every required file, with no corrupt
  safetensors, is reused and nothing is downloaded;
- otherwise the asset is downloaded, Hugging Face first and ModelScope
  as the fallback, then published into the default hub cache so
  from_pretrained / load_dataset hit it with no path argument.

The cache root is HF_HOME (default ~/.cache/huggingface). On this
runner pool that tree lives on the persistent volume shared by every
pod. huggingface_hub's own locks do not exclude writers across pods, so
downloads go to a per-job staging directory on the same volume, get
checked, and are renamed into the snapshot. A mkdir lock per asset lets
one job download while others wait; correctness does not depend on the
lock.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from modelscope import snapshot_download as ms_snapshot_download
from safetensors import SafetensorError, safe_open

HUB_ROOT = Path(os.environ.get(
    'HF_HOME', os.path.expanduser('~/.cache/huggingface'))) / 'hub'
STAGING_DIR = 'guard-staging'
LOCKS_DIR = 'guard-locks'
MODELSCOPE_METADATA = {'.mdl', '.msc', '.mv'}
HEARTBEAT_SECONDS = 30
LOCK_STALE_SECONDS = 300
LOCK_POLL_SECONDS = 10
LOCK_WAIT_SECONDS = 900
STAGING_STALE_SECONDS = 24 * 3600


@dataclass(frozen=True)
class Asset:
    hf_id: str
    # Empty when no ModelScope copy is known. Hugging Face failure then
    # stays a download failure.
    ms_id: str
    repo_type: str
    patterns: tuple[str, ...]
    required: tuple[str, ...]


ASSETS = {
    'GEMMA_MODEL': Asset(
        'google/gemma-2-2b-it',
        'LLM-Research/gemma-2-2b-it',
        'model',
        ('*.json', '*.safetensors', 'tokenizer.model'),
        ('config.json', 'model.safetensors.index.json',
         'model-00001-of-00002.safetensors',
         'model-00002-of-00002.safetensors',
         'tokenizer_config.json'),
    ),
    'LLAMA_MODEL': Asset(
        'JackFram/llama-68m',
        '',
        'model',
        ('config.json', 'generation_config.json', 'special_tokens_map.json',
         'tokenizer_config.json', 'tokenizer.model', 'pytorch_model.bin'),
        ('config.json', 'pytorch_model.bin', 'tokenizer.model'),
    ),
    'ALPACA_DATASET': Asset(
        'yahma/alpaca-cleaned',
        'AI-ModelScope/alpaca-cleaned',
        'dataset',
        ('README.md', 'alpaca_data_cleaned.json'),
        ('alpaca_data_cleaned.json',),
    ),
}


def repo_dir(asset: Asset) -> Path:
    kind = 'datasets' if asset.repo_type == 'dataset' else 'models'
    return HUB_ROOT / f"{kind}--{asset.hf_id.replace('/', '--')}"


def safetensors_header_ok(path: Path) -> bool | None:
    """True when the header parses, False when the file is readable but
    malformed, None when it cannot be read.

    A read error on the shared volume is not evidence of corruption, so
    None must not delete a shard.
    """
    try:
        with safe_open(str(path), framework='numpy') as handle:
            list(handle.keys())
    except SafetensorError:
        return False
    except OSError:
        return None
    return True


def corrupt_shards(root: Path) -> list[Path]:
    return [
        path for path in sorted(root.rglob('*.safetensors'))
        if safetensors_header_ok(path) is False
    ]


def cached_snapshot(asset: Asset) -> Path | None:
    refs = repo_dir(asset) / 'refs' / 'main'
    if not refs.is_file():
        return None
    snap = repo_dir(asset) / 'snapshots' / refs.read_text().strip()
    if not snap.is_dir():
        return None
    if not all(any(path.is_file() for path in snap.glob(pattern))
               for pattern in asset.required):
        return None
    corrupt = corrupt_shards(snap)
    if corrupt:
        print(
            f'[corrupt] {asset.hf_id}: '
            f'{[path.relative_to(snap).as_posix() for path in corrupt]}',
            flush=True,
        )
        return None
    return snap


def purge_corrupt(asset: Asset) -> None:
    """Delete corrupt shards and the blobs behind them.

    snapshot_download copies a blob the cache already holds without
    rechecking it, so a damaged blob is reused until it is removed.
    """
    snapshots = repo_dir(asset) / 'snapshots'
    if not snapshots.is_dir():
        return
    for snap in sorted(snapshots.iterdir()):
        for shard in corrupt_shards(snap):
            blob = Path(os.path.realpath(shard)) if shard.is_symlink() else None
            print(
                f'[purge] {asset.hf_id}: {shard.relative_to(snap).as_posix()}',
                flush=True,
            )
            shard.unlink(missing_ok=True)
            if blob is not None:
                blob.unlink(missing_ok=True)


def revision(asset: Asset) -> str:
    api = HfApi()
    try:
        if asset.repo_type == 'dataset':
            info = api.dataset_info(asset.hf_id)
        else:
            info = api.model_info(asset.hf_id)
        return info.sha
    except Exception:  # noqa: BLE001 - Hugging Face may be why we fall back
        refs = repo_dir(asset) / 'refs' / 'main'
        if refs.is_file():
            return refs.read_text().strip()
        return 'modelscope'


def from_huggingface(asset: Asset, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=asset.hf_id,
        repo_type=asset.repo_type,
        allow_patterns=list(asset.patterns),
        local_dir=str(dest),
    )
    return dest


def from_modelscope(asset: Asset, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    ms_repo_type = 'dataset' if asset.repo_type == 'dataset' else 'model'
    return Path(ms_snapshot_download(
        repo_id=asset.ms_id,
        repo_type=ms_repo_type,
        local_dir=str(dest),
        allow_patterns=list(asset.patterns),
    ))


def checked(src: Path, asset: Asset) -> Path:
    corrupt = corrupt_shards(src)
    if corrupt:
        raise RuntimeError(
            'corrupt safetensors downloaded: '
            f'{[path.relative_to(src).as_posix() for path in corrupt]}')
    missing = [
        pattern for pattern in asset.required
        if not any(path.is_file() for path in src.glob(pattern))
    ]
    if missing:
        raise RuntimeError(f'{asset.hf_id}: download missing {missing}')
    return src


def intact(dest: Path, staged: Path) -> bool:
    try:
        if dest.stat().st_size != staged.stat().st_size:
            return False
    except FileNotFoundError:
        return False
    if dest.suffix != '.safetensors':
        return True
    return safetensors_header_ok(dest) is not False


def publish(asset: Asset, src: Path, sha: str) -> Path:
    """Rename src's files into snapshots/<sha>, then point refs/main at it.

    Intact files stay in place: another job may be reading them.
    """
    snap = repo_dir(asset) / 'snapshots' / sha
    for item in sorted(src.rglob('*')):
        rel = item.relative_to(src)
        if (not item.is_file() or rel.parts[0] == '.cache'
                or rel.as_posix() in MODELSCOPE_METADATA):
            continue
        dest = snap / rel
        if intact(dest, item):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(item, dest)
    refs = repo_dir(asset) / 'refs'
    refs.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            'w', dir=refs, prefix='.main-', delete=False) as handle:
        # No trailing newline: the hub compares this string with the
        # snapshot folder name without stripping it.
        handle.write(sha)
        temp_name = handle.name
    os.replace(temp_name, refs / 'main')
    return snap


def fetch(asset: Asset) -> Path:
    sha = revision(asset)
    staging_root = HUB_ROOT.parent / STAGING_DIR
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f'{repo_dir(asset).name}-', dir=staging_root))
    try:
        print(f'[download] {asset.hf_id} <- Hugging Face', flush=True)
        try:
            src = checked(from_huggingface(asset, staging / 'hf'), asset)
        except Exception as exc:  # noqa: BLE001 - ModelScope is the fallback
            if not asset.ms_id:
                raise
            print(
                f'[fallback] {asset.hf_id}: Hugging Face failed '
                f'({type(exc).__name__}: {exc}); trying ModelScope {asset.ms_id}',
                flush=True,
            )
            src = checked(from_modelscope(asset, staging / 'ms'), asset)
        snap = publish(asset, src, sha)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if cached_snapshot(asset) != snap:
        raise RuntimeError(
            f'{asset.hf_id}: downloaded snapshot {snap} lacks {asset.required}')
    print(f'[cached] {asset.hf_id} -> {snap}', flush=True)
    return snap


def verify_dataset(asset: Asset) -> None:
    """Materialize the dataset from the planted hub snapshot.

    cpu_training.py calls load_dataset on the hub id. HF_HUB_OFFLINE
    keeps that read on the snapshot just published. HF_DATASETS_OFFLINE
    stays unset here so the arrow cache can be built from those local
    files; the run step turns it on afterwards.
    """
    saved = os.environ.get('HF_HUB_OFFLINE')
    os.environ['HF_HUB_OFFLINE'] = '1'
    try:
        dataset = load_dataset(asset.hf_id, split='train')
    finally:
        if saved is None:
            os.environ.pop('HF_HUB_OFFLINE', None)
        else:
            os.environ['HF_HUB_OFFLINE'] = saved
    if len(dataset) == 0:
        raise RuntimeError(f'{asset.hf_id}: dataset resolved empty')
    print(f'[dataset] {asset.hf_id} rows={len(dataset)}', flush=True)


def heartbeat(lock: Path, stop: threading.Event) -> None:
    while not stop.wait(HEARTBEAT_SECONDS):
        try:
            os.utime(lock)
        except OSError:
            return


def break_if_stale(lock: Path) -> None:
    try:
        idle = time.time() - lock.stat().st_mtime
    except FileNotFoundError:
        return
    if idle < LOCK_STALE_SECONDS:
        return
    print(f'[lock] {lock.name}: no heartbeat for {idle:.0f}s, breaking it',
          flush=True)
    stale = lock.with_name(f'{lock.name}.stale-{uuid.uuid4().hex}')
    try:
        os.rename(lock, stale)
    except FileNotFoundError:
        return
    shutil.rmtree(stale, ignore_errors=True)


@contextmanager
def download_lock(asset: Asset) -> Iterator[None]:
    locks = HUB_ROOT.parent / LOCKS_DIR
    locks.mkdir(parents=True, exist_ok=True)
    lock = locks / repo_dir(asset).name
    deadline = time.monotonic() + LOCK_WAIT_SECONDS
    waiting = False
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            pass
        if time.monotonic() >= deadline:
            print(
                f'[lock] {asset.hf_id}: still held after {LOCK_WAIT_SECONDS}s, '
                'downloading without it',
                flush=True,
            )
            yield
            return
        if not waiting:
            print(f'[lock] {asset.hf_id}: another job is downloading it, waiting',
                  flush=True)
            waiting = True
        break_if_stale(lock)
        time.sleep(LOCK_POLL_SECONDS)
    stop = threading.Event()
    beat = threading.Thread(target=heartbeat, args=(lock, stop), daemon=True)
    beat.start()
    try:
        yield
    finally:
        stop.set()
        beat.join()
        shutil.rmtree(lock, ignore_errors=True)


def resolve(name: str) -> Path:
    asset = ASSETS[name]
    snap = cached_snapshot(asset)
    if snap is None:
        with download_lock(asset):
            snap = cached_snapshot(asset)
            if snap is None:
                purge_corrupt(asset)
                snap = fetch(asset)
    else:
        print(f'[cache] {asset.hf_id} -> {snap}', flush=True)
    if asset.repo_type == 'dataset':
        verify_dataset(asset)
    return snap


def sweep_staging() -> None:
    staging_root = HUB_ROOT.parent / STAGING_DIR
    if not staging_root.is_dir():
        return
    cutoff = time.time() - STAGING_STALE_SECONDS
    for entry in staging_root.iterdir():
        try:
            if entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            continue


def main(argv: list[str]) -> int:
    unknown = [name for name in argv if name not in ASSETS]
    if not argv or unknown:
        print(
            f'usage: hub_cache.py NAME... (known: {sorted(ASSETS)}; '
            f'unknown: {unknown})',
            file=sys.stderr,
        )
        return 2
    sweep_staging()
    for name in argv:
        asset = ASSETS[name]
        try:
            snap = resolve(name)
        except Exception as exc:  # noqa: BLE001 - reported as guard noise
            print(
                f'{asset.hf_id}: not in the shared cache and neither '
                f'Hugging Face nor ModelScope delivered it ({exc}) - guard '
                'noise, not an example failure',
                file=sys.stderr,
            )
            return 1
        print(f'{name}={snap}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
