#!/usr/bin/env python3
"""Resolve peft's overlay model paths through the shared HF hub cache.

setup_example.sh passes the env var names its profile needs, for example
`hub_cache.py SFT_MODEL_PATH ROBERTA_BASE_PATH`. For each one this
appends VAR=<snapshot dir> to $GITHUB_ENV for the manifest's
overlay_args:

- a cached snapshot that holds every required file and no corrupt
  safetensors is used as is, without network access;
- otherwise the asset is downloaded, HuggingFace first and ModelScope
  as the fallback, and published into the cache, so the next run hits
  the cache.

The cache root is HF_HOME (default ~/.cache/huggingface), which on the
self-hosted runners lives on the runner pool's persistent NFS volume,
shared by every concurrent matrix leg on every pod. The hub's own
.locks do not exclude across pods there: in run 37764199490 two legs
fetching SD v1.5 appended to one blob and left it larger than the whole
file. So nothing is downloaded into the cache in place. A job
downloads into its own staging dir on the same volume, checks it, and
renames the files into the snapshot, which readers then see whole or
not at all. A mkdir lock per asset lets one leg download while the
others wait for it; correctness does not depend on the lock.
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

from huggingface_hub import HfApi, snapshot_download
from modelscope import snapshot_download as ms_snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tests'))
from doc_test.model_cache import safetensors_header_ok  # noqa: E402

HUB_ROOT = Path(os.environ.get(
    'HF_HOME', os.path.expanduser('~/.cache/huggingface'))) / 'hub'
# Siblings of HUB_ROOT on the same volume, so a rename from staging into
# the snapshot is atomic, kept out of the hub's own cache layout.
STAGING_DIR = 'guard-staging'
LOCKS_DIR = 'guard-locks'
# ModelScope's own snapshot bookkeeping, not repository content.
MODELSCOPE_METADATA = {'.mdl', '.msc', '.mv'}
HEARTBEAT_SECONDS = 30
# A lock this long without a heartbeat belongs to a killed job.
LOCK_STALE_SECONDS = 300
LOCK_POLL_SECONDS = 10
# Past this, download alongside the holder rather than spend the job's
# timeout waiting.
LOCK_WAIT_SECONDS = 900
# Longer than any job lives: the job that made it was killed.
STAGING_STALE_SECONDS = 24 * 3600


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


def corrupt_shards(root: Path) -> list[Path]:
    """Safetensors under root whose header or size is inconsistent."""
    return [p for p in sorted(root.rglob('*.safetensors'))
            if safetensors_header_ok(p) is False]


def cached_snapshot(asset: Asset) -> Path | None:
    """The snapshot refs/main points at, if it holds every required file
    and none of its safetensors is corrupt."""
    refs = repo_dir(asset) / 'refs' / 'main'
    if not refs.is_file():
        return None
    snap = repo_dir(asset) / 'snapshots' / refs.read_text().strip()
    if not snap.is_dir():
        return None
    if not all(any(p.is_file() for p in snap.glob(g)) for g in asset.required):
        return None
    corrupt = corrupt_shards(snap)
    if corrupt:
        print(f'[corrupt] {asset.hf_id}: '
              f'{[p.relative_to(snap).as_posix() for p in corrupt]}', flush=True)
        return None
    return snap


def purge_corrupt(asset: Asset) -> None:
    """Delete corrupt shards and the blobs behind them.

    snapshot_download copies a blob the cache already holds into the
    download target without rechecking it, so a damaged blob is handed
    forward on every retry until it is gone.
    """
    snapshots = repo_dir(asset) / 'snapshots'
    if not snapshots.is_dir():
        return
    for snap in sorted(snapshots.iterdir()):
        for shard in corrupt_shards(snap):
            blob = Path(os.path.realpath(shard)) if shard.is_symlink() else None
            print(f'[purge] {asset.hf_id}: {shard.relative_to(snap).as_posix()}',
                  flush=True)
            shard.unlink(missing_ok=True)
            if blob is not None:
                blob.unlink(missing_ok=True)


def revision(asset: Asset) -> str:
    """The hub commit to file the snapshot under."""
    try:
        return HfApi().model_info(asset.hf_id).sha
    except Exception:  # noqa: BLE001 - HuggingFace may be the reason we fall back
        refs = repo_dir(asset) / 'refs' / 'main'
        return refs.read_text().strip() if refs.is_file() else 'modelscope'


def from_huggingface(asset: Asset, dest: Path) -> Path:
    snapshot_download(
        repo_id=asset.hf_id, allow_patterns=list(asset.patterns),
        local_dir=str(dest))
    return dest


def from_modelscope(asset: Asset, dest: Path) -> Path:
    # snapshot_download does not create its cache root: it then fails
    # mid-transfer.
    dest.mkdir(parents=True, exist_ok=True)
    return Path(ms_snapshot_download(
        asset.ms_id, cache_dir=str(dest),
        allow_patterns=list(asset.patterns)))


def checked(src: Path) -> Path:
    corrupt = corrupt_shards(src)
    if corrupt:
        raise RuntimeError('corrupt safetensors downloaded: '
                           f'{[p.relative_to(src).as_posix() for p in corrupt]}')
    return src


def intact(dest: Path, staged: Path) -> bool:
    """Whether dest already holds staged's content, by size and header."""
    try:
        if dest.stat().st_size != staged.stat().st_size:
            return False
    except FileNotFoundError:
        return False
    return dest.suffix != '.safetensors' or safetensors_header_ok(dest) is not False


def publish(asset: Asset, src: Path, sha: str) -> Path:
    """Rename src's files into snapshots/<sha>, then point refs/main at it.

    Intact files are left in place: another job may be reading them.
    """
    snap = repo_dir(asset) / 'snapshots' / sha
    for item in sorted(src.rglob('*')):
        rel = item.relative_to(src)
        # .cache is huggingface_hub's local_dir download bookkeeping.
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
            'w', dir=refs, prefix='.main-', delete=False) as fh:
        # No trailing newline: the hub compares this string with the
        # snapshot folder name without stripping it.
        fh.write(sha)
    os.replace(fh.name, refs / 'main')
    return snap


def fetch(asset: Asset) -> Path:
    """Download into a staging dir of this job, check it, publish it."""
    sha = revision(asset)
    staging_root = HUB_ROOT.parent / STAGING_DIR
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f'{repo_dir(asset).name}-', dir=staging_root))
    try:
        print(f'[download] {asset.hf_id} <- HuggingFace', flush=True)
        try:
            src = checked(from_huggingface(asset, staging / 'hf'))
        except Exception as exc:  # noqa: BLE001 - falls back below
            print(f'[fallback] {asset.hf_id}: HuggingFace failed '
                  f'({type(exc).__name__}: {exc}); trying ModelScope {asset.ms_id}',
                  flush=True)
            src = checked(from_modelscope(asset, staging / 'ms'))
        snap = publish(asset, src, sha)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if cached_snapshot(asset) != snap:
        raise RuntimeError(
            f'{asset.hf_id}: downloaded snapshot {snap} lacks {asset.required}')
    print(f'[cached] {asset.hf_id} -> {snap}', flush=True)
    return snap


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
    # Rename first: of several waiters breaking it, only one succeeds.
    stale = lock.with_name(f'{lock.name}.stale-{uuid.uuid4().hex}')
    try:
        os.rename(lock, stale)
    except FileNotFoundError:
        return
    shutil.rmtree(stale, ignore_errors=True)


@contextmanager
def download_lock(asset: Asset) -> Iterator[None]:
    """Hold the asset's lock for the body; after LOCK_WAIT_SECONDS of
    waiting, run the body without it."""
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
            print(f'[lock] {asset.hf_id}: still held after {LOCK_WAIT_SECONDS}s, '
                  'downloading without it', flush=True)
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


def resolve(var: str) -> Path:
    asset = ASSETS[var]
    snap = cached_snapshot(asset)
    if snap is None:
        with download_lock(asset):
            # The previous holder may have published it meanwhile.
            snap = cached_snapshot(asset)
            if snap is None:
                purge_corrupt(asset)
                return fetch(asset)
    print(f'[cache] {asset.hf_id} -> {snap}', flush=True)
    return snap


def sweep_staging() -> None:
    """Remove staging dirs that killed jobs left behind."""
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
    unknown = [var for var in argv if var not in ASSETS]
    if not argv or unknown:
        print(f'usage: hub_cache.py VAR... (known: {sorted(ASSETS)}; '
              f'unknown: {unknown})', file=sys.stderr)
        return 2
    sweep_staging()
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
