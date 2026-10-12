#!/usr/bin/env python3
"""Put one Hub repo into the shared Hugging Face cache without tearing it.

Several matrix legs download the same Qwen checkpoint. The runner pool
keeps ~/.cache on one NFS volume, and huggingface_hub's own locks do not
hold across pods. A leg therefore downloads into its own staging directory
on that volume, checks safetensors headers, and renames finished files
into the snapshot. Readers then see a file whole or not at all.

Hugging Face is tried first. ModelScope is the fallback for a failed or
gated repo. Files from either source are stored under the Hugging Face
repo id the example already uses, so a later offline load_dataset or
from_pretrained hits this snapshot.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from modelscope.hub.snapshot_download import dataset_snapshot_download
from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
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
WEIGHT_GLOBS = ('*.safetensors', '*.bin', '*.pt')


def repo_dir(hf_id: str, repo_type: str) -> Path:
    kind = 'datasets' if repo_type == 'dataset' else 'models'
    return HUB_ROOT / f"{kind}--{hf_id.replace('/', '--')}"


def safetensors_header_ok(path: Path) -> bool | None:
    """True when the header parses, False when the file is malformed.

    None means the file could not be read. On the shared volume that is
    not evidence of a bad shard, so the caller must not delete it.
    """
    try:
        with safe_open(str(path), framework='numpy') as handle:
            list(handle.keys())
    except SafetensorError:
        return False
    except Exception:  # noqa: BLE001 - OSError and the like: cannot judge
        return None
    return True


def corrupt_shards(root: Path) -> list[Path]:
    return [path for path in sorted(root.rglob('*.safetensors'))
            if safetensors_header_ok(path) is False]


def _real_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        if '.cache' in path.parts or path.name in MODELSCOPE_METADATA:
            continue
        files.append(path)
    return files


def snapshot_ready(snap: Path, repo_type: str) -> bool:
    if not snap.is_dir() or not _real_files(snap):
        return False
    if corrupt_shards(snap):
        return False
    if repo_type == 'dataset':
        return True
    if not (snap / 'config.json').is_file():
        return False
    return any(any(snap.rglob(pattern)) for pattern in WEIGHT_GLOBS)


def cached_snapshot(hf_id: str, repo_type: str) -> Path | None:
    refs = repo_dir(hf_id, repo_type) / 'refs' / 'main'
    if not refs.is_file():
        return None
    snap = repo_dir(hf_id, repo_type) / 'snapshots' / refs.read_text().strip()
    if not snapshot_ready(snap, repo_type):
        return None
    return snap


def purge_corrupt(hf_id: str, repo_type: str) -> None:
    """Delete a bad shard and the blob it points at.

    snapshot_download copies a cached blob into the download target
    without reading it again, so a damaged blob is reused until removed.
    """
    snapshots = repo_dir(hf_id, repo_type) / 'snapshots'
    if not snapshots.is_dir():
        return
    for snap in sorted(snapshots.iterdir()):
        for shard in corrupt_shards(snap):
            blob = Path(os.path.realpath(shard)) if shard.is_symlink() else None
            print(f'[purge] {hf_id}: {shard.relative_to(snap).as_posix()}',
                  flush=True)
            shard.unlink(missing_ok=True)
            if blob is not None:
                blob.unlink(missing_ok=True)


def revision(hf_id: str, repo_type: str) -> str:
    try:
        info = (HfApi().dataset_info(hf_id) if repo_type == 'dataset'
                else HfApi().model_info(hf_id))
        return info.sha
    except Exception:  # noqa: BLE001 - Hugging Face may be why we fall back
        refs = repo_dir(hf_id, repo_type) / 'refs' / 'main'
        return refs.read_text().strip() if refs.is_file() else 'modelscope'


def from_huggingface(hf_id: str, repo_type: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=hf_id, repo_type=repo_type, local_dir=str(dest))
    return dest


def from_modelscope(hf_id: str, repo_type: str, dest: Path) -> Path:
    # The public ModelScope copy is addressed by the same repo id. The
    # snapshot is still published under the Hugging Face id below.
    dest.mkdir(parents=True, exist_ok=True)
    if repo_type == 'dataset':
        downloaded = Path(dataset_snapshot_download(hf_id, local_dir=str(dest)))
    else:
        downloaded = Path(ms_snapshot_download(hf_id, local_dir=str(dest)))
    if downloaded.is_relative_to(dest):
        return downloaded
    staged = dest / 'tree'
    shutil.copytree(downloaded, staged, dirs_exist_ok=True)
    return staged


def checked(src: Path) -> Path:
    if not _real_files(src):
        raise RuntimeError(f'empty download: {src}')
    corrupt = corrupt_shards(src)
    if corrupt:
        raise RuntimeError(
            'corrupt safetensors downloaded: '
            f'{[path.relative_to(src).as_posix() for path in corrupt]}')
    return src


def intact(dest: Path, staged: Path) -> bool:
    try:
        if dest.stat().st_size != staged.stat().st_size:
            return False
    except FileNotFoundError:
        return False
    return dest.suffix != '.safetensors' or safetensors_header_ok(dest) is not False


def publish(hf_id: str, repo_type: str, src: Path, sha: str) -> Path:
    """Rename src's files into snapshots/<sha>, then point refs/main at it.

    A file that is already the same size and header stays put: another
    job may be reading it.
    """
    snap = repo_dir(hf_id, repo_type) / 'snapshots' / sha
    for item in _real_files(src):
        dest = snap / item.relative_to(src)
        if intact(dest, item):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(item, dest)
    refs = repo_dir(hf_id, repo_type) / 'refs'
    refs.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            'w', dir=refs, prefix='.main-', delete=False) as handle:
        # No trailing newline: the hub compares this string with the
        # snapshot directory name without stripping it.
        handle.write(sha)
    os.replace(handle.name, refs / 'main')
    return snap


def fetch(hf_id: str, repo_type: str) -> Path:
    sha = revision(hf_id, repo_type)
    staging_root = HUB_ROOT.parent / STAGING_DIR
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f'{repo_dir(hf_id, repo_type).name}-', dir=staging_root))
    try:
        print(f'[download] {hf_id} <- HuggingFace', flush=True)
        try:
            src = checked(from_huggingface(hf_id, repo_type, staging / 'hf'))
        except Exception as exc:  # noqa: BLE001 - ModelScope is the fallback
            print(f'[fallback] {hf_id}: HuggingFace failed '
                  f'({type(exc).__name__}: {exc}); trying ModelScope',
                  flush=True)
            src = checked(from_modelscope(hf_id, repo_type, staging / 'ms'))
        snap = publish(hf_id, repo_type, src, sha)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if not snapshot_ready(snap, repo_type):
        raise RuntimeError(f'{hf_id}: published snapshot {snap} is incomplete')
    print(f'[cached] {hf_id} -> {snap}', flush=True)
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
    stale = lock.with_name(f'{lock.name}.stale-{uuid.uuid4().hex}')
    try:
        os.rename(lock, stale)
    except FileNotFoundError:
        return
    shutil.rmtree(stale, ignore_errors=True)


@contextmanager
def download_lock(hf_id: str, repo_type: str) -> Iterator[None]:
    """Hold this repo's lock. After LOCK_WAIT_SECONDS, run without it."""
    locks = HUB_ROOT.parent / LOCKS_DIR
    locks.mkdir(parents=True, exist_ok=True)
    lock = locks / repo_dir(hf_id, repo_type).name
    deadline = time.monotonic() + LOCK_WAIT_SECONDS
    waiting = False
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            pass
        if time.monotonic() >= deadline:
            print(f'[lock] {hf_id}: still held after {LOCK_WAIT_SECONDS}s, '
                  'downloading without it', flush=True)
            yield
            return
        if not waiting:
            print(f'[lock] {hf_id}: another job is downloading it, waiting',
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


def materialize(hf_id: str, repo_type: str) -> Path:
    """Return the local snapshot, downloading it when the cache misses."""
    if repo_type not in ('model', 'dataset'):
        raise ValueError(f'repo_type must be model or dataset, got {repo_type}')
    sweep_staging()
    snap = cached_snapshot(hf_id, repo_type)
    if snap is None:
        with download_lock(hf_id, repo_type):
            snap = cached_snapshot(hf_id, repo_type)
            if snap is None:
                purge_corrupt(hf_id, repo_type)
                try:
                    snap = fetch(hf_id, repo_type)
                except Exception as exc:  # noqa: BLE001 - reported to the job log
                    raise SystemExit(
                        f'{hf_id}: not in the shared cache and neither '
                        'HuggingFace nor ModelScope delivered it '
                        f'({exc}). This is a download failure, not an '
                        'example failure.'
                    ) from exc
    print(f'[cache] {hf_id} -> {snap}', flush=True)
    return snap
