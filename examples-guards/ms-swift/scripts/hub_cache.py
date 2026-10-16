#!/usr/bin/env python3
"""Put the model an ms-swift example loads into the shared HF hub cache.

usage: hub_cache.py <example-script>

The example is run unmodified, so it names its model itself, as
`--model <org>/<name>`. This reads that id from the script and makes sure
the snapshot ms-swift will ask for is already in the cache. The run step
sets USE_HF=1, so ms-swift resolves the id against the cache and no rank
downloads weights. Without this, every rank reaches the weights at the
same moment and a slow download outlasts the collective timeout: in run
36736459261 of the source repo, the 35B GRPO leg died in a barrier with an
HCCL socket timeout, shards still downloading at 6 to 8 MB/s.

- a cached snapshot with every shard of its safetensors index and no
  corrupt safetensors is used as is, without network access;
- otherwise the model is downloaded, HuggingFace first and ModelScope as
  the fallback, and published into the cache, so the next run hits it.

The cache root is HF_HOME (default ~/.cache/huggingface), which on the
self-hosted runners lives on the runner pool's persistent NFS volume,
shared by every concurrent matrix leg on every pod. The hub's own .locks
do not exclude across pods there, and a torn write is not caught by a
size check. So nothing is downloaded into the cache in place. A job
downloads into its own staging dir on the same volume, checks it, and
renames the files into the snapshot, which readers then see whole or
not at all. A mkdir lock per model lets one leg download while the others
wait for it; correctness does not depend on the lock.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from modelscope import snapshot_download as ms_snapshot_download
from safetensors import SafetensorError, safe_open

HUB_ROOT = Path(os.environ.get(
    'HF_HOME', os.path.expanduser('~/.cache/huggingface'))) / 'hub'
# Siblings of HUB_ROOT on the same volume, so a rename from staging into
# the snapshot is atomic, kept out of the hub's own cache layout.
STAGING_DIR = 'guard-staging'
LOCKS_DIR = 'guard-locks'
# The files ms-swift itself skips when it downloads a model
# (swift.utils.hub_utils.safe_snapshot_download). Fetching the same set
# means its later snapshot_download call finds everything it asks for.
IGNORE_PATTERNS = [
    '*.zip', '*.gguf', '*.pth', '*.pt', 'consolidated*', 'onnx/*',
    '*.safetensors.md', '*.msgpack', '*.onnx', '*.ot', '*.h5']
# ModelScope's own snapshot bookkeeping, not repository content.
MODELSCOPE_METADATA = {'.mdl', '.msc', '.mv'}
MODEL_FLAG = re.compile(r"""--model\s+(?:'([^']+)'|"([^"]+)"|([^\s'"\\]+))""")
MODEL_ID = re.compile(r'^[\w.-]+/[\w.-]+$')
HEARTBEAT_SECONDS = 30
# A lock this long without a heartbeat belongs to a killed job.
LOCK_STALE_SECONDS = 300
LOCK_POLL_SECONDS = 10
# The weights run to hundreds of GB, so a waiter gives the holder most of
# a job's timeout before it downloads alongside.
LOCK_WAIT_SECONDS = 5400
# Longer than any job lives: the job that made it was killed.
STAGING_STALE_SECONDS = 24 * 3600


def model_id_from_example(script: Path) -> str:
    match = MODEL_FLAG.search(script.read_text(encoding='utf-8'))
    if match is None:
        raise SystemExit(f'{script}: no --model flag found')
    model_id = next(group for group in match.groups() if group)
    if not MODEL_ID.match(model_id):
        raise SystemExit(f'{script}: --model {model_id!r} is not a hub id')
    return model_id


def repo_dir(model_id: str) -> Path:
    return HUB_ROOT / f"models--{model_id.replace('/', '--')}"


def safetensors_header_ok(path: Path) -> bool | None:
    """True if the header parses and fits the file, False if the file is
    readable but malformed or truncated, None if it cannot be read at all.

    A read failure on the shared NFS volume is not evidence of
    corruption, so None must not get a shard deleted.
    """
    try:
        with safe_open(str(path), framework='numpy') as f:
            list(f.keys())
    except SafetensorError:
        return False
    except Exception:  # noqa: BLE001 - OSError and the like: cannot judge
        return None
    return True


def corrupt_shards(root: Path) -> list[Path]:
    """Safetensors under root whose header or size is inconsistent."""
    return [p for p in sorted(root.rglob('*.safetensors'))
            if safetensors_header_ok(p) is False]


def snapshot_complete(snap: Path) -> bool:
    """config.json plus every weight file the snapshot declares."""
    if not (snap / 'config.json').is_file():
        return False
    index = snap / 'model.safetensors.index.json'
    if index.is_file():
        shards = set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values())
        return all((snap / shard).is_file() for shard in shards)
    return any(p.is_file() for p in snap.glob('*.safetensors'))


def cached_snapshot(model_id: str) -> Path | None:
    """The snapshot refs/main points at, if it is complete and none of its
    safetensors is corrupt."""
    refs = repo_dir(model_id) / 'refs' / 'main'
    if not refs.is_file():
        return None
    snap = repo_dir(model_id) / 'snapshots' / refs.read_text().strip()
    if not snap.is_dir() or not snapshot_complete(snap):
        return None
    corrupt = corrupt_shards(snap)
    if corrupt:
        print(f'[corrupt] {model_id}: '
              f'{[p.relative_to(snap).as_posix() for p in corrupt]}', flush=True)
        return None
    return snap


def purge_corrupt(model_id: str) -> None:
    """Delete corrupt shards and the blobs behind them.

    snapshot_download copies a blob the cache already holds into the
    download target without rechecking it, so a damaged blob is handed
    forward on every retry until it is gone.
    """
    snapshots = repo_dir(model_id) / 'snapshots'
    if not snapshots.is_dir():
        return
    for snap in sorted(snapshots.iterdir()):
        for shard in corrupt_shards(snap):
            blob = Path(os.path.realpath(shard)) if shard.is_symlink() else None
            print(f'[purge] {model_id}: {shard.relative_to(snap).as_posix()}',
                  flush=True)
            shard.unlink(missing_ok=True)
            if blob is not None:
                blob.unlink(missing_ok=True)


def revision(model_id: str) -> str:
    """The hub commit to file the snapshot under."""
    try:
        return HfApi().model_info(model_id).sha
    except Exception:  # noqa: BLE001 - HuggingFace may be the reason we fall back
        refs = repo_dir(model_id) / 'refs' / 'main'
        return refs.read_text().strip() if refs.is_file() else 'modelscope'


def from_huggingface(model_id: str, dest: Path) -> Path:
    snapshot_download(
        repo_id=model_id, ignore_patterns=IGNORE_PATTERNS, local_dir=str(dest))
    return dest


def from_modelscope(model_id: str, dest: Path) -> Path:
    # snapshot_download does not create its cache root: it then fails
    # mid-transfer.
    dest.mkdir(parents=True, exist_ok=True)
    return Path(ms_snapshot_download(
        model_id, cache_dir=str(dest), ignore_patterns=IGNORE_PATTERNS))


def checked(src: Path) -> Path:
    corrupt = corrupt_shards(src)
    if corrupt:
        raise RuntimeError('corrupt safetensors downloaded: '
                           f'{[p.relative_to(src).as_posix() for p in corrupt]}')
    if not snapshot_complete(src):
        raise RuntimeError('download is missing config.json or a weight shard')
    return src


def intact(dest: Path, staged: Path) -> bool:
    """Whether dest already holds staged's content, by size and header."""
    try:
        if dest.stat().st_size != staged.stat().st_size:
            return False
    except FileNotFoundError:
        return False
    return dest.suffix != '.safetensors' or safetensors_header_ok(dest) is not False


def publish(model_id: str, src: Path, sha: str) -> Path:
    """Rename src's files into snapshots/<sha>, then point refs/main at it.

    Intact files are left in place: another job may be reading them.
    """
    snap = repo_dir(model_id) / 'snapshots' / sha
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
    refs = repo_dir(model_id) / 'refs'
    refs.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            'w', dir=refs, prefix='.main-', delete=False) as fh:
        # No trailing newline: the hub compares this string with the
        # snapshot folder name without stripping it.
        fh.write(sha)
    os.replace(fh.name, refs / 'main')
    return snap


def fetch(model_id: str) -> Path:
    """Download into a staging dir of this job, check it, publish it."""
    sha = revision(model_id)
    staging_root = HUB_ROOT.parent / STAGING_DIR
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f'{repo_dir(model_id).name}-', dir=staging_root))
    try:
        print(f'[download] {model_id} <- HuggingFace', flush=True)
        try:
            src = checked(from_huggingface(model_id, staging / 'hf'))
        except Exception as exc:  # noqa: BLE001 - falls back below
            print(f'[fallback] {model_id}: HuggingFace failed '
                  f'({type(exc).__name__}: {exc}); trying ModelScope', flush=True)
            src = checked(from_modelscope(model_id, staging / 'ms'))
        snap = publish(model_id, src, sha)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if cached_snapshot(model_id) != snap:
        raise RuntimeError(f'{model_id}: published snapshot {snap} is not complete')
    print(f'[cached] {model_id} -> {snap}', flush=True)
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
def download_lock(model_id: str) -> Iterator[None]:
    """Hold the model's lock for the body; after LOCK_WAIT_SECONDS of
    waiting, run the body without it."""
    locks = HUB_ROOT.parent / LOCKS_DIR
    locks.mkdir(parents=True, exist_ok=True)
    lock = locks / repo_dir(model_id).name
    deadline = time.monotonic() + LOCK_WAIT_SECONDS
    waiting = False
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            pass
        if time.monotonic() >= deadline:
            print(f'[lock] {model_id}: still held after {LOCK_WAIT_SECONDS}s, '
                  'downloading without it', flush=True)
            yield
            return
        if not waiting:
            print(f'[lock] {model_id}: another job is downloading it, waiting',
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


def resolve(model_id: str) -> Path:
    snap = cached_snapshot(model_id)
    if snap is None:
        with download_lock(model_id):
            # The previous holder may have published it meanwhile.
            snap = cached_snapshot(model_id)
            if snap is None:
                purge_corrupt(model_id)
                return fetch(model_id)
    print(f'[cache] {model_id} -> {snap}', flush=True)
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
    if len(argv) != 1:
        print(f'usage: {sys.argv[0]} <example-script>', file=sys.stderr)
        return 2
    model_id = model_id_from_example(Path(argv[0]))
    sweep_staging()
    try:
        resolve(model_id)
    except Exception as exc:  # noqa: BLE001 - reported as guard noise
        print(f'{model_id}: not in the shared cache and neither HuggingFace '
              f'nor ModelScope delivered it ({exc}) - guard noise, not an '
              'example failure', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
