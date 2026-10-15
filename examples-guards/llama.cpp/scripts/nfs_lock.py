#!/usr/bin/env python3
"""mkdir lock for the shared runner cache.

fcntl locks do not exclude jobs on different pods of this NFS volume.
mkdir is atomic across those pods. A heartbeat refreshes the lock mtime
so a waiter can delete a lock left behind by a killed job.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

HEARTBEAT_SECONDS = 30
LOCK_STALE_SECONDS = 300
LOCK_POLL_SECONDS = 10
LOCK_WAIT_SECONDS = 900
STAGING_STALE_SECONDS = 24 * 3600


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
def download_lock(lock_dir: Path, name: str) -> Iterator[None]:
    """Hold lock_dir/name for the body.

    After LOCK_WAIT_SECONDS the body runs without the lock, so a stuck
    holder cannot consume the whole job timeout.
    """
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_dir / name
    deadline = time.monotonic() + LOCK_WAIT_SECONDS
    waiting = False
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            pass
        if time.monotonic() >= deadline:
            print(f'[lock] {name}: still held after {LOCK_WAIT_SECONDS}s, '
                  'continuing without it', flush=True)
            yield
            return
        if not waiting:
            print(f'[lock] {name}: another job is downloading, waiting',
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


def sweep_staging(staging_root: Path) -> None:
    """Remove staging dirs left by killed jobs."""
    if not staging_root.is_dir():
        return
    cutoff = time.time() - STAGING_STALE_SECONDS
    for entry in staging_root.iterdir():
        try:
            if entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            continue
