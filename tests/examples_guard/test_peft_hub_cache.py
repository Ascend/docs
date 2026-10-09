"""Tests for examples-guards/peft/scripts/hub_cache.py.

HuggingFace and ModelScope are both mocked, so these run offline.
"""
from __future__ import annotations

import io
import os
import re
import struct
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

REPO = Path(__file__).resolve().parents[2]
PEFT = REPO / 'examples-guards' / 'peft'
sys.path.insert(0, str(PEFT / 'scripts'))

import hub_cache  # noqa: E402

SHA = 'a' * 40
SAFETENSORS_HEADER = b'{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'


def write_file(path: Path, corrupt: bool = False) -> None:
    """A valid safetensors for *.safetensors names, text otherwise;
    corrupt appends a second copy, as two concurrent writers did."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix != '.safetensors':
        path.write_text('x')
        return
    body = struct.pack('<Q', len(SAFETENSORS_HEADER)) + SAFETENSORS_HEADER + bytes(4)
    path.write_bytes(body * 2 if corrupt else body)


class HubCacheTests(unittest.TestCase):

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.github_env = self.tmp / 'github_env'
        self.staging = self.tmp / hub_cache.STAGING_DIR
        self.lock = self.tmp / hub_cache.LOCKS_DIR / 'models--roberta-base'
        patches = [
            mock.patch.object(hub_cache, 'HUB_ROOT', self.tmp / 'hub'),
            mock.patch.dict(os.environ, {'GITHUB_ENV': str(self.github_env)}),
        ]
        self.hf = mock.patch.object(hub_cache, 'snapshot_download').start()
        self.ms = mock.patch.object(hub_cache, 'ms_snapshot_download').start()
        self.sleep = mock.patch.object(hub_cache.time, 'sleep').start()
        api = mock.Mock()
        api.model_info.return_value = SimpleNamespace(sha=SHA)
        patches.append(mock.patch.object(hub_cache, 'HfApi', return_value=api))
        for patch in patches:
            patch.start()
        self.addCleanup(mock.patch.stopall)
        self.asset = hub_cache.ASSETS['ROBERTA_BASE_PATH']
        self.snap = hub_cache.repo_dir(self.asset) / 'snapshots' / SHA

    def _snapshot(self, files: list[str], corrupt: tuple[str, ...] = ()) -> Path:
        repo = hub_cache.repo_dir(self.asset)
        for name in files:
            write_file(self.snap / name, corrupt=name in corrupt)
        (repo / 'refs').mkdir(parents=True, exist_ok=True)
        (repo / 'refs' / 'main').write_text(SHA)
        return self.snap

    def _hf_writes(self, files: list[str], corrupt: tuple[str, ...] = ()):
        def download(**kwargs):
            for name in files:
                write_file(Path(kwargs['local_dir']) / name, corrupt=name in corrupt)
            return kwargs['local_dir']
        self.hf.side_effect = download

    def _ms_writes(self, files: list[str]):
        def download(model_id, cache_dir, **_):
            src = Path(cache_dir) / model_id
            for name in files:
                write_file(src / name)
            return str(src)
        self.ms.side_effect = download

    def _main(self, *argv: str) -> tuple[int, str]:
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            code = hub_cache.main(list(argv))
        return code, err.getvalue()

    def _github_env(self) -> str:
        return self.github_env.read_text() if self.github_env.exists() else ''

    def _assert_resolved_intact(self) -> None:
        self.assertEqual(self._github_env(), f'ROBERTA_BASE_PATH={self.snap}\n')
        self.assertEqual(hub_cache.corrupt_shards(self.snap), [])
        self.assertEqual(list(self.staging.iterdir()), [])

    def test_complete_cache_is_used_without_download(self) -> None:
        self._snapshot(['config.json', 'model.safetensors'])
        self.assertEqual(self._main('ROBERTA_BASE_PATH')[0], 0)
        self.hf.assert_not_called()
        self.ms.assert_not_called()
        self.assertEqual(self._github_env(), f'ROBERTA_BASE_PATH={self.snap}\n')

    def test_incomplete_cache_is_downloaded_through_staging(self) -> None:
        self._snapshot(['config.json'])
        self._hf_writes(['config.json', 'model.safetensors'])
        self.assertEqual(self._main('ROBERTA_BASE_PATH')[0], 0)
        kwargs = self.hf.call_args.kwargs
        self.assertEqual((kwargs['repo_id'], kwargs['allow_patterns']),
                         ('roberta-base', list(self.asset.patterns)))
        self.assertEqual(Path(kwargs['local_dir']).parents[1], self.staging)
        self.assertNotIn('cache_dir', kwargs)
        self.ms.assert_not_called()
        self._assert_resolved_intact()
        self.assertFalse(self.lock.exists())

    def test_corrupt_cached_shard_is_replaced_and_intact_files_kept(self) -> None:
        self._snapshot(['config.json', 'model.safetensors'],
                       corrupt=('model.safetensors',))
        config_inode = (self.snap / 'config.json').stat().st_ino
        self._hf_writes(['config.json', 'model.safetensors'])
        self.assertEqual(self._main('ROBERTA_BASE_PATH')[0], 0)
        self.hf.assert_called_once()
        self._assert_resolved_intact()
        self.assertEqual((self.snap / 'config.json').stat().st_ino, config_inode)

    def test_corrupt_shard_and_its_blob_are_purged_before_download(self) -> None:
        """snapshot_download copies a cached blob forward without
        rechecking it, so the damaged bytes must be gone first."""
        self._snapshot(['config.json'])
        blob = hub_cache.repo_dir(self.asset) / 'blobs' / 'deadbeef'
        write_file(blob.with_suffix('.safetensors'), corrupt=True)
        blob.with_suffix('.safetensors').rename(blob)
        shard = self.snap / 'model.safetensors'
        shard.symlink_to(blob)
        seen: list[bool] = []
        self._hf_writes(['config.json', 'model.safetensors'])
        download = self.hf.side_effect
        self.hf.side_effect = lambda **kw: (seen.append(blob.exists()),
                                            download(**kw))[1]
        self.assertEqual(self._main('ROBERTA_BASE_PATH')[0], 0)
        self.assertEqual(seen, [False], 'blob still present at download time')
        self.ms.assert_not_called()
        self._assert_resolved_intact()

    def test_corrupt_huggingface_download_falls_back_to_modelscope(self) -> None:
        self._hf_writes(['config.json', 'model.safetensors'],
                        corrupt=('model.safetensors',))
        self._ms_writes(['config.json', 'model.safetensors'])
        self.assertEqual(self._main('ROBERTA_BASE_PATH')[0], 0)
        self.ms.assert_called_once()
        self._assert_resolved_intact()

    def test_huggingface_failure_falls_back_to_modelscope(self) -> None:
        self.hf.side_effect = OSError('hub unreachable')
        self._ms_writes(['config.json', 'model.safetensors',
                         *hub_cache.MODELSCOPE_METADATA])
        self.assertEqual(self._main('ROBERTA_BASE_PATH')[0], 0)
        self.assertEqual(self.ms.call_args.args[0], 'AI-ModelScope/roberta-base')
        self.assertEqual(Path(self.ms.call_args.kwargs['cache_dir']).parents[1],
                         self.staging)
        repo = hub_cache.repo_dir(self.asset)
        self.assertEqual((repo / 'refs' / 'main').read_text(), SHA)
        self.assertEqual(sorted(p.name for p in self.snap.iterdir()),
                         ['config.json', 'model.safetensors'])
        self._assert_resolved_intact()

    def test_both_sources_failing_is_reported_as_guard_noise(self) -> None:
        self.hf.side_effect = OSError('hub unreachable')
        self.ms.side_effect = OSError('modelscope unreachable')
        code, err = self._main('ROBERTA_BASE_PATH')
        self.assertEqual(code, 1)
        self.assertIn('guard noise', err)
        self.assertEqual(self._github_env(), '')
        self.assertEqual(list(self.staging.iterdir()), [])
        self.assertFalse(self.lock.exists())

    def test_download_missing_required_files_fails(self) -> None:
        self._hf_writes(['config.json'])
        code, err = self._main('ROBERTA_BASE_PATH')
        self.assertEqual(code, 1)
        self.assertIn('lacks', err)

    def test_waiter_uses_what_the_lock_holder_published(self) -> None:
        self.lock.mkdir(parents=True)

        def holder_finishes(_seconds):
            self._snapshot(['config.json', 'model.safetensors'])
            self.lock.rmdir()

        self.sleep.side_effect = holder_finishes
        self.assertEqual(self._main('ROBERTA_BASE_PATH')[0], 0)
        self.hf.assert_not_called()
        self.ms.assert_not_called()
        self.assertEqual(self._github_env(), f'ROBERTA_BASE_PATH={self.snap}\n')

    def test_stale_lock_is_broken(self) -> None:
        self.lock.mkdir(parents=True)
        idle = time.time() - hub_cache.LOCK_STALE_SECONDS - 60
        os.utime(self.lock, (idle, idle))
        self._hf_writes(['config.json', 'model.safetensors'])
        self.assertEqual(self._main('ROBERTA_BASE_PATH')[0], 0)
        self.hf.assert_called_once()
        self._assert_resolved_intact()
        self.assertEqual(list(self.lock.parent.iterdir()), [])

    def test_lock_wait_gives_up_and_downloads_alongside(self) -> None:
        self.lock.mkdir(parents=True)
        self._hf_writes(['config.json', 'model.safetensors'])
        with mock.patch.object(hub_cache, 'LOCK_WAIT_SECONDS', 0):
            self.assertEqual(self._main('ROBERTA_BASE_PATH')[0], 0)
        self.hf.assert_called_once()
        self._assert_resolved_intact()
        self.assertTrue(self.lock.is_dir())

    def test_old_staging_dirs_are_swept(self) -> None:
        self._snapshot(['config.json', 'model.safetensors'])
        old, fresh = self.staging / 'old', self.staging / 'fresh'
        old.mkdir(parents=True)
        fresh.mkdir()
        killed = time.time() - hub_cache.STAGING_STALE_SECONDS - 60
        os.utime(old, (killed, killed))
        self.assertEqual(self._main('ROBERTA_BASE_PATH')[0], 0)
        self.assertEqual(list(self.staging.iterdir()), [fresh])

    def test_unknown_var_is_rejected(self) -> None:
        self.assertEqual(self._main('NOT_AN_ASSET')[0], 2)
        self.hf.assert_not_called()


class PeftWiringTests(unittest.TestCase):
    """setup, run and the manifest agree with hub_cache.ASSETS."""

    def _resolved_vars(self) -> set[str]:
        text = (PEFT / 'scripts' / 'setup_example.sh').read_text(encoding='utf-8')
        found: set[str] = set()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith('resolve_hub_paths '):
                found.update(stripped.split()[1:])
        return found

    def test_setup_resolves_only_known_assets(self) -> None:
        resolved = self._resolved_vars()
        self.assertTrue(resolved)
        self.assertLessEqual(resolved, set(hub_cache.ASSETS))

    def test_every_overlay_asset_var_is_resolved(self) -> None:
        manifest = yaml.safe_load(
            (PEFT / 'examples_manifest.yaml').read_text(encoding='utf-8'))
        used = {var for entry in manifest['supported']
                for arg in entry.get('overlay_args') or []
                for var in re.findall(r'\$\{(\w+)\}', arg)}
        self.assertEqual(used & set(hub_cache.ASSETS), self._resolved_vars())

    def test_setup_and_run_use_huggingface_directly(self) -> None:
        for script in ('setup_example.sh', 'run_example.sh'):
            with self.subTest(script=script):
                text = (PEFT / 'scripts' / script).read_text(encoding='utf-8')
                self.assertIn('export HF_ENDPOINT=https://huggingface.co', text)


if __name__ == '__main__':
    unittest.main()
