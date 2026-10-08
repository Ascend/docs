"""Tests for examples-guards/peft/scripts/hub_cache.py.

HuggingFace and ModelScope are both mocked, so these run offline.
"""
from __future__ import annotations

import io
import os
import re
import sys
import tempfile
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


class HubCacheTests(unittest.TestCase):

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.github_env = self.tmp / 'github_env'
        patches = [
            mock.patch.object(hub_cache, 'HUB_ROOT', self.tmp / 'hub'),
            mock.patch.dict(os.environ, {'GITHUB_ENV': str(self.github_env)}),
        ]
        self.hf = mock.patch.object(hub_cache, 'snapshot_download').start()
        self.ms = mock.patch.object(hub_cache, 'ms_snapshot_download').start()
        api = mock.Mock()
        api.model_info.return_value = SimpleNamespace(sha=SHA)
        patches.append(mock.patch.object(hub_cache, 'HfApi', return_value=api))
        for patch in patches:
            patch.start()
        self.addCleanup(mock.patch.stopall)
        self.asset = hub_cache.ASSETS['ROBERTA_BASE_PATH']

    def _snapshot(self, files: list[str]) -> Path:
        repo = hub_cache.repo_dir(self.asset)
        snap = repo / 'snapshots' / SHA
        for name in files:
            (snap / name).parent.mkdir(parents=True, exist_ok=True)
            (snap / name).write_text('x')
        (repo / 'refs').mkdir(parents=True, exist_ok=True)
        (repo / 'refs' / 'main').write_text(SHA)
        return snap

    def _main(self, *argv: str) -> tuple[int, str]:
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            code = hub_cache.main(list(argv))
        return code, err.getvalue()

    def _github_env(self) -> str:
        return self.github_env.read_text() if self.github_env.exists() else ''

    def test_complete_cache_is_used_without_download(self) -> None:
        snap = self._snapshot(['config.json', 'model.safetensors'])
        self.assertEqual(self._main('ROBERTA_BASE_PATH')[0], 0)
        self.hf.assert_not_called()
        self.ms.assert_not_called()
        self.assertEqual(self._github_env(), f'ROBERTA_BASE_PATH={snap}\n')

    def test_incomplete_cache_is_downloaded_from_huggingface(self) -> None:
        self._snapshot(['config.json'])

        def fetch(**kwargs):
            return str(self._snapshot(['config.json', 'model.safetensors']))

        self.hf.side_effect = fetch
        self.assertEqual(self._main('ROBERTA_BASE_PATH')[0], 0)
        self.hf.assert_called_once_with(
            repo_id='roberta-base', allow_patterns=list(self.asset.patterns),
            cache_dir=str(self.tmp / 'hub'))
        self.ms.assert_not_called()

    def test_huggingface_failure_falls_back_to_modelscope(self) -> None:
        src = self.tmp / 'ms'
        src.mkdir()
        for name in ('config.json', 'model.safetensors', *hub_cache.MODELSCOPE_METADATA):
            (src / name).write_text('x')
        self.hf.side_effect = OSError('hub unreachable')
        self.ms.return_value = str(src)
        self.assertEqual(self._main('ROBERTA_BASE_PATH')[0], 0)
        self.assertEqual(self.ms.call_args.args[0], 'AI-ModelScope/roberta-base')
        repo = hub_cache.repo_dir(self.asset)
        self.assertEqual((repo / 'refs' / 'main').read_text(), SHA)
        snap = repo / 'snapshots' / SHA
        self.assertEqual(sorted(p.name for p in snap.iterdir()),
                         ['config.json', 'model.safetensors'])
        self.assertEqual(self._github_env(), f'ROBERTA_BASE_PATH={snap}\n')

    def test_both_sources_failing_is_reported_as_guard_noise(self) -> None:
        self.hf.side_effect = OSError('hub unreachable')
        self.ms.side_effect = OSError('modelscope unreachable')
        code, err = self._main('ROBERTA_BASE_PATH')
        self.assertEqual(code, 1)
        self.assertIn('guard noise', err)
        self.assertEqual(self._github_env(), '')

    def test_download_missing_required_files_fails(self) -> None:
        self.hf.side_effect = lambda **_: str(self._snapshot(['config.json']))
        code, err = self._main('ROBERTA_BASE_PATH')
        self.assertEqual(code, 1)
        self.assertIn('lacks', err)

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
