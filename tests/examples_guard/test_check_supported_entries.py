"""Tests for scripts/check_supported_entries.py (engine manifest check)."""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))

import check_supported_entries  # noqa: E402

VALID_ENTRY = {
    'path': 'examples/sft/run.sh',
    'profile': 'p',
    'runner': 'linux-aarch64-a2-1',
    'image': 'img:tag',
    'overlay_args': ['--max_steps 1'],
    'timeout_minutes': 90,
}


class ValidateTests(unittest.TestCase):
    """validate() against a temp target tree."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.target = Path(self._tmp.name) / 'target'
        (self.target / 'examples' / 'sft').mkdir(parents=True)
        (self.target / 'examples' / 'sft' / 'run.sh').write_text('#!/bin/sh\n')
        # An unclassified new file: the check must ignore it.
        (self.target / 'examples' / 'brand_new').mkdir()
        (self.target / 'examples' / 'brand_new' / 'new.py').write_text('')

    def _validate(self, supported: list) -> tuple[list, list]:
        return check_supported_entries.validate(
            supported, self.target, self.target)

    def test_valid_entry_passes_and_ignores_unclassified(self) -> None:
        entries, errors = self._validate([VALID_ENTRY])
        self.assertEqual(errors, [])
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['path'], 'examples/sft/run.sh')
        self.assertEqual(entries[0]['overlay_args'], ['--max_steps 1'])
        self.assertEqual(entries[0]['name'], 'examples/sft/run')

    def test_missing_supported_fails(self) -> None:
        (self.target / 'examples' / 'sft' / 'run.sh').unlink()
        _, errors = self._validate([VALID_ENTRY])
        self.assertTrue(
            any('missing from target tree: examples/sft/run.sh' in e
                for e in errors))

    def test_bad_overlay_args_fails(self) -> None:
        entry = dict(VALID_ENTRY, overlay_args=[512])
        _, errors = self._validate([entry])
        self.assertTrue(
            any('overlay_args must be a list of non-empty strings' in e
                for e in errors))

    def test_launcher_field_passes_through(self) -> None:
        entry = dict(VALID_ENTRY, launcher='accelerate-deepspeed')
        entries, errors = self._validate([entry])
        self.assertEqual(errors, [])
        self.assertEqual(entries[0]['launcher'], 'accelerate-deepspeed')

    def test_bad_launcher_fails(self) -> None:
        entry = dict(VALID_ENTRY, launcher=512)
        _, errors = self._validate([entry])
        self.assertTrue(any('launcher must be a non-empty string' in e
                            for e in errors))

    def test_missing_required_field_fails(self) -> None:
        entry = {k: v for k, v in VALID_ENTRY.items() if k != 'image'}
        _, errors = self._validate([entry])
        self.assertTrue(any("missing required field(s): ['image']" in e
                            for e in errors))

    def test_mixed_sources_keep_relative_paths_in_job_names(self) -> None:
        project_case = self.target / 'example' / 'run.py'
        project_case.parent.mkdir()
        project_case.write_text('def test_npu(): pass\n')
        entry = {
            'path': 'example/run.py',
            'source': 'project',
            'profile': 'npu',
            'runner': 'linux-aarch64-a2-1',
            'image': 'img:tag',
            'timeout_minutes': 60,
        }
        entries, errors = self._validate([VALID_ENTRY, entry])
        self.assertEqual(errors, [])
        self.assertEqual(entries[0]['name'], 'examples/sft/run')
        self.assertEqual(entries[1]['name'], 'example/run')
        self.assertEqual(entries[1]['source'], 'project')

    def test_missing_project_case_fails(self) -> None:
        entry = {
            'path': 'example/missing.py',
            'source': 'project',
            'profile': 'npu',
            'runner': 'linux-aarch64-a2-1',
            'image': 'img:tag',
            'timeout_minutes': 60,
        }
        _, errors = self._validate([entry])
        self.assertTrue(any('supported project example missing' in e
                            for e in errors))

    def test_project_case_path_cannot_escape_manifest_directory(self) -> None:
        entry = {
            'path': '../outside.py',
            'source': 'project',
            'profile': 'npu',
            'runner': 'linux-aarch64-a2-1',
            'image': 'img:tag',
            'timeout_minutes': 60,
        }
        _, errors = self._validate([entry])
        self.assertTrue(any('path must be relative' in e for e in errors))


class WriteGithubOutputTests(unittest.TestCase):

    def _outputs(self, entries: list) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'github_output'
            with mock.patch.dict(os.environ, {'GITHUB_OUTPUT': str(out)}):
                check_supported_entries.write_github_output(entries)
            raw = out.read_text(encoding='utf-8')
        result: dict = {}
        payload = raw.split('supported_matrix<<EOF\n')[1].split('\nEOF\n')[0]
        result['supported_matrix'] = json.loads(payload)
        result['has_supported'] = raw.split('has_supported=')[1].splitlines()[0]
        return result

    def test_empty_supported_writes_false(self) -> None:
        outputs = self._outputs([])
        self.assertEqual(outputs['has_supported'], 'false')
        self.assertEqual(outputs['supported_matrix'], [])

    def test_entries_serialize_as_matrix(self) -> None:
        outputs = self._outputs([dict(VALID_ENTRY)])
        self.assertEqual(outputs['has_supported'], 'true')
        self.assertEqual(outputs['supported_matrix'][0]['path'],
                         'examples/sft/run.sh')

    def test_unset_output_is_a_no_op(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != 'GITHUB_OUTPUT'}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, env, clear=True):
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                check_supported_entries.write_github_output([])
            finally:
                os.chdir(cwd)
            self.assertEqual(os.listdir(tmp), [])


class MainTests(unittest.TestCase):
    """main() end to end: yaml manifest on disk, GITHUB_OUTPUT file."""

    MANIFEST = """\
version: 1
scan:
  root: examples
  include_extensions: ['.sh', '.py']
supported:
  - path: examples/sft/run.sh
    profile: p
    runner: linux-aarch64-a2-1
    image: img:tag
    overlay_args: ['--max_steps 1']
    timeout_minutes: 90
"""

    def _run_main(self, manifest_text: str) -> tuple[int, dict, str]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'target'
            (target / 'examples' / 'sft').mkdir(parents=True)
            (target / 'examples' / 'sft' / 'run.sh').write_text('#!/bin/sh\n')
            manifest = root / 'examples_manifest.yaml'
            manifest.write_text(manifest_text, encoding='utf-8')
            out = root / 'github_output'
            argv = [sys.argv[0], '--target-root', str(target),
                    '--manifest', str(manifest)]
            stderr = io.StringIO()
            code = 0
            with mock.patch.dict(os.environ,
                                 {'GITHUB_OUTPUT': str(out)}), \
                    mock.patch.object(sys, 'argv', argv), \
                    redirect_stderr(stderr):
                try:
                    check_supported_entries.main()
                except SystemExit as exc:
                    code = exc.code
            outputs: dict = {}
            if out.exists():
                raw = out.read_text(encoding='utf-8')
                if 'supported_matrix<<EOF\n' in raw:
                    payload = raw.split(
                        'supported_matrix<<EOF\n')[1].split('\nEOF\n')[0]
                    outputs['supported_matrix'] = json.loads(payload)
                if 'has_supported=' in raw:
                    outputs['has_supported'] = raw.split(
                        'has_supported=')[1].splitlines()[0]
            return code, outputs, stderr.getvalue()

    def test_manifest_ok(self) -> None:
        code, outputs, stderr = self._run_main(self.MANIFEST)
        self.assertEqual(code, 0)
        self.assertEqual(outputs['has_supported'], 'true')
        self.assertEqual(len(outputs['supported_matrix']), 1)

    def test_manifest_error_exits_1(self) -> None:
        broken = self.MANIFEST.replace("    image: img:tag\n", '')
        code, outputs, stderr = self._run_main(broken)
        self.assertEqual(code, 1)
        self.assertIn("missing required field(s): ['image']", stderr)
        # The invalid entry is rejected from the matrix, but the
        # outputs are still written before the exit.
        self.assertEqual(outputs['has_supported'], 'false')


if __name__ == '__main__':
    unittest.main()
