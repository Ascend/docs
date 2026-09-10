"""Quick-start test: doc under test is ``sources/flagscale/quick_start.md``.

Run: ``python -m unittest tests.flagscale.test_quick_start_ascend -v 2>&1``

Env (injected by the engine ``quick-start-template.yml``, triggered by
``flagscale-quick-start.yml``): ``MONITORED_DOC_URL``, ``UPSTREAM_REF``,
``NPU_READY=true`` (otherwise the class is skipped).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import unittest
from pathlib import Path

from doc_test.base import MarkdownDocTestBase
from doc_test.model_cache import (
    ensure_safetensors,
    purge_huggingface_corrupt,
    report_huggingface_state,
    resolve_huggingface_cache,
)

_CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'
_ATB_SET_ENV = '/usr/local/Ascend/nnal/atb/latest/atb/set_env.sh'


def _is_truthy(value: str | None) -> bool:
    """``'true'`` -> True (case-insensitive); anything else (including unset) -> False."""
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    """Return True when ``NPU_READY=true`` is set, releasing the skip."""
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """End-to-end test: fetch doc -> validate contract -> run ``#test-setup``
    / ``#test`` in order -> compare against ``#test-result``."""

    DEFAULT_COMMAND_TIMEOUT = 7200
    USER_AGENT = 'ascend-docs/quick-start'
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',  # typo in CANN's Python driver (sic)
        'HostRegisterError',
        'ERR99999',
    )

    _MODEL_ID = 'Qwen/Qwen2.5-0.5B'

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN / ATB once so later ``bash -c`` blocks inherit them.

        Class-level setup: run once per test class, triggered by
        ``setUpClass``. Each labeled fence is a new subprocess, so a
        ``source set_env.sh`` block in the document does not persist.

        Merge is overwrite, not ``setdefault``: the container image may
        already ship ``LD_LIBRARY_PATH``, which would otherwise hide the
        CANN increment from ``set_env.sh``.
        """
        venv_prefix = ''
        virtual_env = os.environ.get('VIRTUAL_ENV')
        if virtual_env:
            venv_prefix = str(Path(virtual_env) / 'bin')

        missing = [p for p in (_CANN_SET_ENV, _ATB_SET_ENV) if not os.path.isfile(p)]
        if missing:
            raise RuntimeError(
                'required Ascend env scripts missing: ' + ', '.join(missing)
            )
        sourced = ' && '.join(
            f'set +u && source {shlex.quote(script)} >/dev/null 2>&1'
            for script in (_CANN_SET_ENV, _ATB_SET_ENV)
        )
        merged = subprocess.run(
            ['bash', '-c', f'{sourced}; env'],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in merged.stdout.splitlines():
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            os.environ[key] = value
        extras = []
        for extra in ('/usr/local/sbin', '/usr/local/bin', '/usr/sbin'):
            parts = os.environ.get('PATH', '').split(':')
            if extra not in parts:
                extras.append(extra)
        if extras:
            os.environ['PATH'] = os.environ.get('PATH', '') + ':' + ':'.join(extras)
        if venv_prefix:
            current = os.environ.get('PATH', '')
            if not current.startswith(venv_prefix + ':'):
                os.environ['PATH'] = f'{venv_prefix}:{current}'
        print('setup: sourced CANN and ATB env')

        ensure_safetensors()
        report_huggingface_state(cls._MODEL_ID)
        purge_huggingface_corrupt(resolve_huggingface_cache())

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per class. ``@unittest.skipIf`` only skips
        the test *method* — ``setUpClass`` itself always runs, so the
        ``if _e2e_enabled()`` guard keeps heavy setup from firing when
        ``NPU_READY`` is unset.
        """
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end requires NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        """Run the full pre_process -> parse -> execute -> post_process flow."""

        self.run_template()


if __name__ == '__main__':
    unittest.main()
