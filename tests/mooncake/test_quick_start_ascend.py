"""Quick-start test: doc under test is ``sources/mooncake/quick_start.md``.

Run: ``python -m unittest tests.mooncake.test_quick_start_ascend -v 2>&1``

Env (injected by the engine ``quick-start-template.yml``, triggered by
``mooncake-quick-start.yml``): ``MONITORED_DOC_URL``, ``UPSTREAM_REF``,
``NPU_READY=true`` (otherwise the class is skipped).
"""

from __future__ import annotations

import os
import subprocess
import unittest

from doc_test.base import MarkdownDocTestBase


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

    # cmake + Ascend Direct compile can take over an hour; the base
    # class uses one timeout for every subprocess.
    DEFAULT_COMMAND_TIMEOUT = 7200
    USER_AGENT = 'ascend-docs/quick-start'
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'Failed to initialize ACL',
        'Failed to set device ACL',
        'getTransferStatus FAILED',
        'Failed to install Ascend transport',
    )

    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'
    _APT_CACHE_HOST = 'cache-service.nginx-pypi-cache.svc.cluster.local'
    _PRESERVE_ENV = frozenset({
        'NPU_READY',
        'MONITORED_DOC_URL',
        'UPSTREAM_REF',
        'GITHUB_TOKEN',
        'GH_TOKEN',
        'GITHUB_ENV',
        'GITHUB_OUTPUT',
        'GITHUB_PATH',
    })

    @classmethod
    def _configure_apt_cache(cls) -> None:
        """Point apt at the cluster nginx cache when that DNS name resolves.

        Hong Kong A2 runners are inside the cluster. coder / a laptop
        are not; leave the image mirrors alone there.
        """
        probe = subprocess.run(
            ['getent', 'hosts', cls._APT_CACHE_HOST],
            capture_output=True,
            check=False,
        )
        if probe.returncode != 0:
            print('setup: cluster APT cache DNS not available, keeping default mirrors')
            return
        sources = '/etc/apt/sources.list'
        if not os.path.isfile(sources):
            print(f'setup: {sources} missing, skipping APT cache rewrite')
            return
        subprocess.run(
            [
                'sed',
                '-Ei',
                f's@(ports|archive).ubuntu.com@{cls._APT_CACHE_HOST}:8081@g',
                sources,
            ],
            check=True,
        )
        print('setup: pointed apt at cluster cache')

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env once so later ``bash -c`` blocks inherit it.

        Class-level setup: run once per test class, triggered by
        ``setUpClass``. Each labeled fence is a new subprocess, so a
        ``source set_env.sh`` block in the document does not persist.

        Merge is overwrite, not ``setdefault``: the container image may
        already ship ``LD_LIBRARY_PATH``, which would otherwise hide the
        CANN increment from ``set_env.sh``. CI-injected names in
        ``_PRESERVE_ENV`` stay.
        """
        cls._configure_apt_cache()

        if os.path.isfile(cls._CANN_SET_ENV):
            merged = subprocess.run(
                ['bash', '-c', f'source {cls._CANN_SET_ENV} >/dev/null 2>&1; env'],
                capture_output=True, text=True, check=True,
            )
            for line in merged.stdout.splitlines():
                if '=' not in line:
                    continue
                key, _, value = line.partition('=')
                if key in cls._PRESERVE_ENV:
                    continue
                os.environ[key] = value
            print('setup: sourced CANN env from set_env.sh')
        else:
            print(
                f'setup: skipping CANN env source ({cls._CANN_SET_ENV} not present)'
            )

        path_dirs = '/usr/local/sbin:/usr/local/bin'
        current_path = os.environ.get('PATH', '')
        if path_dirs not in current_path:
            os.environ['PATH'] = f'{path_dirs}:{current_path}'

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
