"""Quick-start-Ascend documentation test for DeepSpeed on Ascend NPU.

Built on top of ``MarkdownDocTestBase`` (shared engine). The document
under test is ``sources/deepspeed/quick_start.md``, which
follows the ``docs/markdown_doc_test_label.md`` contract: every
``shell`` code block carries one of the ``#test`` / ``#test-setup`` /
``#test-result`` labels plus ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` parameters.

Environment variables (injected by the shared engine
``quick-start-template.yml``):
    ``MONITORED_DOC_URL``         Required; raw URL of the document under test.
    ``UPSTREAM_REF``              Required; captured by the hidden
                                  ``#test-setup store="upstream_ref"`` block
                                  in the doc, then loaded into test commands
                                  where ``<UPSTREAM_REF>`` appears.
    ``NPU_READY=true``            Required; gates the E2E test class.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

from doc_test.base import MarkdownDocTestBase


def _is_truthy(value: str | None) -> bool:
    """``'true'`` -> True (case-insensitive); anything else -> False."""
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    """Return True when ``NPU_READY=true`` is set."""
    return _is_truthy(os.environ.get('NPU_READY'))


def _extract_training_script(doc_text: str) -> str:
    """Get the single Python example in the training-script section."""
    _, heading, section = doc_text.partition('### 编写训练脚本')
    if not heading:
        raise AssertionError('missing training-script section')
    section, next_heading, _ = section.partition('### 单卡训练')
    if not next_heading:
        raise AssertionError('missing single-card section after training script')
    blocks = re.findall(
        r'^```python[ \t]*\n(.*?)^```[ \t]*$', section,
        flags=re.MULTILINE | re.DOTALL,
    )
    if len(blocks) != 1:
        raise AssertionError(f'expected one training-script block, got {len(blocks)}')
    script = blocks[0].rstrip('\n') + '\n'
    compile(script, 'train_cifar10.py', 'exec')
    return script


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """``quick_start.md`` end-to-end test: read doc -> validate
    contract -> run ``#test-setup`` / ``#test`` in order -> compare against
    ``#test-result``."""

    DEFAULT_COMMAND_TIMEOUT = 1800
    USER_AGENT = 'cosdt-ci-test/quick-start'
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    def pre_process(self) -> str:
        """Read the local doc instead of fetching from MONITORED_DOC_URL.
        The doc is bundled in the repo, so the test always uses the version
        that ships with the code.
        """
        doc = Path(__file__).resolve().parents[2] / 'sources' / 'deepspeed' / 'quick_start.md'
        text = doc.read_text(encoding='utf-8')
        Path('train_cifar10.py').write_text(
            _extract_training_script(text), encoding='utf-8',
        )
        return text

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env once so later ``bash -c`` blocks inherit it.

        Class-level setup: run once per test class, triggered by
        ``setUpClass``. Each labeled fence is a new subprocess, so a
        ``source set_env.sh`` block in the document does not persist.

        Also pins NPU cards 0-1 (2-card runner; the doc's 2-card
        distributed run needs the launcher to see both devices), chdirs
        to the document directory (``sources/deepspeed/``) so doc relative
        paths resolve correctly, and installs MPI for the launcher.
        """
        path_dirs = '/usr/local/sbin:/usr/local/bin'
        current_path = os.environ.get('PATH', '')
        if path_dirs not in current_path:
            os.environ['PATH'] = f'{path_dirs}:{current_path}'

        if os.path.isfile(cls._CANN_SET_ENV):
            merged = subprocess.run(
                ['bash', '-c', f'source {cls._CANN_SET_ENV} >/dev/null 2>&1; env'],
                capture_output=True, text=True, check=True,
            )
            for line in merged.stdout.splitlines():
                if '=' not in line:
                    continue
                key, _, value = line.partition('=')
                os.environ.setdefault(key, value)
            print('setup: sourced CANN env from set_env.sh')
        else:
            print(
                f'setup: skipping CANN env source ({cls._CANN_SET_ENV} not present)'
            )

        # ASCEND_RT_VISIBLE_DEVICES=0,1: expose both cards for the
        # single-card + 2-card distributed smoke (--num_gpus 2 needs
        # the launcher to see 2 devices).
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0,1'
        print('setup: pinned ASCEND_RT_VISIBLE_DEVICES=0,1')

        # Resolve the document's relative paths from its directory.
        project_root = Path(__file__).resolve().parents[2] / 'sources' / 'deepspeed'
        os.chdir(project_root)
        print(f'setup: cwd -> {project_root}')

        # Install MPI (libopenmpi-dev + mpi4py) for deepspeed command.
        subprocess.run(['apt-get', 'update'], check=True)
        subprocess.run(['apt-get', 'install', '-y', 'libopenmpi-dev'], check=True)
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 'mpi4py'],
            check=True,
        )
        print('setup: installed MPI (libopenmpi-dev + mpi4py)')

        # CIFAR10 download is handled inside the doc's train_cifar10.py
        # (CN mirror fast path + torchvision official fallback).

        # The document installs torch, torch_npu, and torchvision in order.

    @classmethod
    def setUpClass(cls) -> None:
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end tests require NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        self.run_template()


if __name__ == '__main__':
    unittest.main()
