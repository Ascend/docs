"""Quick-start-Ascend test: doc under test is ``sources/accelerate/quick_start.md``.

The doc follows the markdown doc-test contract: every ``shell`` code block
carries one of the ``#test`` / ``#test-setup`` / ``#test-result`` labels plus
``id=`` / ``store=`` / ``load='x>>y'`` / ``fuzzy='xxx'`` parameters.

Run: ``python -m unittest tests.accelerate.test_quick_start_ascend -v 2>&1``

Environment variables (injected by the GitHub workflow
``accelerate-quick-start.yml``):
    ``MONITORED_DOC_URL``   Required; raw URL of the document under test.
    ``UPSTREAM_REF``        Required; bash reads ``$UPSTREAM_REF`` to get the
                            latest release tag. The value is captured into
                            ``captures`` via the ``#test-setup
                            store="upstream_ref"`` block's stdout, then
                            substituted where ``<ref>`` appears.
    ``NPU_READY=true``      Required, otherwise the class is skipped. The
                            end-to-end run needs a real ``/dev/davinci*``
                            device; elsewhere ``import torch_npu`` would fail.
"""

from __future__ import annotations

import os
import subprocess
import unittest

# Dual-NPU test runner (``a2-2``): expose both mounted devices so the
# cross-process ``gather_for_metrics`` #test can run a real 2-rank
# all_gather. ``setdefault`` lets an outer test harness override
# (e.g. for debugging on a single-card host).
os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '0,1')

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
    """``quick_start.md`` end-to-end test: fetch doc -> validate contract ->
    run ``#test-setup`` / ``#test`` in order -> compare against ``#test-result``.

    The test subclass owns no ``test_*`` method beyond the template entry;
    the doc body is the spec. ``prepare_environment`` makes sure ``torch_npu``
    and the CUDA-free pip environment are in place before the framework starts
    executing doc commands — Accelerate itself is installed by the doc body.
    """

    # accelerate's smallest meaningful training command (a 3-step toy SGD on
    # a 64-sample TensorDataset) finishes in well under a minute on a single
    # 910B; 15 minutes leaves room for `git clone` of accelerate + `uv pip
    # install -e .` + a cold `accelerate launch` first-time setup.
    DEFAULT_COMMAND_TIMEOUT = 900

    USER_AGENT = 'cosdt-ci-test/quick-start'  # monitored source lives under this org

    # Extend the base ERROR_MARKERS with CANN's typo + sentinel so a CANN
    # failure surfaces a full stderr dump (head/tail by default would hide
    # the line that names the failure).
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,  # generic [ERROR] + Traceback
        'applicaiton exception',  # typo in CANN's Python driver (sic)
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    # Process-level CUDA exclusion list: written to /tmp and exported, so
    # subprocesses (subprocess.run inherits parent env by default) see it.
    # Accelerate's default extras (`deepspeed`, `rich`) sometimes pull CUDA
    # wheels transitively.
    _CUDA_CONSTRAINTS = (
        'cuda-toolkit<0',
        'cuda-python<0',
        'cuda-bindings<0',
        'cuda-core<0',
        'cuda-pathfinder<0',
        'flashinfer-python<0',
        'nvidia-cublas<0',
        'nvidia-cuda-runtime<0',
        'nvidia-cuda-nvrtc<0',
        'nvidia-cuda-cupti<0',
        'nvidia-cudnn<0',
        'nvidia-cudnn-frontend<0',
        'nvidia-cufft<0',
        'nvidia-curand<0',
        'nvidia-cusolver<0',
        'nvidia-cusparse<0',
        'nvidia-cutlass-dsl<0',
        'nvidia-cutlass-dsl-libs-base<0',
        'nvidia-cutlass-dsl-libs-core<0',
        'nvidia-cutlass-dsl-libs-cu12<0',
        'nvidia-ml-py<0',
        'nvidia-nccl<0',
        'nvidia-nvjitlink<0',
        'nvidia-nvtx<0',
        'nvidia-cublas-cu12<0',
        'nvidia-cuda-nvdisasm<0',
        'nvidia-cuda-runtime-cu12<0',
        'nvidia-cuda-nvrtc-cu12<0',
        'nvidia-cuda-cupti-cu12<0',
        'nvidia-cudnn-cu12<0',
        'nvidia-cufft-cu12<0',
        'nvidia-curand-cu12<0',
        'nvidia-cusolver-cu12<0',
        'nvidia-cusparse-cu12<0',
        'nvidia-cusparselt-cu12<0',
        'nvidia-nccl-cu12<0',
        'nvidia-nvjitlink-cu12<0',
        'nvidia-nvtx-cu12<0',
    )
    _CONSTRAINTS_FILE = '/tmp/accelerate_npu_constraints.txt'

    # Cluster-internal nginx PyPI cache + Huawei Cloud ascend dual-source.
    _CLUSTER_INDEX = 'http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple'
    _ASCEND_EXTRA = 'https://repo.huaweicloud.com/ascend/repos/pypi'

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is hard-coded, tied to the GitHub workflow container image.
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    @classmethod
    def prepare_environment(cls) -> None:
        """Install CANN env + CUDA constraints + torch stack + transformers.

        Accelerate itself is intentionally NOT pre-installed: the doc body
        runs ``git clone`` + ``uv pip install -e .`` against the upstream
        release tag injected by the workflow, so the test exercises the exact
        install path users get.
        """
        # 0) CANN env: source set_env.sh and merge the env stream into
        # os.environ. setdefault so workflow-injected envs win.
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

        # 1) CUDA exclusion list + process-level env
        with open(cls._CONSTRAINTS_FILE, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(cls._CUDA_CONSTRAINTS) + '\n')
        os.environ['PIP_CONSTRAINT'] = cls._CONSTRAINTS_FILE
        os.environ['UV_CONSTRAINT'] = cls._CONSTRAINTS_FILE

        # 2) uv: the doc body's install step is ``uv pip install -e .``, whose
        # clean stdout keeps the ``accelerate xxx`` fuzzy match unpolluted.
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

        # 3) torch stack probe: when the image's pre-installed wheels already
        # match the matrix, reuse them to avoid the cluster cache resolving
        # a ``+cpu`` build.
        _PROBE_SCRIPT = (
            'import torch, torch_npu\n'
            "raise SystemExit(0 if "
            "torch.__version__.startswith('2.9.0') "
            "and torch_npu.__version__.startswith('2.9.0') "
            "else 1)"
        )
        probe = subprocess.run(
            ['python', '-c', _PROBE_SCRIPT],
            capture_output=True,
            check=False,  # probe's exit code is the branch signal
        )
        if probe.returncode == 0:
            _VERSIONS_SCRIPT = (
                'import torch, torch_npu; '
                'print(torch.__version__, torch_npu.__version__)'
            )
            versions = subprocess.run(
                ['python', '-c', _VERSIONS_SCRIPT],
                capture_output=True, text=True, check=True,
            )
            print(f'setup: reusing image torch stack ({versions.stdout.strip()})')
        else:
            print('setup: installing torch==2.9.0 torch_npu==2.9.0.post2')
            subprocess.run(
                [
                    'python', '-m', 'pip', 'install',
                    '--index-url', cls._CLUSTER_INDEX,
                    '--extra-index-url', cls._ASCEND_EXTRA,
                    'torch==2.9.0', 'torch_npu==2.9.0.post2',
                ],
                check=True,
            )

        # 4) transformers: the big-model-inference section builds a toy
        # LlamaConfig / LlamaForCausalLM, and `accelerate env` prints
        # transformers info when installed.
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'transformers<5.0'],
            check=True,
        )

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per class. ``@unittest.skipIf`` only skips
        the test *method* — ``setUpClass`` itself always runs, so the
        ``if _e2e_enabled()`` guard keeps heavy setup from firing on
        non-NPU runners.
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
