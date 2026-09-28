"""Quick-start-Ascend documentation test: end-to-end case built on top
of the MarkdownDocTestBase contract.

Document under test: projects/timm/docs/Quick-start-Ascend.md
(follows the docs/markdown_doc_test_label.md contract: every shell
code block carries one of the #test / #test-setup / #test-result labels
plus id= / store= / load='x>>y' / fuzzy='xxx' parameters).

Run: python -m unittest tests.test_quick_start_ascend -v 2>&1

Environment variables (injected by GitHub workflow timm-quick-start.yml):
    MONITORED_DOC_URL   Used by the engine's monitor step (ubuntu, not the
                        NPU runner) for doc hash checking. The test's
                        pre_process reads the doc from the local checkout
                        instead, because raw.githubusercontent.com is not
                        reachable from the NPU runner's cluster.
    NPU_READY=true      Required, otherwise the class is skipped.
                        End-to-end tests only run on the NPU runner: local
                        dev machines / normal ubuntu runners have no
                        /dev/davinci* device, and the hard run would fail
                        on import torch_npu.
"""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

from doc_test.base import MarkdownDocTestBase
from doc_test.model_cache import (
    ensure_safetensors,
    purge_modelscope_corrupt,
    resolve_modelscope_cache,
)


def _is_truthy(value: str | None) -> bool:
    """'true' -> True (case-insensitive); anything else (including unset) -> False."""
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    """Return True when NPU_READY=true is set."""
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """Quick-start-Ascend.md end-to-end test: fetch doc -> validate
    contract -> run #test-setup / #test in order -> compare against
    #test-result.

    Scope: install + the quick-start flow's #test smoke commands,
    constructing the model on device='npu:0' with pretrained weights
    auto-downloaded to the default modelscope cache (by the doc's own
    snapshot_download inside the example). Covers the upstream quickstart
    in order: image classification inference / multi-scale feature
    extraction.
    """

    # pip install timm + the doc's two quick-start #test smoke
    # commands (inference / features) + modelscope weight download.
    # The stack is small (torch / torchvision / pyyaml / huggingface_hub /
    # safetensors / modelscope / timm), so 30 min covers cold cache +
    # first-time wheel pulls + the 45 MB model download + the ~1s smoke
    # commands (incl. a single resnet18 fwd pass on npu:0) comfortably.
    DEFAULT_COMMAND_TIMEOUT = 1800

    USER_AGENT = 'cosdt-ci-test/quick-start'

    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',
        'ERR99999',
    )

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
    _CONSTRAINTS_FILE = '/tmp/timm_npu_constraints.txt'

    _CLUSTER_INDEX = 'http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple'
    _ASCEND_EXTRA = 'https://repo.huaweicloud.com/ascend/repos/pypi'
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    def pre_process(self) -> str:
        """Read Quick-start-Ascend.md from the local checkout."""
        doc_path = (
            Path(__file__).resolve().parent.parent
            / 'docs'
            / 'Quick-start-Ascend.md'
        )
        if not doc_path.is_file():
            raise RuntimeError(
                f'doc not found in local checkout: {doc_path}'
            )
        return doc_path.read_text(encoding='utf-8')

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env + write CUDA exclusion list +
        torch stack + torchvision + modelscope + model cache sanity.

        The doc's install-timm section is the single source of truth for
        which timm version gets installed; this class only handles torch /
        torch_npu / torchvision here (via the cluster cache + Huawei ascend
        dual-source). timm itself installs itself in document order via the
        #test machinery.

        modelscope is installed here because the doc's examples call
        snapshot_download; the modelscope cache is purged of corrupt shards
        before the doc's download runs.
        """
        # 0) CANN env
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

        # 1) CUDA exclusion list
        with open(cls._CONSTRAINTS_FILE, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(cls._CUDA_CONSTRAINTS) + '\n')
        os.environ['PIP_CONSTRAINT'] = cls._CONSTRAINTS_FILE

        # 2) torch stack probe + install
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
            check=False,
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

        # 3) torchvision (pinned to match torch 2.9.0)
        subprocess.run(
            ['python', '-m', 'pip', 'install', '--no-deps', 'torchvision==0.24.0'],
            check=True,
        )

        # 4) modelscope
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'modelscope'],
            check=True,
        )

        # 5) model cache sanity
        ensure_safetensors()
        purge_modelscope_corrupt(resolve_modelscope_cache())

        # 6) timm itself is NOT installed here - the doc's
        # install-timm block installs timm via pip.

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class."""
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end requires NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        """Template-method entry point."""
        self.run_template()


if __name__ == '__main__':
    unittest.main()
