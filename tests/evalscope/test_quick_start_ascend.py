"""Quick-start test: doc under test is ``sources/evalscope/quick_start.md``.

Run: ``python -m unittest tests.evalscope.test_quick_start_ascend -v 2>&1``

Env (injected by the engine ``quick-start-template.yml``, triggered by
``evalscope-quick-start.yml``): ``MONITORED_DOC_URL``, ``UPSTREAM_REF``,
``NPU_READY=true`` (otherwise the class is skipped).
"""

from __future__ import annotations

import os
import subprocess
import unittest

from doc_test.base import MarkdownDocTestBase
from doc_test.model_cache import (
    diagnose_mount_environment,
    ensure_safetensors,
    purge_modelscope_corrupt,
    report_modelscope_state,
    resolve_modelscope_cache,
)


def _is_truthy(value: str | None) -> bool:
    """``'true'`` -> True (case-insensitive); anything else (unset) -> False."""
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    """Return True when ``NPU_READY=true`` is set, releasing the skip."""
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """``sources/evalscope/quick_start.md`` end-to-end test: fetch doc ->
    validate contract -> run ``#test-setup`` / ``#test`` in order ->
    compare against ``#test-result``.

    Scope: env check + evalscope install + single-card NPU smoke runs of
    the official ``gsm8k`` and ``arc`` (ARC-Easy / ARC-Challenge) datasets
    on Qwen/Qwen2.5-0.5B-Instruct (weights and both datasets downloaded via
    ModelScope on the first run).
    """

    # 60 min per command: long enough for the ~1 GB Qwen2.5-0.5B
    # snapshot_download on the first run plus the NPU eval run.
    DEFAULT_COMMAND_TIMEOUT = 3600

    USER_AGENT = 'cosdt-ci-test/quick-start'

    # Extend the base ERROR_MARKERS with CANN's typo + sentinel so a CANN
    # failure surfaces a full stderr dump.
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,  # generic [ERROR] + Traceback
        'applicaiton exception',  # typo in CANN's Python driver (sic)
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    # Process-level CUDA exclusion list. evalscope's dep tree
    # (torch / transformers / accelerate / ...) shouldn't pull CUDA, but
    # installing ``evalscope`` resolves extras that could drag nvidia-* in
    # via transitive deps.
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
    _CONSTRAINTS_FILE = '/tmp/evalscope_npu_constraints.txt'

    # Cluster-internal nginx PyPI cache + Huawei Cloud ascend dual-source.
    _CLUSTER_INDEX = 'http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple'
    _ASCEND_EXTRA = 'https://repo.huaweicloud.com/ascend/repos/pypi'

    _MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is tied to the container image pinned by the ``image:`` input of
    # ``evalscope-quick-start.yml``.
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env + write CUDA exclusion list + pin NPU card 0 +
        install torch stack + safetensors + modelscope cache validation.

        The doc's ``## 3. 安装 EvalScope`` block is the single source of
        truth for which evalscope version gets installed; this hook only
        handles ``torch`` / ``torch_npu`` (via the cluster cache + Huawei
        Cloud ascend dual-source). ``evalscope`` installs itself in document
        order through the ``#test-setup`` / ``#test`` machinery.
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
                # Don't overwrite envs explicitly injected by the workflow;
                # only fill in CANN keys that are missing.
                os.environ.setdefault(key, value)
            print('setup: sourced CANN env from set_env.sh')
        else:
            print(
                f'setup: skipping CANN env source ({cls._CANN_SET_ENV} not present)'
            )

        # CUDA exclusion list + process-level env (subprocess.run inherits it)
        with open(cls._CONSTRAINTS_FILE, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(cls._CUDA_CONSTRAINTS) + '\n')
        os.environ['PIP_CONSTRAINT'] = cls._CONSTRAINTS_FILE
        os.environ['UV_CONSTRAINT'] = cls._CONSTRAINTS_FILE

        # Pin the visible NPU card to 0; the runner may expose several
        # davinci devices.
        os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '0')

        # torch stack probe: reuse the image's pre-installed wheels when the
        # version matches, to avoid a mismatched resolution from the cache.
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

        # safetensors: native loader used by the cache validation below.
        ensure_safetensors()

        # Cache validation: the doc downloads Qwen/Qwen2.5-0.5B-Instruct via
        # ModelScope on the first run. A persistent host-side cache can hold
        # truncated safetensors from interrupted runs; walk every shard under
        # each model dir and purge it on failure, so modelscope re-downloads
        # cleanly on next access.
        diagnose_mount_environment(model_id=cls._MODEL_ID)

        report_modelscope_state(cls._MODEL_ID)
        purge_modelscope_corrupt(resolve_modelscope_cache())

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per class. ``@unittest.skipIf`` only skips the
        test *method* — ``setUpClass`` itself always runs, so the
        ``if _e2e_enabled()`` guard inside keeps heavy setup from firing on
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
