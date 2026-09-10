"""Quick-start test: doc under test is ``sources/modelscope/quick_start.md``.

End-to-end case built on top of the ``MarkdownDocTestBase`` contract:
env check + modelscope source install at the latest release tag +
Qwen2.5-0.5B-Instruct download via the modelscope CLI + a single-card
text-generation smoke on NPU. The inference block goes through
``AutoModelForCausalLM`` + ``.to('npu:0')`` (with a transformers
fallback) because modelscope's own ``verify_device`` whitelist rejects
``device='npu'`` — that limitation is documented in the doc itself.

Run: ``python -m unittest tests.modelscope.test_quick_start_ascend -v 2>&1``

Environment variables (injected by GitHub workflow
``modelscope-quick-start.yml``):
    ``MONITORED_DOC_URL``         doc to fetch and execute
    ``UPSTREAM_REF``              release tag substituted into the doc's <ref>
    ``NPU_READY=true``            Required, otherwise the class is skipped.
                                  End-to-end tests only run on the NPU runner:
                                  local dev machines / normal ubuntu runners
                                  have no ``/dev/davinci*`` device, and the
                                  hard run would fail on ``import torch_npu``.
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
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    # 60 min per command: long enough for the modelscope source install
    # (`git clone` + `uv pip install -e '.[framework]'`, whose aarch64
    # resolver is steered off CUDA wheels by the CUDA exclusion list)
    # plus the first-run Qwen2.5-0.5B-Instruct (~1 GB) download, which
    # needs longer than the 30 min default of the smaller projects.
    DEFAULT_COMMAND_TIMEOUT = 3600

    USER_AGENT = 'cosdt-ci-test/quick-start'  # monitored source lives under this org

    # Extend the base ERROR_MARKERS with CANN's typo + sentinel so a CANN
    # failure surfaces a full stderr dump (head/tail by default would hide
    # the line that names the failure).
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,  # generic [ERROR] + Traceback
        'applicaiton exception',  # typo in CANN's Python driver (sic)
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    # Process-level CUDA exclusion list. Same rationale as the other
    # projects' tests: write to /tmp and export, so subprocesses
    # (subprocess.run inherits parent env by default) see it. modelscope's
    # dep tree itself doesn't pull CUDA, but bare ``torch`` /
    # ``torchvision`` wheels from the default PyPI index depend on
    # nvidia-* packages; the constraints keep any accidental re-resolution
    # of the torch stack from dragging them in and breaking the
    # torch_npu pairing.
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
    _CONSTRAINTS_FILE = '/tmp/modelscope_npu_constraints.txt'

    # Cluster-internal nginx PyPI cache + Huawei Cloud ascend dual-source.
    _CLUSTER_INDEX = 'http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple'
    _ASCEND_EXTRA = 'https://repo.huaweicloud.com/ascend/repos/pypi'

    # ModelScope hub repo carrying Qwen2.5-0.5B-Instruct; downloads go
    # through modelscope's own CLI / from_pretrained patch (the project
    # under test IS the ModelScope hub client, so no HF fallback here).
    _MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is hard-coded, tied to the container image pinned by the
    # ``image:`` input of ``modelscope-quick-start.yml``.
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env + write CUDA exclusion list + install uv +
        torch stack probe + safetensors + modelscope cache validation.

        The doc's ``## 安装 modelscope`` block is the single source of
        truth for which modelscope deps get installed (source install);
        this class only handles ``torch`` / ``torch_npu`` here (via the
        cluster cache + Huawei ascend dual-source). Everything else
        installs in document order via the ``#test`` machinery.
        """
        # 0) CANN env: source set_env.sh and merge the env stream into
        # os.environ
        if os.path.isfile(cls._CANN_SET_ENV):
            merged = subprocess.run(
                ['bash', '-c', f'source {cls._CANN_SET_ENV} >/dev/null 2>&1; env'],
                capture_output=True, text=True, check=True,
            )
            for line in merged.stdout.splitlines():
                if '=' not in line:
                    continue
                key, _, value = line.partition('=')
                # Don't overwrite envs explicitly injected by the
                # workflow (jobs.env / steps.env); only fill in CANN
                # keys that are missing, to avoid conflicts.
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

        # 2) uv: the doc's install blocks call ``uv pip`` (source install
        # of modelscope + transformers pin), so uv must exist before the
        # doc executes. Inherit ``PIP_INDEX_URL`` + ``PIP_TRUSTED_HOST``
        # from the yml job-level env (cluster cache path + trusted-host).
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

        # 3) torch stack probe: when version matches the image's
        # pre-installed wheels, reuse them to avoid the cluster cache
        # triggering ``+cpu`` resolution.
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
            check=False,  # probe's success/failure is the branch signal — don't raise
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

        # 4) Single-card determinism: the doc's check-torch block asserts
        # ``count: 1``; pinning the visible device keeps that assertion
        # true even if the cluster assigns a multi-card container.
        os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '0')

        # 5) safetensors: native loader used by the cache validation
        # step below. Pulled in transitively by torch on most images;
        # install defensively in case the CANN base ships without it.
        ensure_safetensors()

        # 6) ModelScope cache validation: the doc downloads
        # Qwen/Qwen2.5-0.5B-Instruct (~1 GB) via the modelscope CLI on
        # the first run. A persistent host-side cache can hold truncated
        # safetensors from interrupted runs; report the current state,
        # then walk every shard under each model dir and purge it on
        # failure so the next access re-downloads cleanly. The
        # throughput / mount probes inside diagnose_mount_environment
        # target the modelscope cache root by default.
        diagnose_mount_environment()

        report_modelscope_state(cls._MODEL_ID)
        purge_modelscope_corrupt(resolve_modelscope_cache())

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
        """Template-method entry point. The base class ``run_template()``
        runs the full ``pre_process`` -> ``parse`` -> ``execute`` ->
        ``post_process`` flow."""

        self.run_template()


if __name__ == '__main__':
    unittest.main()
