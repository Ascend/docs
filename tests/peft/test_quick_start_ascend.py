"""Quick-start-Ascend test: doc under test is ``sources/peft/quick_start.md``.

Run: ``python -m unittest tests.peft.test_quick_start_ascend -v 2>&1``

Env (injected by the engine ``quick-start-template.yml``, triggered by
``peft-quick-start.yml``): ``MONITORED_DOC_URL``, ``UPSTREAM_REF``,
``NPU_READY=true`` (otherwise the class is skipped — ``import torch_npu``
hard-fails off the NPU runner).
"""

from __future__ import annotations

import os
import subprocess
import unittest

from doc_test.base import MarkdownDocTestBase
from doc_test.model_cache import (
    diagnose_mount_environment,
    ensure_safetensors,
    purge_huggingface_corrupt,
    report_huggingface_state,
    resolve_huggingface_cache,
)


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

    # 2 h, matching job ``timeout_minutes: 120``. Cold-cache download of
    # Qwen2.5-3B-Instruct takes ~80 min on the NPU runner at ~1.2 MB/s,
    # so this leaves room for the incremental re-download while still
    # capped at the job timeout. Warm caches finish in <5 min.
    DEFAULT_COMMAND_TIMEOUT = 7200
    USER_AGENT = 'cosdt-ci-test/quick-start'  # monitored source lives under this org
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,  # generic [ERROR] + Traceback
        'applicaiton exception',  # typo in CANN's Python driver (sic)
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    # Process-level CUDA exclusion: written to a file and exported as
    # PIP_CONSTRAINT / UV_CONSTRAINT so subprocess installs inherit it.
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
    _CONSTRAINTS_FILE = '/tmp/peft_npu_constraints.txt'

    # Cluster-internal nginx PyPI cache + Huawei Cloud ascend dual-source.
    _CLUSTER_INDEX = 'http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple'
    _ASCEND_EXTRA = 'https://repo.huaweicloud.com/ascend/repos/pypi'

    # Doc step that pulls this via ``snapshot_download``; pre-flight cache
    # log in ``prepare_environment`` is keyed off it.
    _MODEL_ID = 'Qwen/Qwen2.5-3B-Instruct'

    # Hard-coded path inside the CANN container image.
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env + CUDA exclusion + uv + torch probe + safetensors
        + huggingface cache validation. The doc's ``#test`` blocks install
        ``transformers`` / ``huggingface_hub`` / ``peft`` in document order.
        """
        # 0) CANN env: source set_env.sh and merge into os.environ.
        if os.path.isfile(cls._CANN_SET_ENV):
            merged = subprocess.run(
                ['bash', '-c', f'source {cls._CANN_SET_ENV} >/dev/null 2>&1; env'],
                capture_output=True, text=True, check=True,
            )
            for line in merged.stdout.splitlines():
                if '=' not in line:
                    continue
                key, _, value = line.partition('=')
                # Don't overwrite workflow-injected envs; only fill in
                # CANN keys that are missing.
                os.environ.setdefault(key, value)
            print('setup: sourced CANN env from set_env.sh')
        else:
            print(
                f'setup: skipping CANN env source ({cls._CANN_SET_ENV} not present)'
            )

        # 1) CUDA exclusion list + process-level env.
        with open(cls._CONSTRAINTS_FILE, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(cls._CUDA_CONSTRAINTS) + '\n')
        os.environ['PIP_CONSTRAINT'] = cls._CONSTRAINTS_FILE
        os.environ['UV_CONSTRAINT'] = cls._CONSTRAINTS_FILE

        # 2) uv: ``peft-install-source`` runs ``uv pip install -e .`` for
        # PEP 517 builds; pip alone is less reliable there.
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

        # 3) torch stack probe + install: reuse image's pre-installed
        # wheels when versions match; otherwise install via cluster
        # cache + Huawei ascend dual-source to avoid the ``+cpu`` pull.
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

        # 4) safetensors: pulled in transitively by torch on most images;
        # install defensively in case the CANN base ships without it.
        ensure_safetensors()

        # 4b) tqdm: hard dep of transformers/huggingface_hub so
        # ``snapshot_download`` prints an ETA-bearing progress bar during
        # cold-cache downloads. The CANN base image ships a slim Python
        # that may strip it; defensive install keeps the progress bar
        # visible in CI logs.
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'tqdm'],
            check=True,
        )

        # 5) Mount environment diagnostic: probes bind-mount state
        # (findmnt / mountinfo / capabilities / cgroup) before the
        # cache log so the state line has its environment context.
        # ``model_id`` enables the HF probe (cache existence + primary-
        # endpoint throughput to huggingface.co), kept for future
        # ``snapshot_download`` provider swaps.
        diagnose_mount_environment(model_id=cls._MODEL_ID)

        # 6) Cache validation: pre-flight log for our model + full-cache
        # scan + purge of corrupt shards. Doc pulls via
        # ``huggingface_hub.snapshot_download``, so we validate the HF
        # cache layout (models--<org>--<model>/{blobs,snapshots}).
        report_huggingface_state(cls._MODEL_ID)
        purge_huggingface_corrupt(resolve_huggingface_cache())

        # transformers / huggingface_hub / peft are installed by the doc's
        # ``#test`` blocks. accelerate intentionally omitted — quickstart
        # uses explicit ``.to("npu:0")`` only, no ``device_map`` /
        # ``Accelerator`` involvement.

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per class. ``@unittest.skipIf`` only skips
        the test *method* — ``setUpClass`` itself always runs, so the
        ``if _e2e_enabled()`` guard inside ``prepare_environment`` keeps
        heavy setup from firing on non-NPU runners.
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