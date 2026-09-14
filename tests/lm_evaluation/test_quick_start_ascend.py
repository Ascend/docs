"""Quick-start-Ascend documentation test: end-to-end case built on top
of the ``MarkdownDocTestBase`` contract.

Document under test: ``projects/lm-evaluation-harness/docs/Quick-start-Ascend.md``
(follows the ``docs/markdown_doc_test_label.md`` contract: every ``shell``
code block carries one of the ``#test`` / ``#test-setup`` /
``#test-result`` labels plus ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` parameters).

Run: ``python -m unittest tests.test_quick_start_ascend -v 2>&1``

Environment variables (injected by GitHub workflow
``lm-evaluation-harness-quick-start.yml``):
    ``MONITORED_DOC_URL``         Used by the engine's monitor step (ubuntu,
                                  not the NPU runner) for doc hash checking.
                                  The test's ``pre_process`` reads the doc
                                  from the local checkout instead, because
                                  ``raw.githubusercontent.com`` is not
                                  reachable from the NPU runner's cluster.
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
from pathlib import Path

from doc_test.base import MarkdownDocTestBase, TestCommand
from doc_test.model_cache import (
    ensure_safetensors,
    purge_modelscope_corrupt,
    resolve_modelscope_cache,
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
    """``Quick-start-Ascend.md`` end-to-end test: fetch doc -> validate
    contract -> run ``#test-setup`` / ``#test`` in order -> compare against
    ``#test-result``.

    Scope: env check + lm_eval[hf] install + single-card NPU smoke runs of
    the official ``arc_easy`` and ``winogrande`` tasks on
    ``Qwen/Qwen2.5-0.5B-Instruct`` (weights and both datasets downloaded via
    ModelScope on the first run).
    """

    # 60 min per command: long enough for the ~1 GB Qwen2.5-0.5B
    # snapshot_download on the first run plus the NPU eval run, which needs
    # longer than the 20 min default of the smaller projects.
    DEFAULT_COMMAND_TIMEOUT = 3600

    # Monitored source is the cosdt-ci-test/workflows fork (this repo):
    # the doc lives at projects/lm-evaluation-harness/docs/Quick-start-Ascend.md
    # and the engine sets MONITORED_DOC_URL to the raw.githubusercontent.com
    # URL for the same path.
    USER_AGENT = 'cosdt-ci-test/quick-start'

    # Extend the base ERROR_MARKERS with CANN's typo + sentinel so a CANN
    # failure surfaces a full stderr dump (head/tail by default would hide
    # the line that names the failure).
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',  # CANN toolkit emits this typo (sic)
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    # Process-level CUDA exclusion list. Same rationale as the other
    # projects' tests: write to /tmp and export, so subprocesses
    # (subprocess.run inherits parent env by default) see it. lm-eval's
    # own dep tree (torch / transformers / accelerate / ...) shouldn't
    # pull CUDA, but installing ``lm_eval[hf]`` resolves extras that could
    # drag nvidia-* in via transitive deps.
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
    _CONSTRAINTS_FILE = '/tmp/lm_eval_npu_constraints.txt'

    # Cluster-internal nginx PyPI cache + Huawei Cloud ascend dual-source.
    _CLUSTER_INDEX = 'http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple'
    _ASCEND_EXTRA = 'https://repo.huaweicloud.com/ascend/repos/pypi'

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is hard-coded, tied to the container image pinned by the
    # ``image:`` input of ``lm-evaluation-harness-quick-start.yml``.
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    _RESULT_DIR = Path('output/lm_eval_out')
    _TASKS = ('arc_easy', 'winogrande')

    def _verify_eval_results(self) -> None:
        """CI-side guard: the doc only prints the two accuracy values.

        A run that produces no result JSON, drops a task, or reports an
        accuracy outside [0, 1] is the failure mode the doc used to assert
        inline; keeping the check here preserves the coverage without
        exposing assertions to readers of the quick start. Runs after
        ``run-eval``, which now covers download + eval + result reporting
        in one block.
        """
        import glob
        import json

        found: dict = {}
        for path in glob.glob(
            str(self._RESULT_DIR / '**' / '*.json'), recursive=True
        ):
            with open(path, encoding='utf-8') as fh:
                payload = json.load(fh)
            for task, metrics in payload.get('results', {}).items():
                if task not in self._TASKS:
                    continue
                # v0.4.x stores metrics as "acc,none" / "acc_norm,none"
                for key in metrics:
                    if key.startswith('acc') and 'norm' not in key:
                        value = metrics[key]
                        found[task] = (
                            value.get('value', value)
                            if isinstance(value, dict) else value
                        )
                        break

        for task in self._TASKS:
            if task not in found:
                raise AssertionError(
                    f'no accuracy for task {task!r} under {self._RESULT_DIR}'
                )
            acc = float(found[task])
            if not 0.0 <= acc <= 1.0:
                raise AssertionError(
                    f'{task} accuracy out of [0, 1]: {acc}'
                )
        self.log(
            '[Step] verified eval results '
            + ', '.join(f'{t}={float(found[t]):.4f}' for t in self._TASKS)
        )

    def _run_one(self, cmd, results, env, cwd, timeout, idx):
        if (
            isinstance(cmd, TestCommand)
            and getattr(cmd, 'id', None) == 'run-eval'
        ):
            super()._run_one(cmd, results, env, cwd, timeout, idx)
            self._verify_eval_results()
            return
        return super()._run_one(cmd, results, env, cwd, timeout, idx)

    # ----------------------------------------------------------
    # pre_process: read doc from local checkout instead of fetching
    # raw.githubusercontent.com — the NPU runner sits behind a cluster
    # firewall that cannot reach it (known limitation, see
    # quick-start-template.yml). The checkout step always has the
    # latest doc for the branch under test, so there is no stale drift.
    # ----------------------------------------------------------

    def pre_process(self) -> str:
        """Read ``Quick-start-Ascend.md`` from the local checkout.

        Overrides the base ``MarkdownDocTestBase.pre_process`` which
        fetches ``MONITORED_DOC_URL`` via ``urllib`` — that endpoint
        (``raw.githubusercontent.com``) is not reachable from the NPU
        runner's cluster network. The workflow checkout already places
        the doc at
        ``projects/lm-evaluation-harness/docs/Quick-start-Ascend.md``
        relative to this file, so reading it locally is both reliable
        and always in sync with the branch under test.
        """
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

    # ----------------------------------------------------------
    # prepare_environment: CANN env + CUDA constraints + NPU card pin +
    # torch stack probe + safetensors + modelscope cache validation
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env + write CUDA exclusion list + pin NPU card 0 +
        torch stack probe + safetensors + modelscope cache
        validation.

        The doc's ``## 安装 lm-eval`` block is the single source of truth
        for which lm_eval version gets installed; this class only handles
        ``torch`` / ``torch_npu`` here (via the cluster cache + Huawei
        ascend dual-source). ``lm_eval`` installs itself in document order
        via the ``#test`` machinery (``install-lmeval``).

        Class-level setup: run once per test class, triggered by
        ``setUpClass``. Not the same as ``unittest.TestCase.setUp`` —
        that lifecycle hook fires before every test method, which is
        wrong for a one-shot install.
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

        # 2) Pin the visible NPU card to 0; the doc's run-eval block
        #    targets --device npu:0 and the runner may expose several
        #    davinci devices. Same subprocess-inheritance rationale as
        #    the constraints above.
        os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '0')

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

        # 5) safetensors: native loader used by the cache validation
        # step below. Pulled in transitively by torch on most images;
        # install defensively in case the CANN base ships without it.
        ensure_safetensors()

        # 6) Cache validation: the doc downloads Qwen/Qwen2.5-0.5B-Instruct
        # via ModelScope on the first run. A persistent host-side bind mount
        # can hold truncated safetensors from interrupted runs; walk every
        # shard under each model dir and purge it on failure. modelscope
        # will re-download cleanly on next access. Implementation lives in
        # workflows.model_cache; see that module's docstring for the
        # full rationale.
        purge_modelscope_corrupt(resolve_modelscope_cache())

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class: CANN env + CUDA constraints +
        HF endpoint + torch stack + safetensors + modelscope cache
        validation.

        ``@unittest.skipIf`` only skips the test *method* — ``setUpClass``
        itself always runs. The ``if _e2e_enabled()`` body guard below is
        what actually keeps heavy setup from firing when ``NPU_READY`` is
        unset.
        """
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end requires NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        """Template-method entry point. The base class
        ``run_template()`` runs the full ``pre_process`` -> ``parse`` ->
        ``execute`` -> ``post_process`` flow. ``prepare_environment`` is
        triggered by ``setUpClass`` once, not from ``run_template``."""

        self.run_template()


if __name__ == '__main__':
    unittest.main()
