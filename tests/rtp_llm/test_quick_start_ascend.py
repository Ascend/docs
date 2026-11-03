"""Quick-start test: doc under test is ``sources/rtp_llm/quick_start.md``.

End-to-end case built on top of the ``MarkdownDocTestBase`` contract:
Python 3.10 venv at ``/opt/conda310`` -> clone the upstream
``feature/ascend`` branch -> adapt the aarch64 build inputs (upstream
pins x86_64 torches / x86 lock direct links / x86 ``http_archive``
entries / a bare ``torch_npu`` in the wheel metadata) -> install the
runtime dependency set -> ``bazelisk build //rtp_llm:rtp_llm_whl`` ->
install the wheel -> load the C++ engine ops -> download Qwen3-0.6B ->
start the HTTP server and issue one generation request.

The Ascend adaptation is not on ``main``: it landed via upstream PR #1481
(merged 2026-10-27) into ``alibaba/rtp-llm:feature/ascend``, and the
follow-up Ascend PRs target that branch too. The trigger pins
``fixed_ref: feature/ascend``, whose tip SHA is the monitor's change key;
the doc clones the branch by name, so the guard and a reader run the
same command.

Run: ``python -m unittest tests.rtp_llm.test_quick_start_ascend -v 2>&1``

Environment variables (injected by the engine
``quick-start-template.yml``, triggered by ``rtp-llm-quick-start.yml``):
    ``MONITORED_DOC_URL``         doc to fetch and execute
    ``NPU_READY=true``            Required, otherwise the class is skipped.
                                  End-to-end tests only run on the NPU
                                  runner: local dev machines / normal ubuntu
                                  runners have no ``/dev/davinci*`` device.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import unittest

from doc_test.base import MarkdownDocTestBase


def _is_truthy(value: str | None) -> bool:
    """``'true'`` -> True (case-insensitive); anything else (unset) -> False."""
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    """Return True when ``NPU_READY=true`` is set, releasing the skip."""
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """``sources/rtp_llm/quick_start.md`` end-to-end test: fetch doc ->
    validate contract -> run ``#test-setup`` / ``#test`` in order ->
    compare against ``#test-result``.

    Working directory is pinned to ``/root/rtp-llm-test``: the doc is
    cwd-relative (clones to ``./rtp-llm``, model lands in
    ``/root/rtp-llm-test/models``), so the clone tree must not collide
    with the ``workflows`` checkout the engine runs the test from.
    """

    # The bazel build is the dominant cost (external repo fetch + ~12k
    # actions). 4h per command leaves headroom for a cold build; the
    # trigger's timeout_minutes (240) matches this ceiling.
    DEFAULT_COMMAND_TIMEOUT = 14400

    USER_AGENT = 'ascend-docs/quick-start'

    # Extend the base ERROR_MARKERS with CANN's typo + sentinel so a
    # CANN runtime failure surfaces a full stderr dump instead of a
    # head/tail window that hides the failing line.
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',  # CANN toolkit emits this typo (sic)
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    _CANN_SET_ENV_CANDIDATES = (
        '/usr/local/Ascend/ascend-toolkit/set_env.sh',
        '/usr/local/Ascend/cann/set_env.sh',
    )

    # Upstream hardcodes this interpreter path in deps/pip.bzl and in the
    # bazel `ascend` config, and every lock entry is a cp310 wheel, so the
    # doc's `python` must resolve here. The doc creates the venv itself.
    _PYTHON_PREFIX = '/opt/conda310'

    # Build tools the bazel build needs; the engine only bootstraps
    # curl / jq / git.
    _BUILD_TOOLS: dict[str, str] = {
        'gcc': 'gcc',
        'g++': 'g++',
        'make': 'make',
        'git': 'git',
        'curl': 'curl',
    }

    _WORK_DIR = '/root/rtp-llm-test'

    # Ubuntu's ca-certificates-java regenerates this JVM truststore from
    # /etc/ssl/certs on every CA update, i.e. it trusts exactly what curl
    # trusts. Bazel's downloader runs inside a JVM and only reads the JDK's
    # own cacerts, so on a network whose TLS is intercepted by a proxy the
    # very first external archive fetch dies with `PKIX path building
    # failed` while curl on the same host succeeds.
    _JVM_TRUSTSTORE = '/etc/ssl/certs/java/cacerts'

    # ----------------------------------------------------------
    # prepare_environment: CANN env + build tools + JVM truststore + cwd
    # (the Python 3.10 venv at _PYTHON_PREFIX is created by the doc itself)
    # ----------------------------------------------------------

    @classmethod
    def _ensure_jvm_truststore(cls) -> None:
        """Point the JVM at a truststore derived from the system CA bundle.

        ``bazelisk build`` fetches ~40 external archives with Bazel's Java
        downloader. That downloader validates TLS against the JDK's bundled
        cacerts only, so on a runner whose egress goes through a
        TLS-intercepting proxy every fetch fails with ``PKIX path building
        failed: unable to find valid certification path`` — while ``curl``
        in the same container downloads fine, because curl reads the system
        bundle that has the proxy's root CA.

        ``ca-certificates-java`` generates ``/etc/ssl/certs/java/cacerts``
        from that same system bundle, and ``JAVA_TOOL_OPTIONS`` makes every
        JVM Bazel spawns pick it up (the build block runs unchanged).
        """
        truststore = cls._JVM_TRUSTSTORE
        if not os.path.isfile(truststore):
            print(f'setup: {truststore} missing, installing ca-certificates-java')
            subprocess.run(['apt-get', 'update', '-qq'], check=False)
            subprocess.run(
                ['apt-get', 'install', '-y', '-qq', 'ca-certificates-java'],
                check=False,
            )
        if not os.path.isfile(truststore):
            print(
                f'setup: {truststore} still missing - if the build dies with '
                'PKIX path building failed, the JVM cannot see the proxy CA'
            )
            probe = subprocess.run(
                [
                    'bash', '-c',
                    'echo | openssl s_client -connect github.com:443 '
                    '-servername github.com 2>/dev/null '
                    "| grep -m3 -E 'Verify return code|^ *i:|^ *s:'",
                ],
                capture_output=True, text=True, check=False,
            )
            for line in probe.stdout.splitlines():
                print(f'setup:   tls probe: {line.strip()}')
            return
        opts = (
            f'-Djavax.net.ssl.trustStore={truststore} '
            '-Djavax.net.ssl.trustStorePassword=changeit'
        )
        existing = os.environ.get('JAVA_TOOL_OPTIONS', '').strip()
        os.environ['JAVA_TOOL_OPTIONS'] = f'{existing} {opts}'.strip()
        print(f'setup: JAVA_TOOL_OPTIONS={os.environ["JAVA_TOOL_OPTIONS"]}')

    @classmethod
    def prepare_environment(cls) -> None:
        """CANN env + build tools + JVM truststore + working directory.

        Everything business-facing is in the document: installing bazelisk
        and the Python 3.10 venv, the aarch64 build-input adaptation, the
        dependency install, ``bazelisk build``, the wheel install, the
        model download and the server are the steps a reader follows.
        """
        # 0) PATH: /opt/conda310/bin first so the doc's bare `python`
        # resolves to 3.10; /usr/local/sbin for npu-smi.
        current_path = os.environ.get('PATH', '')
        prefix_dirs = f'{cls._PYTHON_PREFIX}/bin:/usr/local/sbin:/usr/local/bin'
        if not current_path.startswith(prefix_dirs):
            os.environ['PATH'] = f'{prefix_dirs}:{current_path}'

        # 1) CANN env: source set_env.sh and merge the env stream into
        # os.environ so every doc block's subprocess inherits
        # LD_LIBRARY_PATH / ASCEND_* without re-sourcing.
        sourced = False
        for set_env in cls._CANN_SET_ENV_CANDIDATES:
            if os.path.isfile(set_env):
                merged = subprocess.run(
                    ['bash', '-c', f'source {set_env} >/dev/null 2>&1; env'],
                    capture_output=True, text=True, check=True,
                )
                for line in merged.stdout.splitlines():
                    if '=' not in line:
                        continue
                    key, _, value = line.partition('=')
                    # Don't overwrite what the workflow injected.
                    os.environ.setdefault(key, value)
                print(f'setup: sourced CANN env from {set_env}')
                sourced = True
                break
        if not sourced:
            print(
                'setup: no CANN set_env.sh found (expected '
                f'{cls._CANN_SET_ENV_CANDIDATES}) - build will fail'
            )

        # 2) Build tools.
        missing_pkgs = [
            pkg for tool, pkg in cls._BUILD_TOOLS.items()
            if shutil.which(tool) is None
        ]
        if missing_pkgs:
            print(f'setup: installing build tools: {missing_pkgs}')
            subprocess.run(['apt-get', 'update', '-qq'], check=True)
            subprocess.run(
                ['apt-get', 'install', '-y', '-qq', *missing_pkgs],
                check=True,
            )

        # 3) JVM truststore for Bazel's Java downloader.
        cls._ensure_jvm_truststore()

        # 4) Working directory: pinned before run_template() so the doc's
        # cwd-relative clone/build never lands in the guard checkout.
        os.makedirs(cls._WORK_DIR, exist_ok=True)
        os.chdir(cls._WORK_DIR)
        print(f'setup: working dir {cls._WORK_DIR}')

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class.

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
