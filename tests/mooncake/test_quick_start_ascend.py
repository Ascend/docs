"""Quick-start test: doc under test is ``sources/mooncake/quick_start.md``.

Run: ``python -m unittest tests.mooncake.test_quick_start_ascend -v 2>&1``

Env (injected by the engine ``quick-start-template.yml``, triggered by
``mooncake-quick-start.yml``): ``MONITORED_DOC_URL``, ``UPSTREAM_REF``,
``NPU_READY=true`` (otherwise the class is skipped).
"""

from __future__ import annotations

import os
import re
import shutil
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
    _HCCN_CONF = '/etc/hccn.conf'
    _HCCN_TOOL_CANDIDATES = (
        '/usr/local/Ascend/driver/tools/hccn_tool',
        '/usr/local/sbin/hccn_tool',
        '/usr/local/bin/hccn_tool',
    )
    _HCCN_CONF_CANDIDATES = (
        '/usr/local/Ascend/driver/tools/hccn.conf',
    )
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
    def _npu_phy_ids(cls) -> list[int]:
        """Physical NPU ids visible in this container.

        Prefer ``ASCEND_VISIBLE_DEVICES`` / ``ASCEND_RT_VISIBLE_DEVICES``.
        Fall back to ``npu-smi info`` rows that name a 910-series chip
        (PR run 34184104089 showed phy 0 and 2, not remapped 0/1).
        """
        for key in ('ASCEND_VISIBLE_DEVICES', 'ASCEND_RT_VISIBLE_DEVICES'):
            raw = os.environ.get(key, '')
            ids = [int(part) for part in raw.split(',') if part.strip().isdigit()]
            if ids:
                return ids
        info = subprocess.run(
            ['npu-smi', 'info'],
            capture_output=True,
            text=True,
            check=False,
        )
        found: list[int] = []
        for line in info.stdout.splitlines():
            match = re.search(r'\|\s+(\d+)\s+910', line)
            if match:
                phy = int(match.group(1))
                if phy not in found:
                    found.append(phy)
        return found

    @classmethod
    def _resolve_hccn_tool(cls) -> str | None:
        found = shutil.which('hccn_tool')
        if found:
            return found
        for path in cls._HCCN_TOOL_CANDIDATES:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        return None

    @classmethod
    def _parse_hccn_ip(cls, text: str) -> str | None:
        """Parse ``hccn_tool -i N -ip -g`` stdout (``ipaddr:1.2.3.4``)."""
        for line in text.splitlines():
            if 'ipaddr' not in line.lower():
                continue
            _, _, value = line.partition(':')
            ip = value.strip()
            if ip:
                return ip
        return None

    @classmethod
    def _log_hccn_probe(cls) -> None:
        """Dump what the job container actually has for ADXL device IPs.

        Hong Kong A2 uses ARC k8s hooks. ``container.options --volume``
        is ignored there (run 34184104089 still had no ``/etc/hccn.conf``).
        """
        subprocess.run(
            [
                'bash',
                '-c',
                r'''
set +e
echo "setup: hccn probe begin"
echo "setup: ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES-}"
echo "setup: ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES-}"
ls -la /etc/hccn.conf /etc/ascend_install.info 2>&1
command -v hccn_tool
ls -la /usr/local/Ascend/driver/tools/hccn_tool 2>&1
ls /usr/local/Ascend/driver/tools 2>&1 | head
find /usr/local/Ascend /etc -name hccn.conf -o -name hccn_tool 2>/dev/null
echo "setup: hccn probe end"
''',
            ],
            check=False,
        )

    @classmethod
    def _flatten_dir_shaped_hccn(cls) -> bool:
        """ARC k8s ``container.volumes`` does ``mkdir -p`` on the target.

        A file mount then lands as ``/etc/hccn.conf/hccn.conf``.
        """
        dest = cls._HCCN_CONF
        nested = os.path.join(dest, 'hccn.conf')
        if not (os.path.isdir(dest) and os.path.isfile(nested)):
            return False
        with open(nested, 'rb') as handle:
            data = handle.read()
        shutil.rmtree(dest)
        with open(dest, 'wb') as handle:
            handle.write(data)
        print(f'setup: flattened directory-shaped {dest}')
        return True

    @classmethod
    def _write_hccn_from_tool(cls, tool: str) -> bool:
        phy_ids = cls._npu_phy_ids()
        if not phy_ids:
            print('setup: no NPU phy ids; cannot query hccn_tool')
            return False
        lines: list[str] = []
        for phy in phy_ids:
            query = subprocess.run(
                [tool, '-i', str(phy), '-ip', '-g'],
                capture_output=True,
                text=True,
                check=False,
            )
            ip = cls._parse_hccn_ip(query.stdout) or cls._parse_hccn_ip(query.stderr)
            if not ip:
                print(
                    f'setup: hccn_tool -i {phy} -ip -g failed '
                    f'rc={query.returncode} stdout={query.stdout!r} '
                    f'stderr={query.stderr!r}'
                )
                continue
            lines.append(f'address_{phy}={ip}\n')
        if not lines:
            return False
        try:
            with open(cls._HCCN_CONF, 'w', encoding='utf-8') as handle:
                handle.writelines(lines)
        except OSError as exc:
            print(f'setup: cannot write {cls._HCCN_CONF}: {exc}')
            return False
        print(
            f'setup: wrote {cls._HCCN_CONF} from hccn_tool '
            f'({len(lines)} address lines)'
        )
        return True

    @classmethod
    def _ensure_hccn_conf(cls) -> None:
        """Make ``/etc/hccn.conf`` exist so the doc ``#test id="hccn"`` and ADXL agree.

        Users mount the host file. This runner cannot: ARC k8s drops
        Docker ``--volume``. Recover from a copy already in the job
        container, or from ``hccn_tool`` on the same devices. Do not
        invent IPs from host NICs.
        """
        cls._log_hccn_probe()
        if os.path.isfile(cls._HCCN_CONF):
            print(f'setup: {cls._HCCN_CONF} already present')
            return
        if cls._flatten_dir_shaped_hccn():
            return
        for src in cls._HCCN_CONF_CANDIDATES:
            if os.path.isfile(src):
                try:
                    shutil.copyfile(src, cls._HCCN_CONF)
                except OSError as exc:
                    print(f'setup: cannot copy {src} -> {cls._HCCN_CONF}: {exc}')
                    return
                print(f'setup: copied {src} -> {cls._HCCN_CONF}')
                return
        tool = cls._resolve_hccn_tool()
        if tool is None:
            print('setup: no hccn_tool; cannot materialize /etc/hccn.conf')
            return
        if not cls._write_hccn_from_tool(tool):
            print('setup: hccn_tool produced no device IPs')

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

        path_dirs = '/usr/local/sbin:/usr/local/bin:/usr/local/Ascend/driver/tools'
        current_path = os.environ.get('PATH', '')
        if path_dirs not in current_path:
            os.environ['PATH'] = f'{path_dirs}:{current_path}'

        cls._ensure_hccn_conf()

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
