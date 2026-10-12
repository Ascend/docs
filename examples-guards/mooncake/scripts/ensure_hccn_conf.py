#!/usr/bin/env python3
"""Make /etc/hccn.conf a regular file of this container's device NIC IPs.

Ascend Direct reads address_<phyId>=<ip> from that path. The source
workflow bind-mounted the host file. This engine does not pass container
volumes, and the Hong Kong ARC runners ignore Docker --volume, so the
mount is not here. Recover the IPs already visible in the job: flatten a
directory-shaped mount, copy a driver copy, or query hccn_tool. Do not
invent addresses from host NICs.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

HCCN_CONF = '/etc/hccn.conf'
DRIVER_HCCN_CONF = '/usr/local/Ascend/driver/tools/hccn.conf'


def _phy_ids_from_env() -> list[int]:
    for key in ('ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES'):
        ids = [
            int(part)
            for part in os.environ.get(key, '').split(',')
            if part.strip().isdigit()
        ]
        if ids:
            return ids
    return []


def _phy_ids_from_npu_smi() -> list[int]:
    info = subprocess.run(
        ['npu-smi', 'info'],
        capture_output=True,
        text=True,
        check=False,
    )
    found: list[int] = []
    for line in info.stdout.splitlines():
        match = re.search(r'\|\s+(\d+)\s+910', line)
        if match is None:
            continue
        phy = int(match.group(1))
        if phy not in found:
            found.append(phy)
    return found


def _parse_hccn_ip(text: str) -> str | None:
    """Parse `hccn_tool -i N -ip -g` output (`ipaddr:1.2.3.4`)."""
    for line in text.splitlines():
        if 'ipaddr' not in line.lower():
            continue
        _, _, value = line.partition(':')
        ip = value.strip()
        if ip:
            return ip
    return None


def _flatten_dir_shaped_hccn() -> bool:
    """A file mount that ARC created with mkdir -p lands as a directory."""
    nested = os.path.join(HCCN_CONF, 'hccn.conf')
    if not (os.path.isdir(HCCN_CONF) and os.path.isfile(nested)):
        return False
    with open(nested, 'rb') as handle:
        data = handle.read()
    shutil.rmtree(HCCN_CONF)
    with open(HCCN_CONF, 'wb') as handle:
        handle.write(data)
    print(f'setup: flattened directory-shaped {HCCN_CONF}', flush=True)
    return True


def _copy_driver_conf() -> bool:
    if not os.path.isfile(DRIVER_HCCN_CONF):
        return False
    shutil.copyfile(DRIVER_HCCN_CONF, HCCN_CONF)
    print(f'setup: copied {DRIVER_HCCN_CONF} -> {HCCN_CONF}', flush=True)
    return True


def _write_from_tool(tool: str, phy_ids: list[int]) -> bool:
    if not phy_ids:
        print('setup: no NPU phy ids; cannot query hccn_tool', flush=True)
        return False
    lines: list[str] = []
    for phy in phy_ids:
        query = subprocess.run(
            [tool, '-i', str(phy), '-ip', '-g'],
            capture_output=True,
            text=True,
            check=False,
        )
        ip = _parse_hccn_ip(query.stdout) or _parse_hccn_ip(query.stderr)
        if not ip:
            print(
                f'setup: hccn_tool -i {phy} -ip -g failed '
                f'rc={query.returncode} stdout={query.stdout!r} '
                f'stderr={query.stderr!r}',
                flush=True,
            )
            continue
        lines.append(f'address_{phy}={ip}\n')
    if not lines:
        return False
    with open(HCCN_CONF, 'w', encoding='utf-8') as handle:
        handle.writelines(lines)
    print(
        f'setup: wrote {HCCN_CONF} from hccn_tool ({len(lines)} address lines)',
        flush=True,
    )
    return True


def ensure() -> None:
    if os.path.isfile(HCCN_CONF):
        print(f'setup: {HCCN_CONF} already present', flush=True)
        return
    if _flatten_dir_shaped_hccn():
        return
    if _copy_driver_conf():
        return
    # setup_example.sh puts driver/tools on PATH before calling this.
    tool = shutil.which('hccn_tool')
    if tool is None:
        raise SystemExit('setup: no hccn_tool; cannot materialize /etc/hccn.conf')
    phy_ids = _phy_ids_from_env()
    if _write_from_tool(tool, phy_ids):
        return
    smi_ids = _phy_ids_from_npu_smi()
    if smi_ids != phy_ids and _write_from_tool(tool, smi_ids):
        return
    raise SystemExit('setup: hccn_tool produced no device IPs')


if __name__ == '__main__':
    ensure()
