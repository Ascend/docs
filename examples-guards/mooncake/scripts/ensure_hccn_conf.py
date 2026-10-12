#!/usr/bin/env python3
"""Make /etc/hccn.conf a regular file of this container's device NIC IPs.

Ascend Direct reads address_<phyId>=<ip> from that path. The source
workflow bind-mounted the host file. This engine does not pass container
volumes, and the Hong Kong ARC runners ignore Docker --volume, so the
mount is not here. Recover the IPs already visible in the job: flatten a
directory-shaped mount, copy a driver copy, or query hccn_tool. Do not
invent addresses from host NICs.

The engine then sets ASCEND_RT_VISIBLE_DEVICES from the manifest, which
is 0,1. That list is not the physical id on every 2-card pod. Run
38249171972 queried -i 0, got "invalid device id", wrote the one id that
answered, and the initiator aborted because ADXL asked for phy id 3.
Probe with that mask unset, keep every IP the tool returns, and point
the run step at two physical ids that actually resolved.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

HCCN_CONF = '/etc/hccn.conf'
DRIVER_HCCN_CONF = '/usr/local/Ascend/driver/tools/hccn.conf'
CARDS_NEEDED = 2
PROBE_LIMIT = 16


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


def _probe_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop('ASCEND_RT_VISIBLE_DEVICES', None)
    env.pop('ASCEND_VISIBLE_DEVICES', None)
    return env


def _query_ip(tool: str, phy: int, env: dict[str, str]) -> str | None:
    query = subprocess.run(
        [tool, '-i', str(phy), '-ip', '-g'],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    return _parse_hccn_ip(query.stdout) or _parse_hccn_ip(query.stderr)


def _choose_visible(resolved: dict[int, str], smi_ids: list[int]) -> list[int]:
    """Two physical ids for logic id 0 and 1. Prefer the manifest pair."""
    env_ids = [phy for phy in _phy_ids_from_env() if phy in resolved]
    if len(env_ids) >= CARDS_NEEDED:
        return env_ids[:CARDS_NEEDED]
    smi_hit = [phy for phy in smi_ids if phy in resolved]
    if len(smi_hit) >= CARDS_NEEDED:
        return smi_hit[:CARDS_NEEDED]
    return sorted(resolved)[:CARDS_NEEDED]


def _publish_visible_devices(phy_ids: list[int]) -> None:
    path = os.environ.get('GITHUB_ENV')
    if not path:
        return
    value = ','.join(str(phy) for phy in phy_ids)
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(f'ASCEND_RT_VISIBLE_DEVICES={value}\n')
    print(
        f'setup: run step ASCEND_RT_VISIBLE_DEVICES={value}',
        flush=True,
    )


def _write_probed(tool: str) -> None:
    env = _probe_env()
    smi_ids = _phy_ids_from_npu_smi()
    probe_ids: list[int] = []
    for phy in [*smi_ids, *range(PROBE_LIMIT)]:
        if phy not in probe_ids:
            probe_ids.append(phy)
    resolved: dict[int, str] = {}
    for phy in probe_ids:
        ip = _query_ip(tool, phy, env)
        if ip:
            resolved[phy] = ip
    if len(resolved) < CARDS_NEEDED:
        found = ','.join(str(phy) for phy in sorted(resolved)) or 'none'
        raise SystemExit(
            f'setup: hccn_tool resolved {len(resolved)} device IP(s) '
            f'[{found}], need {CARDS_NEEDED}')
    with open(HCCN_CONF, 'w', encoding='utf-8') as handle:
        for phy in sorted(resolved):
            handle.write(f'address_{phy}={resolved[phy]}\n')
    chosen = _choose_visible(resolved, smi_ids)
    print(
        f'setup: wrote {HCCN_CONF} from hccn_tool '
        f'({len(resolved)} address lines, visible {chosen})',
        flush=True,
    )


def _addresses_in_conf() -> dict[int, str]:
    resolved: dict[int, str] = {}
    with open(HCCN_CONF, encoding='utf-8') as handle:
        for line in handle:
            text = line.strip()
            if not text.startswith('address_') or '=' not in text:
                continue
            key, _, ip = text.partition('=')
            phy_text = key[len('address_'):]
            if phy_text.isdigit() and ip.strip():
                resolved[int(phy_text)] = ip.strip()
    return resolved


def _align_visible_devices() -> None:
    """Point the run step at two physical ids this file actually contains."""
    if not os.path.isfile(HCCN_CONF):
        return
    resolved = _addresses_in_conf()
    if len(resolved) < CARDS_NEEDED:
        return
    chosen = _choose_visible(resolved, _phy_ids_from_npu_smi())
    _publish_visible_devices(chosen)


def ensure() -> None:
    if os.path.isfile(HCCN_CONF):
        print(f'setup: {HCCN_CONF} already present', flush=True)
    elif _flatten_dir_shaped_hccn():
        pass
    elif _copy_driver_conf():
        pass
    else:
        # setup_example.sh puts driver/tools on PATH before calling this.
        tool = shutil.which('hccn_tool')
        if tool is None:
            raise SystemExit(
                'setup: no hccn_tool; cannot materialize /etc/hccn.conf')
        _write_probed(tool)
    _align_visible_devices()


if __name__ == '__main__':
    ensure()
