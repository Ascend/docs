"""Model-hub cache validation: detect / purge stale safetensors shards.

Used by ``prepare_environment`` of every quick-start test that
downloads model weights via ``modelscope`` or HuggingFace Hub.
The runner bind-mounts a host-side cache so it survives across CI
runs — useful for hits, but stale files (from kill -9 / OOM /
network drop) leak into the next run and trip
``safetensors.safe_open`` on deserialization.

This module validates each shard's header with the native
``safetensors`` loader and ``rmtree``-s the parent model dir on
any failure; the originating library will re-download the whole
model on next access. ``report_<provider>_state`` is also exposed
for pre-flight logging.

Provider split
--------------

The module is organized by *provider* (``modelscope`` /
``huggingface``) rather than a single dispatcher: callers pick the
provider explicitly via the function they import. Reasons:

- Only two providers are in scope; a dispatch table is overkill.
- Explicit imports make the project's chosen provider visible at
  the call site — no hidden resolution.
- Adding a third provider (e.g. OpenMind, HF 国内镜像) means adding
  the same four functions (``resolve_<provider>_cache``,
  ``report_<provider>_state``, ``purge_<provider>_corrupt``), no
  dispatcher surgery.

Helpers shared across providers
-------------------------------

- ``ensure_safetensors`` — install ``safetensors`` if the CANN
  base image doesn't ship it.
- ``safetensors_header_ok`` — validate one shard via
  ``safe_open``. Provider-agnostic because the safetensors file
  format is the same regardless of which hub served it.
- ``_curl_throughput_probe`` — shared throughput probe used by
  both the modelscope.cn and hf-mirror.com reachability checks.
- ``diagnose_mount_environment`` — mount / capability / network
  diagnostics. The HF cache probe inside already speaks HF layout.

Layouts
-------

ModelScope (``$MODELSCOPE_CACHE`` / ``~/.cache/modelscope``)::

    <cache_root>/hub/models/<org>/<model>/<revision>/*.safetensors

Dotted model ids (``.`` → ``___``) live in a masked dir with a
symlink of the original name; we walk the masked dir directly.

HuggingFace Hub (``$HF_HUB_CACHE`` / ``~/.cache/huggingface/hub``)::

    <cache_root>/models--<org>--<model>/
        blobs/<sha256>           # real file content
        snapshots/<commit_sha>/  # symlinks to blobs + small files
            *.safetensors
            model.safetensors.index.json
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

# Cap on per-section file listings inside ``report_*_state`` so a
# model with hundreds of tokenizer shards (e.g. some multilingual
# tokenizers ship 100+ ``merges.txt``-style files) doesn't flood the
# CI log. The total count + total size always print, so truncation is
# obvious.
_MAX_LISTED_FILES = 10


# ======================================================================
# Provider-agnostic helpers
# ======================================================================


def ensure_safetensors() -> None:
    """Defensively install ``safetensors`` if not already importable.

    The CANN base image used by the runner may not ship
    ``safetensors``; torch < 4.20 doesn't hard-depend on it. The
    validation helpers below need it, so callers should invoke
    this once in their ``prepare_environment`` before
    ``purge_<provider>_corrupt``. No-op when already installed.

    Inherits the parent env, so any ``PIP_INDEX_URL`` /
    ``PIP_CONSTRAINT`` / ``UV_*`` configured by the workflow
    carries through to the install.
    """
    try:
        import safetensors  # noqa: F401
    except ImportError:
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'safetensors'],
            check=True,
        )


def safetensors_header_ok(path: Path) -> bool:
    """Use safetensors' native loader to validate the file header.
    Returns True iff ``safe_open`` accepts the file (header parses,
    tensor offsets fit within the file). ``SafetensorError`` or
    ``OSError`` means the shard is unusable.

    Lazy import: the module itself loads fine on machines that
    don't have ``safetensors`` installed yet — the import only
    fires when a caller actually invokes this function. Callers
    should run ``ensure_safetensors()`` once first if they intend
    to call this in an environment where ``safetensors`` may be
    missing.

    Provider-agnostic: the safetensors file format is the same
    whether the file was served by modelscope, HuggingFace Hub,
    or any other mirror that ships identical weights.
    """
    from safetensors import safe_open, SafetensorError  # noqa: I001
    try:
        # framework='numpy' instead of 'pt': only ``.keys()`` is read so
        # the framework is a no-op for our use case, but safetensors
        # 0.8.0 makes ``framework`` required and 'pt' would force
        # ``import torch`` — bare CANN 9.1.0 image doesn't ship torch.
        with safe_open(str(path), framework='numpy') as f:
            list(f.keys())  # force header read
    except (SafetensorError, OSError):
        return False
    return True


def _curl_throughput_probe(url: str, label: str) -> None:
    """Run ``curl -w`` against ``url``, print per-phase timing +
    throughput + verdict. Shared by the modelscope and HF probes so
    both report in the same format for easy side-by-side comparison.

    Verdict thresholds:
      > 10 MiB/s      healthy
      1 – 10 MiB/s    slow — likely server-side throttling
      < 1 MiB/s       very slow — investigate network path
    """
    print(f'  network throughput (GET {label}):')
    try:
        out = subprocess.run(
            [
                'curl', '-fsSL',
                '--max-time', '60',
                '-o', '/dev/null',
                '-w',
                (
                    '%{time_namelookup}|%{time_connect}|'
                    '%{time_appconnect}|%{time_starttransfer}|'
                    '%{time_total}|%{speed_download}|'
                    '%{http_code}|%{size_download}'
                ),
                url,
            ],
            capture_output=True, text=True, check=False, timeout=65,
        )
        if out.returncode == 0 and out.stdout.strip():
            parts = out.stdout.strip().split('|')
            if len(parts) == 8:
                dns, conn, ssl, ttfb, total, speed_bps, http, size = parts
                print(
                    f'    phases: dns={dns}s connect={conn}s '
                    f'ssl={ssl}s ttfb={ttfb}s total={total}s '
                    f'http={http} size={size}B'
                )
                try:
                    speed_mib = float(speed_bps) / (1024 * 1024)
                    speed_mbs = float(speed_bps) / 1_000_000
                    print(
                        f'    → avg throughput: '
                        f'{speed_mib:.2f} MiB/s ({speed_mbs:.2f} MB/s)'
                    )
                    if speed_mib >= 10:
                        verdict = 'healthy (>10 MiB/s)'
                    elif speed_mib >= 1:
                        verdict = (
                            'slow (1-10 MiB/s) — likely server-side '
                            'throttling'
                        )
                    else:
                        verdict = (
                            'very slow (<1 MiB/s) — investigate DNS / '
                            'firewall / cluster egress'
                        )
                    print(f'    verdict: {verdict}')
                except ValueError:
                    print(f'    numeric parse failed: {speed_bps!r}')
            else:
                print(f'    unexpected format: {out.stdout.strip()[:200]}')
        elif out.returncode == 0:
            print('    empty output')
        else:
            print(
                f'    curl failed (rc={out.returncode}): '
                f'{out.stderr.strip()[:200]}'
            )
    except FileNotFoundError:
        print('    (curl not available)')
    except subprocess.TimeoutExpired:
        print('    (curl timed out)')


def diagnose_mount_environment(
    cache_root: Path | None = None,
    model_id: str | None = None,
) -> None:
    """Probe the container's view of ``cache_root`` to disambiguate
    "the bind mount isn't visible" cases.

    Companion to ``report_<provider>_state`` — called from
    ``prepare_environment`` first so the cache-state log has its
    environment context. Provider-agnostic: takes whatever
    ``cache_root`` the caller passes (modelscope or HF).

    Each probe answers a specific question:

    - ``findmnt -T <path>`` — canonical "is this a mountpoint".
      Empty output means it isn't.
    - ``stat -c '%d:%i'`` on the path *and* its parent — mountpoint
      detection without ``findmnt``: a separate mountpoint lives on
      a different device (``st_dev``) than its parent.
    - ``/proc/self/mountinfo`` — kernel-level mount table, more
      authoritative than ``mount`` (which reads ``/etc/mtab`` /
      ``/proc/mounts`` and can be filtered in some runtimes).
    - ``hostname`` + ``/proc/1/cgroup`` — confirm we're inside a
      Kubernetes pod (cgroup paths contain ``kubepods/``);
      bind-mount behavior differs there.
    - ``CapBnd`` from ``/proc/self/status`` — ``CAP_SYS_ADMIN``
      (bit 21) is required for bind mounts on most runtimes;
      absence explains silent bind-mount failures.
    - ``df -h`` — backing filesystem / size for when the path *is*
      a mountpoint (useful sanity check that it's actually
      persistent).
    - ``curl`` throughput to modelscope.cn — disambiguates "the
      download is slow" (see :func:`_curl_throughput_probe`).
    - HF default cache + ``curl`` throughput to hf-mirror.com —
      fallback-mirror probe, used to decide whether to swap
      ``snapshot_download`` providers. Only runs when ``model_id``
      is provided.

    All probes are read-only and have 5–65 s timeouts; missing tools
    print ``(not available)`` rather than raising.
    """
    cache_root = cache_root or resolve_modelscope_cache()
    print('mount-diag: probing container mount environment')

    # findmnt — canonical "is this a mountpoint"
    print(f'  findmnt -T {cache_root}:')
    try:
        out = subprocess.run(
            ['findmnt', '-T', str(cache_root)],
            capture_output=True, text=True, check=False, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            for line in out.stdout.splitlines():
                print(f'    {line}')
        elif out.returncode == 0:
            print('    (empty — path is NOT a mountpoint)')
        else:
            print(
                f'    findmnt failed (rc={out.returncode}): '
                f'{out.stderr.strip()[:200]}'
            )
    except FileNotFoundError:
        print('    (findmnt not available)')
    except subprocess.TimeoutExpired:
        print('    (findmnt timed out)')

    # stat (dev, inode) — corroborating mountpoint check
    print(f'  stat -c "%d:%i" {cache_root} vs parent:')
    try:
        path_stat = cache_root.stat()
        parent_stat = cache_root.parent.stat()
        print(
            f'    {cache_root}: '
            f'dev={path_stat.st_dev} inode={path_stat.st_ino}'
        )
        print(
            f'    {cache_root.parent}: '
            f'dev={parent_stat.st_dev} inode={parent_stat.st_ino}'
        )
        if path_stat.st_dev != parent_stat.st_dev:
            print(
                '    → different device than parent → mountpoint '
                '(separate filesystem)'
            )
        else:
            print(
                '    → same device as parent → NOT a mountpoint '
                '(regular directory on the parent fs)'
            )
    except OSError as e:
        print(f'    stat failed: {e}')

    # /proc/self/mountinfo — kernel view
    print(f'  /proc/self/mountinfo entries for {cache_root}:')
    try:
        with open('/proc/self/mountinfo', encoding='utf-8') as fh:
            matches = [
                line.rstrip() for line in fh if str(cache_root) in line
            ]
        if matches:
            for line in matches[:_MAX_LISTED_FILES]:
                print(f'    {line}')
            if len(matches) > _MAX_LISTED_FILES:
                print(
                    f'    ... and {len(matches) - _MAX_LISTED_FILES} more'
                )
        else:
            print(f'    (no entry for {cache_root})')
    except OSError as e:
        print(f'    read /proc/self/mountinfo failed: {e}')

    # host identity — K8s detection
    print('  host identity:')
    try:
        out = subprocess.run(
            ['hostname'],
            capture_output=True, text=True, check=False, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            print(f'    hostname: {out.stdout.strip()}')
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        with open('/proc/1/cgroup', encoding='utf-8') as fh:
            cgroup_lines = fh.read().splitlines()
        is_k8s = any('kubepods' in line for line in cgroup_lines)
        print(
            f'    /proc/1/cgroup: {"K8s pod" if is_k8s else "non-K8s"}'
        )
        for line in cgroup_lines[:_MAX_LISTED_FILES]:
            print(f'      {line}')
        if len(cgroup_lines) > _MAX_LISTED_FILES:
            print(
                f'      ... and {len(cgroup_lines) - _MAX_LISTED_FILES} more'
            )
    except OSError as e:
        print(f'    read /proc/1/cgroup failed: {e}')

    # capabilities — CAP_SYS_ADMIN (bit 21) required for bind mounts
    print('  capabilities (from /proc/self/status):')
    try:
        cap_lines: list[str] = []
        cap_bnd: int | None = None
        with open('/proc/self/status', encoding='utf-8') as fh:
            for line in fh:
                if line.startswith('Cap'):
                    cap_lines.append(line.rstrip())
                if line.startswith('CapBnd:'):
                    cap_bnd = int(line.split()[1], 16)
        if cap_lines:
            for line in cap_lines:
                print(f'    {line}')
        else:
            print('    (no Cap lines in /proc/self/status)')
        if cap_bnd is not None:
            has_sys_admin = bool(cap_bnd & (1 << 21))
            print(
                f'    CAP_SYS_ADMIN (bit 21): '
                f'{"YES" if has_sys_admin else "NO"} '
                f'(CapBnd=0x{cap_bnd:x})'
            )
    except OSError as e:
        print(f'    read /proc/self/status failed: {e}')

    # df — backing filesystem / size
    print(f'  df -h {cache_root}:')
    try:
        out = subprocess.run(
            ['df', '-h', str(cache_root)],
            capture_output=True, text=True, check=False, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            for line in out.stdout.splitlines():
                print(f'    {line}')
        elif out.returncode == 0:
            print('    (empty)')
        else:
            print(f'    df failed: {out.stderr.strip()[:200]}')
    except FileNotFoundError:
        print('    (df not available)')
    except subprocess.TimeoutExpired:
        print('    (df timed out)')

    # Network throughput — disambiguates "the download is slow".
    # 7 MiB ``tokenizer.json`` is the probe: large enough to escape
    # TCP slow-start, small enough to finish in seconds on a
    # healthy link; ``%{speed_download}`` includes DNS+connect time
    # so a slow DNS still shows up as low throughput.
    _curl_throughput_probe(
        'https://www.modelscope.cn/Qwen/Qwen2.5-3B-Instruct/'
        'resolve/master/tokenizer.json',
        'modelscope.cn/tokenizer.json',
    )

    # HF fallback probe — only runs when caller passes ``model_id``.
    # Checks (a) whether the model is already in HF's default cache
    # ``~/.cache/huggingface/hub/`` so a future switch to
    # ``huggingface_hub.snapshot_download`` would be a no-op download,
    # and (b) throughput to ``hf-mirror.com`` so we can compare against
    # modelscope.cn and decide whether the swap is worth it.
    if model_id:
        # HF cache layout: ~/.cache/huggingface/hub/models--<org>--<model>/.
        # The ``/`` separator in the model id becomes ``--`` in the dir name.
        hf_cache_root = resolve_huggingface_cache()
        hf_model_dir_name = 'models--' + model_id.replace('/', '--')
        hf_model_dir = hf_cache_root / hf_model_dir_name
        print(f'  HF default cache ({hf_cache_root}/{hf_model_dir_name}):')
        if hf_model_dir.is_dir():
            print(f'    present: {hf_model_dir}')
            snapshots_dir = hf_model_dir / 'snapshots'
            if snapshots_dir.is_dir():
                snapshots = [s for s in snapshots_dir.iterdir() if s.is_dir()]
                for snap in sorted(snapshots)[:_MAX_LISTED_FILES]:
                    safetensors = sorted(snap.glob('*.safetensors'))
                    if safetensors:
                        total_gb = sum(
                            f.stat().st_size for f in safetensors
                        ) / (1 << 30)
                        print(
                            f'    snapshot {snap.name}: '
                            f'{len(safetensors)} safetensor(s), '
                            f'{total_gb:.2f} GB total'
                        )
                        for f in safetensors[:_MAX_LISTED_FILES]:
                            size_gb = f.stat().st_size / (1 << 30)
                            print(f'      {f.name}  {size_gb:.2f} GB')
                if len(snapshots) > _MAX_LISTED_FILES:
                    print(
                        f'    ... and {len(snapshots) - _MAX_LISTED_FILES} '
                        f'more snapshots'
                    )
            else:
                print('    (no snapshots dir)')
        else:
            print(f'    not present: {hf_model_dir}')

        # Throughput to huggingface.co (HF's primary endpoint, which is
        # what ``huggingface_hub.snapshot_download`` hits by default).
        # Probe the *original* — not hf-mirror.com or any other mirror —
        # so the comparison against modelscope.cn above is apples-to-
        # apples (primary-of-A vs primary-of-B). Mirror speeds are a
        # different question; the decision at hand is "should we switch
        # providers", which is governed by primary throughput.
        _curl_throughput_probe(
            f'https://huggingface.co/{model_id}/resolve/main/tokenizer.json',
            'huggingface.co/tokenizer.json',
        )
    else:
        print('  HF fallback probe: skipped (model_id not provided)')


# ======================================================================
# Provider: ModelScope
# ======================================================================


def resolve_modelscope_cache() -> Path:
    """Return the modelscope cache root the same way modelscope does:
    prefer ``$MODELSCOPE_CACHE`` if set, otherwise
    ``~/.cache/modelscope``. Computed at call time (not import time)
    so tests / subprocesses that mutate the env get the right value."""
    return Path(
        os.environ.get('MODELSCOPE_CACHE')
        or str(Path.home() / '.cache' / 'modelscope')
    )


def report_modelscope_state(
    model_id: str, cache_root: Path | None = None,
) -> bool:
    """Pre-flight log for ``model_id``; returns whether every expected
    shard is present and header-valid.

    Called from ``prepare_environment`` right before
    ``snapshot_download`` so the CI log shows per-shard sizes +
    validity (not just the aggregate ``cache: hit`` line from
    ``purge_modelscope_corrupt``).

    Walks the masked dir (``.`` → ``___``): ``Qwen/Qwen2.5-3B-Instruct``
    lives at ``.../Qwen/Qwen2___5-3B-Instruct/`` on disk.

    Stricter than ``snapshot_download`` (existence + size only):
    parses the safetensors header to confirm tensor offsets fit —
    catches truncated-content files the size check would silently
    accept.

    Returns True iff all expected shards (from ``index.json``) are
    present and valid; single-file models fall back to "≥1 valid
    shard". Does not modify disk state.
    """
    cache_root = cache_root or resolve_modelscope_cache()
    org, _, model = model_id.partition('/')
    if not org or not model:
        raise ValueError(
            f'model_id must be "<org>/<model>", got {model_id!r}'
        )
    # Modelscope masks dotted model ids on disk: ``Qwen2.5-...`` →
    # ``Qwen2___5-...``. The original-id dir is a symlink; we walk the
    # masked dir directly so we don't depend on symlink resolution.
    masked_model = model.replace('.', '___')
    masked_dir = cache_root / 'hub' / 'models' / org / masked_model

    print(f'cache: pre-flight for {model_id}')
    print(f'  masked dir: {masked_dir}')

    if not masked_dir.is_dir():
        print('  state: MISS — directory not present')
        return False

    safetensors_shards: list[Path] = []
    other_files: list[Path] = []
    for p in sorted(masked_dir.rglob('*')):
        if not p.is_file():
            continue
        if p.suffix == '.safetensors':
            safetensors_shards.append(p)
        else:
            other_files.append(p)

    # Determine *expected* shard set. Sharded models ship a
    # ``model.safetensors.index.json`` whose ``weight_map`` lists every
    # shard file by name — that's the ground truth for "did the download
    # actually finish". Single-file models have no index.json; we fall
    # back to "any safetensors file counts" for those.
    expected_shard_names: set[str] = set()
    for idx_path in masked_dir.rglob('model.safetensors.index.json'):
        try:
            weight_map = json.loads(idx_path.read_text()).get('weight_map', {})
        except (json.JSONDecodeError, OSError):
            continue
        expected_shard_names.update(weight_map.values())

    found_shard_names = {p.name for p in safetensors_shards}
    missing_shard_names = (
        expected_shard_names - found_shard_names
        if expected_shard_names
        else set()
    )

    # Validate present shards once; reuse for both per-shard log and
    # aggregate return value.
    shard_status: list[tuple[Path, bool]] = [
        (p, safetensors_header_ok(p)) for p in safetensors_shards
    ]
    valid_count = sum(1 for _, ok in shard_status if ok)
    corrupt_count = len(shard_status) - valid_count

    shard_bytes = sum(p.stat().st_size for p in safetensors_shards)
    other_bytes = sum(p.stat().st_size for p in other_files)
    total_expected_shards = len(safetensors_shards) + len(missing_shard_names)
    summary = (
        f'  safetensors: {len(safetensors_shards)}'
        + (f'/{total_expected_shards} expected' if expected_shard_names else '')
        + ' shard(s) present, '
        + f'{shard_bytes / (1 << 30):.2f} GB total, '
        + f'{valid_count} valid / {corrupt_count} corrupt'
        + (
            f', {len(missing_shard_names)} missing'
            if missing_shard_names else ''
        )
    )
    print(summary)
    for shard, ok in shard_status[:_MAX_LISTED_FILES]:
        size_gb = shard.stat().st_size / (1 << 30)
        marker = 'OK' if ok else 'CORRUPT'
        rel = shard.relative_to(masked_dir)
        print(f'    [{marker}] {rel}  {size_gb:.2f} GB')
    if len(shard_status) > _MAX_LISTED_FILES:
        print(f'    ... and {len(shard_status) - _MAX_LISTED_FILES} more')
    for name in sorted(missing_shard_names)[:_MAX_LISTED_FILES]:
        print(f'    [MISSING] {name}')
    if len(missing_shard_names) > _MAX_LISTED_FILES:
        print(
            f'    ... and {len(missing_shard_names) - _MAX_LISTED_FILES} '
            f'more missing'
        )

    print(
        f'  other files: {len(other_files)} '
        f'({other_bytes / (1 << 20):.1f} MB)'
    )
    for p in other_files[:_MAX_LISTED_FILES]:
        rel = p.relative_to(masked_dir)
        size_kb = p.stat().st_size / (1 << 10)
        print(f'    {rel}  {size_kb:.1f} KB')
    if len(other_files) > _MAX_LISTED_FILES:
        print(f'    ... and {len(other_files) - _MAX_LISTED_FILES} more')

    # Diagnostics: raw `ls` of the masked dir, for when "CI says X
    # but my ls says Y" — typically a bind-mount issue. Print before
    # the state line so the conclusion lines up with the evidence.
    # Mount-level diagnostics live in ``diagnose_mount_environment``
    # (called separately from ``prepare_environment``).
    print(f'  ls -la {masked_dir}/:')
    try:
        ls = subprocess.run(
            ['ls', '-la', f'{masked_dir}/'],
            capture_output=True, text=True, check=False, timeout=5,
        )
        if ls.returncode == 0:
            lines = ls.stdout.splitlines()
            for line in lines[:_MAX_LISTED_FILES + 5]:
                print(f'    {line}')
            if len(lines) > _MAX_LISTED_FILES + 5:
                print(
                    f'    ... and {len(lines) - _MAX_LISTED_FILES - 5} '
                    f'more lines'
                )
        else:
            print(
                f'    ls failed (rc={ls.returncode}): '
                f'{ls.stderr.strip()[:200]}'
            )
    except FileNotFoundError:
        print('    (ls not available)')
    except subprocess.TimeoutExpired:
        print('    (ls timed out)')

    # HIT requires: (a) every expected shard present, (b) every
    # present shard header-valid. When no index.json is present
    # (single-file model), fall back to "at least one valid shard".
    if expected_shard_names:
        all_present = not missing_shard_names
        all_valid = all_present and corrupt_count == 0
    else:
        all_valid = bool(shard_status) and corrupt_count == 0
    if all_valid:
        state_msg = 'HIT (all shards present + valid)'
    elif missing_shard_names and corrupt_count == 0:
        state_msg = (
            f'PARTIAL — {len(missing_shard_names)} shard(s) missing; '
            f'modelscope will download missing on next step'
        )
    elif corrupt_count and missing_shard_names:
        state_msg = (
            f'PARTIAL/CORRUPT — {corrupt_count} corrupt + '
            f'{len(missing_shard_names)} missing; purge + download on next step'
        )
    else:
        state_msg = (
            f'CORRUPT — {corrupt_count} shard(s) bad; '
            f'purge + re-download on next step'
        )
    print(f'  state: {state_msg}')
    return all_valid


def purge_modelscope_corrupt(cache_root: Path) -> None:
    """Scan every ``*.safetensors`` file under each model dir and purge
    the model dir if any shard is corrupt. ``modelscope`` will
    re-download the whole model on next access. No-op when the
    cache root is absent (fresh container, or first-time setup).

    Walks the full model dir (not just ``blobs/``) because
    ModelScope's layout is
    ``<model_dir>/<revision>/*.safetensors`` — unlike HuggingFace
    Hub which uses ``blobs/`` + symlinks. ``safe_open`` resolves
    symlinks transparently, so this also catches a future
    modelscope release that switches to the HF-style layout.

    For dotted model ids (``Qwen2.5-...``), modelscope stores the
    files in a masked dir (``.`` → ``___``) plus a *symlink* named
    after the original id for readability. ``is_dir()`` follows the
    symlink, so without the ``not is_symlink()`` filter both entries
    match: each shard gets validated twice, and when a corrupt shard
    triggers the purge, ``shutil.rmtree`` on the symlink entry raises
    ``OSError`` (rmtree refuses symlinks by design) and crashes
    ``prepare_environment`` — exactly in the scenario the purge
    exists for. Skipping the symlink entries keeps the purge on the
    masked dir only.

    Parameters
    ----------
    cache_root : Path
        The modelscope cache root (i.e. the value of
        ``$MODELSCOPE_CACHE`` or its default ``~/.cache/modelscope``).
        Passed in as a parameter so the function is reusable from
        tests with a tmp dir and doesn't carry an implicit dependency
        on a module-level constant.
    """
    hub_models = cache_root / 'hub' / 'models'
    if not hub_models.exists():
        print(f'cache: miss ({hub_models} not present yet); nothing to validate')
        return
    model_dirs = [
        d for d in hub_models.glob('*/*')
        if d.is_dir() and not d.is_symlink()
    ]
    purged = 0
    for model_dir in model_dirs:
        corrupt = [
            p for p in model_dir.rglob('*.safetensors')
            if not safetensors_header_ok(p)
        ]
        if not corrupt:
            continue
        print(
            f'cache: purging {model_dir.parent.name}/{model_dir.name} '
            f'({len(corrupt)} corrupt shard(s)); modelscope will re-download'
        )
        shutil.rmtree(model_dir)
        purged += 1
    if purged:
        print(
            f'cache: partial — validated {len(model_dirs)} model dir(s), '
            f'purged {purged}'
        )
    else:
        print(
            f'cache: hit {len(model_dirs)} model dir(s), all healthy'
        )


# ======================================================================
# Provider: HuggingFace Hub
# ======================================================================


def resolve_huggingface_cache() -> Path:
    """Return the HuggingFace Hub cache root the same way
    ``huggingface_hub`` does: prefer ``$HF_HUB_CACHE`` if set,
    otherwise ``$HF_HOME/hub`` if ``HF_HOME`` is set, otherwise
    ``~/.cache/huggingface/hub``.

    Resolved at call time (not import time) so tests / subprocesses
    that mutate the env get the right value.

    Note on the default: ``huggingface_hub.constants.HF_HUB_CACHE``
    resolves to ``HF_HOME/hub`` (so ``~/.cache/huggingface/hub``
    when neither var is set). We replicate that resolution here to
    avoid importing ``huggingface_hub`` at module load — the CANN
    runner base image may not ship it yet, and
    ``prepare_environment`` runs before the doc installs it.
    """
    hub_cache = os.environ.get('HF_HUB_CACHE')
    if hub_cache:
        return Path(hub_cache)
    hf_home = os.environ.get('HF_HOME')
    if hf_home:
        return Path(hf_home) / 'hub'
    return Path.home() / '.cache' / 'huggingface' / 'hub'


def _huggingface_model_dir(
    model_id: str, cache_root: Path,
) -> Path:
    """Resolve the on-disk model dir for ``model_id`` under HF's
    default layout: ``<cache_root>/models--<org>--<model>/``.

    HF replaces the ``/`` in the model id with ``--`` to flatten
    the namespace into a single dir name. We don't validate the
    id here — callers (report/purge) handle missing dirs cleanly.
    """
    return cache_root / ('models--' + model_id.replace('/', '--'))


def report_huggingface_state(
    model_id: str, cache_root: Path | None = None,
) -> bool:
    """Pre-flight log for ``model_id`` against the HF Hub cache;
    returns whether every expected shard is present and header-valid.

    Layout walked::

        <cache_root>/models--<org>--<model>/
            refs/<rev>                 # tiny file holding commit SHA
            snapshots/<commit_sha>/
                *.safetensors           # symlinks into blobs/
                model.safetensors.index.json
            blobs/<sha256>              # real file content

    For sharded models the expected shard set is read from
    ``model.safetensors.index.json`` (just like the modelscope
    report). For single-file models we fall back to "≥1 valid
    shard" since HF doesn't ship an index.json in that case.

    Revision resolution: HF may have multiple snapshot dirs under
    ``snapshots/`` (one per commit), e.g. ``main``, a tagged
    release SHA, or a previous install's pinned revision. We pick
    the lexicographically first snapshot dir — in practice this is
    ``main`` (created by ``huggingface_hub`` when ``revision`` is
    not specified) since its name sorts before any 40-char SHA.
    Callers that pin a specific revision should set
    ``HF_HUB_CACHE`` to a per-revision cache root instead.

    Stricter than ``snapshot_download`` (existence + size only):
    parses the safetensors header to confirm tensor offsets fit —
    catches truncated-content files the size check would silently
    accept.

    Returns True iff all expected shards (from ``index.json``) are
    present and valid; single-file models fall back to "≥1 valid
    shard". Does not modify disk state.
    """
    cache_root = cache_root or resolve_huggingface_cache()
    org, _, model = model_id.partition('/')
    if not org or not model:
        raise ValueError(
            f'model_id must be "<org>/<model>", got {model_id!r}'
        )
    model_dir = _huggingface_model_dir(model_id, cache_root)
    snapshots_dir = model_dir / 'snapshots'

    print(f'cache: pre-flight for {model_id} (HF)')
    print(f'  model dir: {model_dir}')

    if not model_dir.is_dir():
        print('  state: MISS — model dir not present')
        return False

    if not snapshots_dir.is_dir():
        print(f'  state: MISS — snapshots dir not present ({snapshots_dir})')
        return False

    # Pick a snapshot dir. Prefer 'main' if present (it's the
    # default revision ``snapshot_download`` uses), otherwise the
    # lexicographically first entry. 'main' sorts before any 40-char
    # SHA, but an explicit lexicographic pick still gives a stable
    # answer when 'main' is absent.
    snapshots = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
    if not snapshots:
        print('  state: MISS — no snapshot revisions')
        return False
    rev_dir = next(
        (p for p in snapshots if p.name == 'main'),
        snapshots[0],
    )
    print(f'  revision dir: {rev_dir}')

    safetensors_shards: list[Path] = sorted(rev_dir.glob('*.safetensors'))
    # ``other_files`` in the rev dir: tokenizer / config / etc.
    # Useful for "did the download at least *start*" sanity check,
    # since on a partial download the small JSONs land first and
    # the weight shards come later.
    other_files: list[Path] = [
        p for p in rev_dir.iterdir()
        if p.is_file() and p.suffix != '.safetensors'
    ]

    expected_shard_names: set[str] = set()
    index_path = rev_dir / 'model.safetensors.index.json'
    if index_path.is_file():
        try:
            weight_map = json.loads(index_path.read_text()).get(
                'weight_map', {},
            )
        except (json.JSONDecodeError, OSError):
            weight_map = {}
        expected_shard_names.update(weight_map.values())

    found_shard_names = {p.name for p in safetensors_shards}
    missing_shard_names = (
        expected_shard_names - found_shard_names
        if expected_shard_names
        else set()
    )

    shard_status: list[tuple[Path, bool]] = [
        (p, safetensors_header_ok(p)) for p in safetensors_shards
    ]
    valid_count = sum(1 for _, ok in shard_status if ok)
    corrupt_count = len(shard_status) - valid_count

    shard_bytes = sum(p.stat().st_size for p in safetensors_shards)
    other_bytes = sum(p.stat().st_size for p in other_files)
    total_expected_shards = len(safetensors_shards) + len(missing_shard_names)
    summary = (
        f'  safetensors: {len(safetensors_shards)}'
        + (f'/{total_expected_shards} expected' if expected_shard_names else '')
        + ' shard(s) present, '
        + f'{shard_bytes / (1 << 30):.2f} GB total, '
        + f'{valid_count} valid / {corrupt_count} corrupt'
        + (
            f', {len(missing_shard_names)} missing'
            if missing_shard_names else ''
        )
    )
    print(summary)
    for shard, ok in shard_status[:_MAX_LISTED_FILES]:
        size_gb = shard.stat().st_size / (1 << 30)
        marker = 'OK' if ok else 'CORRUPT'
        # In HF layout the rev_dir holds symlinks; resolve so the
        # "real size" matches what blobs/ actually has on disk.
        target = shard.resolve()
        print(f'    [{marker}] {shard.name}  {size_gb:.2f} GB'
              + (f'  -> {target}' if target != shard else ''))
    if len(shard_status) > _MAX_LISTED_FILES:
        print(f'    ... and {len(shard_status) - _MAX_LISTED_FILES} more')
    for name in sorted(missing_shard_names)[:_MAX_LISTED_FILES]:
        print(f'    [MISSING] {name}')
    if len(missing_shard_names) > _MAX_LISTED_FILES:
        print(
            f'    ... and {len(missing_shard_names) - _MAX_LISTED_FILES} '
            f'more missing'
        )

    print(
        f'  other files: {len(other_files)} '
        f'({other_bytes / (1 << 20):.1f} MB)'
    )
    for p in sorted(other_files)[:_MAX_LISTED_FILES]:
        size_kb = p.stat().st_size / (1 << 10)
        print(f'    {p.name}  {size_kb:.1f} KB')
    if len(other_files) > _MAX_LISTED_FILES:
        print(f'    ... and {len(other_files) - _MAX_LISTED_FILES} more')

    # Diagnostics: raw `ls` of the rev dir. Same motivation as the
    # modelscope report — when "CI says X but my ls says Y" comes
    # up, the bind-mount state is usually to blame.
    print(f'  ls -la {rev_dir}/:')
    try:
        ls = subprocess.run(
            ['ls', '-la', f'{rev_dir}/'],
            capture_output=True, text=True, check=False, timeout=5,
        )
        if ls.returncode == 0:
            lines = ls.stdout.splitlines()
            for line in lines[:_MAX_LISTED_FILES + 5]:
                print(f'    {line}')
            if len(lines) > _MAX_LISTED_FILES + 5:
                print(
                    f'    ... and {len(lines) - _MAX_LISTED_FILES - 5} '
                    f'more lines'
                )
        else:
            print(
                f'    ls failed (rc={ls.returncode}): '
                f'{ls.stderr.strip()[:200]}'
            )
    except FileNotFoundError:
        print('    (ls not available)')
    except subprocess.TimeoutExpired:
        print('    (ls timed out)')

    if expected_shard_names:
        all_present = not missing_shard_names
        all_valid = all_present and corrupt_count == 0
    else:
        all_valid = bool(shard_status) and corrupt_count == 0
    if all_valid:
        state_msg = 'HIT (all shards present + valid)'
    elif missing_shard_names and corrupt_count == 0:
        state_msg = (
            f'PARTIAL — {len(missing_shard_names)} shard(s) missing; '
            f'huggingface_hub will download missing on next step'
        )
    elif corrupt_count and missing_shard_names:
        state_msg = (
            f'PARTIAL/CORRUPT — {corrupt_count} corrupt + '
            f'{len(missing_shard_names)} missing; purge + download on next step'
        )
    else:
        state_msg = (
            f'CORRUPT — {corrupt_count} shard(s) bad; '
            f'purge + re-download on next step'
        )
    print(f'  state: {state_msg}')
    return all_valid


def purge_huggingface_corrupt(cache_root: Path) -> None:
    """Scan every ``*.safetensors`` file under each model dir in HF's
    cache and purge the model dir if any shard is corrupt. HF will
    re-download the whole model on next access. No-op when the
    cache root is absent.

    HF layout (default)::

        <cache_root>/models--<org>--<model>/
            blobs/<sha256>           # real content (the one we read)
            snapshots/<commit_sha>/  # symlinks into blobs/

    We validate the *symlinks* under ``snapshots/<rev>/`` rather
    than the blobs directly because the snapshots are the
    user-visible file paths and the entry point HF consults on
    next access. ``safetensors.safe_open`` follows symlinks
    transparently, so a blob with a corrupt header trips the
    purge even though we never opened the blob by name.

    Purge target is the *model dir* (``models--<org>--<model>/``),
    not the snapshot dir: blobs/ holds the only copy of the file
    content and a model-level purge is what ``snapshot_download``
    recreates cleanly on next access.

    Parameters
    ----------
    cache_root : Path
        The HF Hub cache root (i.e. ``$HF_HUB_CACHE`` or its
        default ``~/.cache/huggingface/hub``).
    """
    if not cache_root.exists():
        print(f'cache: miss ({cache_root} not present yet); nothing to validate')
        return
    model_dirs = [
        d for d in cache_root.glob('models--*')
        if d.is_dir() and not d.is_symlink()
    ]
    purged = 0
    for model_dir in model_dirs:
        # Walk every snapshot revision — a model with multiple
        # revisions can have one corrupt one and one healthy one;
        # we still purge the whole model dir since ``snapshot_download``
        # recreates both consistently.
        snapshots_dir = model_dir / 'snapshots'
        if not snapshots_dir.is_dir():
            continue
        corrupt: list[Path] = []
        for snap in snapshots_dir.iterdir():
            if not snap.is_dir():
                continue
            for p in snap.glob('*.safetensors'):
                if not safetensors_header_ok(p):
                    corrupt.append(p)
        if not corrupt:
            continue
        print(
            f'cache: purging {model_dir.name} '
            f'({len(corrupt)} corrupt shard(s)); '
            f'huggingface_hub will re-download'
        )
        shutil.rmtree(model_dir)
        purged += 1
    if purged:
        print(
            f'cache: partial — validated {len(model_dirs)} model dir(s), '
            f'purged {purged}'
        )
    else:
        print(
            f'cache: hit {len(model_dirs)} model dir(s), all healthy'
        )
