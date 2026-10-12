"""Copy CUDA_VISIBLE_DEVICES onto ASCEND_RT_VISIBLE_DEVICES at interpreter start.

Upstream examples mask cards with CUDA_VISIBLE_DEVICES on the python or
torchrun command line. torch_npu reads ASCEND_RT_VISIBLE_DEVICES. The job
exports the whole container slice on ASCEND_RT_VISIBLE_DEVICES, so without
this copy the trainer would see the vLLM cards as well.

vLLM then narrows ASCEND_RT_VISIBLE_DEVICES again for each data-parallel
child, while CUDA_VISIBLE_DEVICES stays the wider parent mask. Replacing
that narrower list puts every child on the first card.
"""
import os
from collections.abc import MutableMapping


def _device_ids(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(',') if part.strip()]


def ascend_mask_is_narrower(ascend: str, cuda: str) -> bool:
    """True when ascend is a non-empty subset of cuda and no longer."""
    ascend_ids = _device_ids(ascend)
    cuda_ids = _device_ids(cuda)
    if not ascend_ids or not cuda_ids:
        return False
    return set(ascend_ids).issubset(cuda_ids) and len(ascend_ids) <= len(cuda_ids)


def sync_ascend_visible_devices(env: MutableMapping[str, str]) -> None:
    cuda = env.get('CUDA_VISIBLE_DEVICES')
    if not cuda:
        return
    ascend = env.get('ASCEND_RT_VISIBLE_DEVICES')
    if ascend and ascend_mask_is_narrower(ascend, cuda):
        return
    env['ASCEND_RT_VISIBLE_DEVICES'] = cuda


sync_ascend_visible_devices(os.environ)
