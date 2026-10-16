"""Hold four fractional allocations and verify that a fifth task must queue.

Source: https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html
"""

import asyncio
import os
import time

import ray


def test_fractional_npu_allocation() -> None:
    expected_id = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    assert expected_id and "," not in expected_id

    @ray.remote(num_cpus=0, max_concurrency=16)
    class AllocationGate:
        def __init__(self):
            self.arrivals = {}
            self.full = asyncio.Event()
            self.fifth = asyncio.Event()
            self.release_event = asyncio.Event()

        async def enter(self, index: int, device_id: str):
            self.arrivals[index] = device_id
            if len(self.arrivals) >= 4:
                self.full.set()
            if index == 4:
                self.fifth.set()
            else:
                await asyncio.wait_for(self.release_event.wait(), timeout=90)

        async def wait_until_full(self):
            await asyncio.wait_for(self.full.wait(), timeout=60)
            return dict(self.arrivals)

        async def fifth_entered_within(self, timeout: float) -> bool:
            try:
                await asyncio.wait_for(self.fifth.wait(), timeout=timeout)
                return True
            except asyncio.TimeoutError:
                return False

        def release(self):
            self.release_event.set()

    # CPU availability must not be the reason the fifth task queues.
    @ray.remote(resources={"NPU": 0.25}, num_cpus=0)
    def fractional_task(index: int, gate) -> str:
        ids = ray.get_runtime_context().get_accelerator_ids()["NPU"]
        assert len(ids) == 1, ids
        visible = os.environ["ASCEND_RT_VISIBLE_DEVICES"]
        assert visible == str(ids[0])
        ray.get(gate.enter.remote(index, visible), timeout=100)
        return visible

    ray.init(include_dashboard=False, log_to_driver=False)
    try:
        assert int(ray.cluster_resources().get("NPU", 0)) == 1
        gate = AllocationGate.remote()
        held = [fractional_task.remote(index, gate) for index in range(4)]
        arrivals = ray.get(gate.wait_until_full.remote(), timeout=75)
        assert arrivals == {index: expected_id for index in range(4)}
        # Resource snapshots can lag behind the workers' startup notifications.
        deadline = time.monotonic() + 10
        while (
            ray.available_resources().get("NPU", 0) >= 0.25
            and time.monotonic() < deadline
        ):
            time.sleep(0.1)
        assert ray.available_resources().get("NPU", 0) < 0.25
        fifth = fractional_task.remote(4, gate)
        assert not ray.get(gate.fifth_entered_within.remote(5), timeout=10), (
            "NPU capacity was oversubscribed"
        )
        ray.get(gate.release.remote(), timeout=10)
        assigned = ray.get(held + [fifth], timeout=75)
        assert assigned == [expected_id] * 5
    finally:
        ray.shutdown()
