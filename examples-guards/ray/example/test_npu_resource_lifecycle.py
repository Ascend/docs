"""A reserved NPU must queue work and become reusable after actor death."""

import os

import ray


def test_npu_reservation_and_release() -> None:
    expected_id = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    assert expected_id and "," not in expected_id

    ray.init(include_dashboard=False, log_to_driver=False)

    @ray.remote(resources={"NPU": 1})
    class Reservation:
        def ready(self) -> str:
            return str(ray.get_runtime_context().get_accelerator_ids()["NPU"][0])

    @ray.remote(resources={"NPU": 1})
    def queued_work() -> str:
        import torch
        import torch_npu  # noqa: F401

        ids = ray.get_runtime_context().get_accelerator_ids()["NPU"]
        assert len(ids) == 1
        assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == str(ids[0])
        assert torch.ones(1, device="npu:0").cpu().item() == 1.0
        return str(ids[0])

    actor = None
    try:
        assert int(ray.cluster_resources().get("NPU", 0)) == 1
        actor = Reservation.remote()
        assert ray.get(actor.ready.remote(), timeout=30) == expected_id
        pending = queued_work.remote()
        ready, waiting = ray.wait([pending], timeout=2)
        assert not ready and waiting == [pending], "NPU was double-allocated"
        ray.kill(actor)
        actor = None
        assert ray.get(pending, timeout=60) == expected_id
    finally:
        if actor is not None:
            ray.kill(actor)
        ray.shutdown()
