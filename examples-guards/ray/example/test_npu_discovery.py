"""Ray Core must discover the real NPU count without a resource override."""

import os

import ray


def test_npu_discovery() -> None:
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    expected_ids = visible.split(",") if visible else []
    assert expected_ids and all(device.isdigit() for device in expected_ids)

    # Deliberately omit resources={"NPU": ...}: this checks auto-detection.
    ray.init(include_dashboard=False, log_to_driver=False)
    try:
        resources = ray.cluster_resources()
        assert int(resources.get("NPU", 0)) == len(expected_ids), resources

        @ray.remote(resources={"NPU": 1})
        def assigned_device() -> tuple[str, str]:
            ids = ray.get_runtime_context().get_accelerator_ids()["NPU"]
            assert len(ids) == 1, ids
            return str(ids[0]), os.environ["ASCEND_RT_VISIBLE_DEVICES"]

        assigned, worker_visible = ray.get(assigned_device.remote(), timeout=45)
        assert assigned in expected_ids
        assert worker_visible == assigned
    finally:
        ray.shutdown()
