"""Two Ray Serve applications must run inference on separate NPUs."""

import os
import time

import ray


def test_two_serve_replicas_use_distinct_npus() -> None:
    import requests
    from ray import serve

    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    expected = sorted(visible.split(",")) if visible else []
    assert len(expected) == 2, expected

    @serve.deployment(ray_actor_options={"resources": {"NPU": 1}})
    class NPUService:
        def __call__(self, request) -> dict:
            import torch
            import torch_npu  # noqa: F401

            ids = ray.get_runtime_context().get_accelerator_ids()["NPU"]
            assert len(ids) == 1, ids
            physical_id = str(ids[0])
            assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == physical_id
            value = torch.ones(4, device="npu:0").sum().cpu().item()
            return {"npu_id": physical_id, "value": float(value)}

    ray.init(log_to_driver=False)
    serve_started = False
    try:
        assert int(ray.cluster_resources().get("NPU", 0)) == 2
        serve.start(http_options={"host": "127.0.0.1", "port": 18080})
        serve_started = True
        serve.run(NPUService.bind(), name="ray-npu-a", route_prefix="/npu-a")
        serve.run(NPUService.bind(), name="ray-npu-b", route_prefix="/npu-b")

        def fetch_inference(session, name: str) -> dict:
            deadline = time.monotonic() + 60
            last_error = "replica not ready"
            while time.monotonic() < deadline:
                try:
                    response = session.get(
                        f"http://127.0.0.1:18080/{name}", timeout=5
                    )
                    if response.status_code == 200:
                        return response.json()
                    last_error = f"HTTP {response.status_code}: {response.text}"
                except requests.RequestException as exc:
                    last_error = str(exc)
                time.sleep(1)
            raise AssertionError(f"Serve NPU inference failed: {last_error}")

        with requests.Session() as session:
            session.trust_env = False
            results = [
                fetch_inference(session, name)
                for name in ("npu-a", "npu-b")
            ]
        assert sorted(result["npu_id"] for result in results) == expected
        assert [result["value"] for result in results] == [4.0, 4.0]
    finally:
        if serve_started:
            serve.shutdown()
        ray.shutdown()
