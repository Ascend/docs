"""Serve a persistent NPU model with the upstream dynamic batching pattern.

Source: https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html
"""

import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import ray


def test_http_requests_are_batched_for_npu_model_inference() -> None:
    import requests
    from ray import serve

    expected_id = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    assert expected_id and "," not in expected_id

    @serve.deployment(
        ray_actor_options={"resources": {"NPU": 1}}, max_ongoing_requests=8
    )
    class BatchedModel:
        def __init__(self):
            import torch
            import torch_npu  # noqa: F401

            ids = ray.get_runtime_context().get_accelerator_ids()["NPU"]
            assert len(ids) == 1
            self.npu_id = str(ids[0])
            assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == self.npu_id
            self.model = torch.nn.Linear(1, 1).eval().to("npu:0")
            with torch.no_grad():
                self.model.weight.fill_(2.0)
                self.model.bias.fill_(1.0)
            self.instance = uuid.uuid4().hex

        @serve.batch(max_batch_size=4, batch_wait_timeout_s=2.0)
        async def predict(self, items: list[dict]) -> list[dict]:
            import torch

            inputs = torch.tensor(
                [[item["x"]] for item in items], device="npu:0", dtype=torch.float32
            )
            assert (
                inputs.device.type == next(self.model.parameters()).device.type == "npu"
            )
            with torch.inference_mode():
                values = self.model(inputs).flatten().cpu().tolist()
            return [
                {
                    "id": item["id"],
                    "value": value,
                    "batch_size": len(items),
                    "npu_id": self.npu_id,
                    "instance": self.instance,
                }
                for item, value in zip(items, values)
            ]

        async def __call__(self, request):
            return await self.predict(await request.json())

    ray.init(include_dashboard=False, log_to_driver=False)
    started = False
    try:
        assert int(ray.cluster_resources().get("NPU", 0)) == 1
        serve.start(http_options={"host": "127.0.0.1", "port": 18081})
        started = True
        serve.run(BatchedModel.bind(), name="npu-batching", route_prefix="/predict")

        def request_prediction(index, barrier=None):
            if barrier is not None:
                barrier.wait(timeout=15)
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(
                    "http://127.0.0.1:18081/predict",
                    json={"id": index, "x": float(index)},
                    timeout=(5, 60),
                )
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["id"] == index, result
            return result

        # Initialize the device before measuring whether concurrent requests batch.
        warmup = request_prediction(-1)
        assert warmup["value"] == -1.0
        results = []
        for start in (0, 4):
            barrier = threading.Barrier(4)
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(request_prediction, index, barrier)
                    for index in range(start, start + 4)
                ]
                results.extend(future.result(timeout=75) for future in futures)
        assert sorted(item["id"] for item in results) == list(range(8))
        assert {item["npu_id"] for item in results} == {expected_id}
        assert {item["instance"] for item in results} == {warmup["instance"]}
        assert max(item["batch_size"] for item in results) > 1, results
        assert all(1 <= item["batch_size"] <= 4 for item in results)
        for item in results:
            assert abs(item["value"] - (2 * item["id"] + 1)) < 1e-5, item
    finally:
        try:
            if started:
                serve.shutdown()
        finally:
            ray.shutdown()
