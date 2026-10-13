"""Use a callable-class model as in the upstream offline batch inference guide.

Source: https://docs.ray.io/en/latest/data/batch_inference.html
"""

import os
import uuid

import numpy as np
import ray


def test_actor_pool_reuses_npu_model_across_batches() -> None:
    expected_id = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    assert expected_id and "," not in expected_id

    class Predictor:
        def __init__(self):
            import torch
            import torch_npu  # noqa: F401

            ids = ray.get_runtime_context().get_accelerator_ids()["NPU"]
            assert len(ids) == 1
            self.npu_id = str(ids[0])
            assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == self.npu_id
            self.model = torch.nn.Linear(2, 1).eval().to("npu:0")
            with torch.no_grad():
                self.model.weight.copy_(torch.tensor([[2.0, -1.0]], device="npu:0"))
                self.model.bias.fill_(0.5)
            self.instance = uuid.uuid4().hex
            self.calls = 0

        def __call__(self, batch: dict) -> dict:
            import torch

            self.calls += 1
            features = np.column_stack((batch["x0"], batch["x1"])).astype(np.float32)
            inputs = torch.as_tensor(features, device="npu:0")
            assert (
                inputs.device.type == next(self.model.parameters()).device.type == "npu"
            )
            with torch.inference_mode():
                predictions = self.model(inputs).flatten().cpu().numpy()
            size = len(predictions)
            return {
                "id": batch["id"],
                "prediction": predictions,
                "npu_id": np.full(size, self.npu_id),
                "instance": np.full(size, self.instance),
                "call": np.full(size, self.calls),
            }

    ray.init(include_dashboard=False, log_to_driver=False)
    try:
        assert int(ray.cluster_resources().get("NPU", 0)) == 1
        rows = (
            ray.data.from_items(
                [
                    {"id": index, "x0": float(index), "x1": float(index + 1)}
                    for index in range(8)
                ],
                override_num_blocks=4,
            )
            .map_batches(
                Predictor,
                batch_size=2,
                compute=ray.data.ActorPoolStrategy(size=1),
                resources={"NPU": 1},
            )
            .take_all()
        )
        assert len(rows) == 8 and sorted(int(row["id"]) for row in rows) == list(
            range(8)
        )
        assert {row["npu_id"] for row in rows} == {expected_id}
        assert len({row["instance"] for row in rows}) == 1
        assert len({int(row["call"]) for row in rows}) >= 2
        ordered = sorted(rows, key=lambda row: row["id"])
        # CPU reference for the fixed linear model: 2*x0 - x1 + 0.5.
        np.testing.assert_allclose(
            [row["prediction"] for row in ordered],
            np.arange(8) - 0.5,
            rtol=1e-4,
            atol=1e-5,
        )
    finally:
        ray.shutdown()
