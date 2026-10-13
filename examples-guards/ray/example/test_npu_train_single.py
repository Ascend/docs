"""Ray Train must perform a real optimizer step on an assigned Ascend NPU."""

import math
import os
import tempfile

import ray
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer


def test_single_worker_npu_optimizer_step() -> None:
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    assert visible and "," not in visible

    def train_step() -> None:
        import torch
        import torch_npu  # noqa: F401
        from ray import train
        from ray.train.torch import get_device

        ids = ray.get_runtime_context().get_accelerator_ids()["NPU"]
        assert len(ids) == 1, ids
        assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == str(ids[0])
        device = get_device()
        assert device.type == "npu", device

        model = torch.nn.Linear(4, 1).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        inputs = torch.ones((4, 4), device=device)
        targets = torch.zeros((4, 1), device=device)
        before = model.weight.detach().clone()
        loss = (model(inputs) - targets).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert not torch.equal(before, model.weight.detach())
        value = float(loss.detach().cpu().item())
        assert math.isfinite(value)
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            torch.save(
                {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                },
                os.path.join(checkpoint_dir, "model.pt"),
            )
            train.report(
                {"loss": value, "device": device.type},
                checkpoint=train.Checkpoint.from_directory(checkpoint_dir),
            )

    ray.init(include_dashboard=False, log_to_driver=False)
    try:
        assert int(ray.cluster_resources().get("NPU", 0)) == 1
        trainer = TorchTrainer(
            train_loop_per_worker=train_step,
            scaling_config=ScalingConfig(
                num_workers=1,
                resources_per_worker={"CPU": 1, "NPU": 1},
            ),
        )
        result = trainer.fit()
        assert result.metrics["device"] == "npu", result.metrics
        assert math.isfinite(float(result.metrics["loss"]))
    finally:
        ray.shutdown()
