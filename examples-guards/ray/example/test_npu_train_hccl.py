"""Two local Ray Train workers must complete an HCCL collective on NPUs."""

import os
import tempfile

import ray
from ray.train import ScalingConfig
from ray.train.torch import TorchConfig, TorchTrainer


def test_two_worker_hccl_all_reduce() -> None:
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    expected = sorted(visible.split(",")) if visible else []
    assert len(expected) == 2, expected

    def train_step() -> None:
        import torch
        import torch.distributed as dist
        import torch_npu  # noqa: F401
        from ray import train
        from ray.train.torch import get_device

        ids = ray.get_runtime_context().get_accelerator_ids()["NPU"]
        assert len(ids) == 1, ids
        assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == str(ids[0])
        device = get_device()
        assert device.type == "npu", device
        assert dist.is_initialized()
        assert dist.get_backend() == "hccl"
        assert dist.get_world_size() == 2

        rank = train.get_context().get_world_rank()
        assigned_id = torch.tensor([float(ids[0])], device=device)
        all_ids = [torch.zeros_like(assigned_id) for _ in range(2)]
        dist.all_gather(all_ids, assigned_id)
        assert sorted(int(item.cpu().item()) for item in all_ids) == sorted(
            int(item) for item in expected
        )
        value = torch.tensor([float(rank + 1)], device=device)
        dist.all_reduce(value)
        total = float(value.cpu().item())
        assert total == 3.0, (rank, total)
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            checkpoint = None
            if rank == 0:
                torch.save(
                    {"collective_sum": total, "world_size": 2},
                    os.path.join(checkpoint_dir, "result.pt"),
                )
                checkpoint = train.Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {"collective_sum": total, "world_size": 2},
                checkpoint=checkpoint,
            )

    ray.init(include_dashboard=False, log_to_driver=False)
    try:
        assert int(ray.cluster_resources().get("NPU", 0)) == 2
        trainer = TorchTrainer(
            train_loop_per_worker=train_step,
            scaling_config=ScalingConfig(
                num_workers=2,
                resources_per_worker={"CPU": 1, "NPU": 1},
            ),
            torch_config=TorchConfig(backend="hccl", timeout_s=90),
        )
        result = trainer.fit()
        assert result.metrics["collective_sum"] == 3.0, result.metrics
        assert result.metrics["world_size"] == 2
    finally:
        ray.shutdown()
