"""Adapt the upstream PyTorch Train guide to a two-NPU regression problem.

Source: https://docs.ray.io/en/latest/train/getting-started-pytorch.html
"""

import os
import tempfile

import ray
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchConfig, TorchTrainer


def test_ddp_updates_match_across_npus_and_cpu_reference() -> None:
    expected_ids = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
    assert len(expected_ids) == 2 and all(item.isdigit() for item in expected_ids)

    def train_step() -> None:
        import torch
        import torch.distributed as dist
        import torch_npu  # noqa: F401
        from ray import train
        from ray.train.torch import get_device, prepare_data_loader, prepare_model
        from torch.nn.parallel import DistributedDataParallel
        from torch.utils.data import DataLoader, TensorDataset

        ids = ray.get_runtime_context().get_accelerator_ids()["NPU"]
        assert len(ids) == 1
        assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == str(ids[0])
        device = get_device()
        assert device.type == "npu"
        torch.npu.set_device(device)
        assert dist.get_backend() == "hccl" and dist.get_world_size() == 2

        # Equal-sized, disjoint shards make the averaged DDP gradient equal to
        # the gradient of a CPU model trained on the complete dataset.
        x = torch.arange(1, 5, dtype=torch.float32).reshape(-1, 1) / 4
        y = 2 * x
        sample_ids = torch.arange(4, dtype=torch.float32)
        loader = prepare_data_loader(
            DataLoader(TensorDataset(x, y, sample_ids), batch_size=2, shuffle=False),
            auto_transfer=False,
        )
        model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(model.weight)
        model = prepare_model(
            model,
            parallel_strategy_kwargs={
                "device_ids": [device.index],
                "output_device": device.index,
            },
        )
        assert isinstance(model, DistributedDataParallel)
        assert next(model.parameters()).device.type == "npu"
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        reference = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(reference.weight)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        local_samples = None
        for epoch in range(3):
            loader.sampler.set_epoch(epoch)
            batches = 0
            for features, targets, row_ids in loader:
                assert features.device.type == targets.device.type == "npu"
                local_samples = row_ids
                optimizer.zero_grad()
                loss = (model(features) - targets).square().mean()
                loss.backward()
                optimizer.step()
                batches += 1
            assert batches == 1
            reference_optimizer.zero_grad()
            reference_loss = (reference(x) - y).square().mean()
            reference_loss.backward()
            reference_optimizer.step()

        assert local_samples is not None
        shards = [torch.empty_like(local_samples) for _ in range(2)]
        dist.all_gather(shards, local_samples)
        observed = [int(item) for shard in shards for item in shard.cpu().tolist()]
        assert sorted(observed) == list(range(4)), observed
        assert set(shards[0].cpu().tolist()).isdisjoint(shards[1].cpu().tolist())

        weights = model.module.weight.detach()
        all_weights = [torch.empty_like(weights) for _ in range(2)]
        dist.all_gather(all_weights, weights)
        for weight in all_weights:
            torch.testing.assert_close(
                weight.cpu(), reference.weight.detach(), rtol=1e-4, atol=1e-5
            )
        assert weights.cpu().item() > 0
        physical_id = torch.tensor([float(ids[0])], device=device)
        all_ids = [torch.empty_like(physical_id) for _ in range(2)]
        dist.all_gather(all_ids, physical_id)
        assert sorted(int(item.cpu().item()) for item in all_ids) == sorted(
            map(int, expected_ids)
        )

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            checkpoint = None
            if train.get_context().get_world_rank() == 0:
                torch.save(
                    {"weight": weights.cpu()}, os.path.join(checkpoint_dir, "model.pt")
                )
                checkpoint = train.Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {"weight": weights.cpu().item(), "world_size": 2, "samples": 4},
                checkpoint=checkpoint,
            )

    ray.init(include_dashboard=False, log_to_driver=False)
    try:
        assert int(ray.cluster_resources().get("NPU", 0)) == 2
        with tempfile.TemporaryDirectory() as storage:
            result = TorchTrainer(
                train_step,
                scaling_config=ScalingConfig(
                    num_workers=2, resources_per_worker={"CPU": 1, "NPU": 1}
                ),
                torch_config=TorchConfig(backend="hccl", timeout_s=90),
                run_config=RunConfig(storage_path=storage, name="npu-ddp"),
            ).fit()
            assert result.checkpoint is not None
            assert result.metrics is not None
            assert result.metrics["world_size"] == 2 and result.metrics["samples"] == 4
            assert result.metrics["weight"] > 0
    finally:
        ray.shutdown()
