"""Resume an NPU training run using the upstream checkpoint handoff pattern.

Source: https://docs.ray.io/en/latest/train/user-guides/checkpoints.html
"""

import os
import tempfile

import ray
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer


def test_checkpoint_restores_model_optimizer_and_training_step() -> None:
    import torch

    expected_id = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    assert expected_id and "," not in expected_id

    def train_step(config: dict) -> None:
        import copy

        import torch
        import torch_npu  # noqa: F401
        from ray import train
        from ray.train.torch import get_device

        ids = ray.get_runtime_context().get_accelerator_ids()["NPU"]
        assert [str(item) for item in ids] == [expected_id]
        assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == expected_id
        device = get_device()
        assert device.type == "npu"
        model = torch.nn.Linear(1, 1, bias=False).to(device)
        torch.nn.init.zeros_(model.weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        start_step = 0

        # Train V2 accepts an explicitly supplied checkpoint via train_loop_config;
        # get_checkpoint() also supports recovery of the current training run.
        checkpoint = train.get_checkpoint() or config.get("checkpoint")
        if checkpoint is not None:
            with checkpoint.as_directory() as checkpoint_dir:
                state = torch.load(
                    os.path.join(checkpoint_dir, "state.pt"),
                    map_location="cpu",
                    weights_only=True,
                )
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_step = state["step"]
            torch.testing.assert_close(
                model.weight.detach().cpu(), state["model"]["weight"]
            )
            momentum = optimizer.state[model.weight]["momentum_buffer"]
            assert momentum.device.type == "npu"
            torch.testing.assert_close(
                momentum.cpu(), state["optimizer"]["state"][0]["momentum_buffer"]
            )
            assert optimizer.param_groups[0]["lr"] == 0.1
        assert start_step == config["expected_start"]

        inputs = torch.ones((4, 1), device=device)
        targets = 2 * inputs
        for _ in range(start_step, config["steps"]):
            optimizer.zero_grad()
            loss = (model(inputs) - targets).square().mean()
            loss.backward()
            optimizer.step()
        final_loss = float((model(inputs) - targets).square().mean().detach().cpu())

        # Store CPU tensors so the checkpoint is portable across worker processes.
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        for state in optimizer_state["state"].values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.detach().cpu()
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            torch.save(
                {
                    "model": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "optimizer": optimizer_state,
                    "step": config["steps"],
                },
                os.path.join(checkpoint_dir, "state.pt"),
            )
            train.report(
                {
                    "loss": final_loss,
                    "start_step": start_step,
                    "step": config["steps"],
                    "device": device.type,
                },
                checkpoint=train.Checkpoint.from_directory(checkpoint_dir),
            )

    def read_state(checkpoint):
        assert checkpoint is not None
        with checkpoint.as_directory() as checkpoint_dir:
            return torch.load(
                os.path.join(checkpoint_dir, "state.pt"),
                map_location="cpu",
                weights_only=True,
            )

    ray.init(include_dashboard=False, log_to_driver=False)
    try:
        assert int(ray.cluster_resources().get("NPU", 0)) == 1
        with tempfile.TemporaryDirectory() as storage:

            def run(name, steps, expected_start=0, checkpoint=None):
                result = TorchTrainer(
                    train_step,
                    train_loop_config={
                        "steps": steps,
                        "expected_start": expected_start,
                        "checkpoint": checkpoint,
                    },
                    scaling_config=ScalingConfig(
                        num_workers=1, resources_per_worker={"CPU": 1, "NPU": 1}
                    ),
                    run_config=RunConfig(storage_path=storage, name=name),
                ).fit()
                assert result.metrics is not None and result.metrics["device"] == "npu"
                assert result.metrics["start_step"] == expected_start
                assert result.metrics["step"] == steps
                return result

            baseline = run("continuous", 4)
            partial = run("partial", 2)
            resumed = run("resumed", 4, expected_start=2, checkpoint=partial.checkpoint)
            baseline_state, partial_state, resumed_state = (
                read_state(result.checkpoint) for result in (baseline, partial, resumed)
            )
            assert baseline_state["step"] == resumed_state["step"] == 4
            assert partial_state["step"] == 2
            assert not torch.equal(
                partial_state["model"]["weight"], resumed_state["model"]["weight"]
            )
            torch.testing.assert_close(
                resumed_state["model"], baseline_state["model"], rtol=1e-4, atol=1e-5
            )
            torch.testing.assert_close(
                resumed_state["optimizer"]["state"],
                baseline_state["optimizer"]["state"],
                rtol=1e-4,
                atol=1e-5,
            )
            assert (
                resumed_state["optimizer"]["param_groups"]
                == baseline_state["optimizer"]["param_groups"]
            )
            assert abs(resumed.metrics["loss"] - baseline.metrics["loss"]) < 1e-4
    finally:
        ray.shutdown()
