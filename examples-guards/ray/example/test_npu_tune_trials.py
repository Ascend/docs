"""Tune two learning rates with real NPU training and select the best result.

Source: https://docs.ray.io/en/latest/tune/examples/tune-pytorch-cifar.html
"""

import asyncio
import math
import os
import tempfile

import ray


def test_tune_trials_reserve_npus() -> None:
    from ray import tune

    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    expected = set(visible.split(",")) if visible else set()
    assert len(expected) == 2, expected

    @ray.remote(num_cpus=0, max_concurrency=2)
    class TrialBarrier:
        def __init__(self):
            self.arrivals = set()
            self.both_ready = asyncio.Event()

        async def arrive(self, device_id: str) -> list[str]:
            self.arrivals.add(device_id)
            if len(self.arrivals) == 2:
                self.both_ready.set()
            await asyncio.wait_for(self.both_ready.wait(), timeout=60)
            return sorted(self.arrivals)

    def objective(config: dict, barrier) -> None:
        import torch
        import torch_npu  # noqa: F401

        ids = ray.get_runtime_context().get_accelerator_ids()["NPU"]
        assert len(ids) == 1, ids
        physical_id = str(ids[0])
        assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == physical_id
        assert set(ray.get(barrier.arrive.remote(physical_id), timeout=70)) == expected
        model = torch.nn.Linear(1, 1, bias=False).to("npu:0")
        torch.nn.init.zeros_(model.weight)
        inputs = torch.ones((4, 1), device="npu:0")
        targets = 2 * inputs
        assert next(model.parameters()).device.type == inputs.device.type == "npu"
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
        for step in range(4):
            optimizer.zero_grad()
            loss = (model(inputs) - targets).square().mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                measured_loss = float((model(inputs) - targets).square().mean().cpu())
            tune.report(
                {
                    "loss": measured_loss,
                    "weight": float(model.weight.detach().cpu().item()),
                    "step": step + 1,
                    "npu_id": physical_id,
                }
            )

    ray.init(include_dashboard=False, log_to_driver=False)
    try:
        assert int(ray.cluster_resources().get("NPU", 0)) == 2
        barrier = TrialBarrier.remote()
        with tempfile.TemporaryDirectory() as results_dir:
            results = tune.Tuner(
                tune.with_resources(
                    tune.with_parameters(objective, barrier=barrier),
                    resources={"cpu": 1, "NPU": 1},
                ),
                param_space={"lr": tune.grid_search([0.05, 0.2])},
                tune_config=tune.TuneConfig(
                    metric="loss", mode="min", max_concurrent_trials=2
                ),
                run_config=tune.RunConfig(storage_path=results_dir, verbose=0),
            ).fit()
            assert len(results) == 2 and not results.errors, results.errors
            assert {result.config["lr"] for result in results} == {0.05, 0.2}
            assert {result.metrics["npu_id"] for result in results} == expected
            for result in results:
                assert result.metrics["step"] == 4
                # Closed-form CPU reference for zero-initialized scalar regression.
                residual = 2 * (1 - 2 * result.config["lr"]) ** 4
                assert math.isclose(
                    result.metrics["loss"], residual**2, rel_tol=1e-3, abs_tol=1e-5
                )
                assert math.isclose(
                    result.metrics["weight"], 2 - residual, rel_tol=1e-3, abs_tol=1e-5
                )
            best = results.get_best_result(metric="loss", mode="min", scope="last")
            assert best.config["lr"] == 0.2, best
    finally:
        ray.shutdown()
