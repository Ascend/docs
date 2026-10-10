"""NPU resources survive task errors and actor process restarts.

Source: https://docs.ray.io/en/latest/ray-core/fault_tolerance/actors.html
"""

import os
import uuid

import pytest
import ray


def test_npu_is_reusable_after_worker_error() -> None:
    expected_id = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    assert expected_id and "," not in expected_id

    ray.init(include_dashboard=False, log_to_driver=False)

    @ray.remote(resources={"NPU": 1}, max_retries=0)
    def failing_task() -> None:
        import torch
        import torch_npu  # noqa: F401

        assert torch.ones(1, device="npu:0").cpu().item() == 1.0
        raise RuntimeError("intentional NPU task failure")

    @ray.remote(resources={"NPU": 1})
    def replacement_task() -> tuple[str, float]:
        import torch
        import torch_npu  # noqa: F401

        ids = ray.get_runtime_context().get_accelerator_ids()["NPU"]
        assert len(ids) == 1
        assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == str(ids[0])
        return str(ids[0]), float(torch.ones(2, device="npu:0").sum().cpu().item())

    try:
        with pytest.raises(ray.exceptions.RayTaskError, match="intentional NPU"):
            ray.get(failing_task.remote(), timeout=45)
        assert ray.get(replacement_task.remote(), timeout=60) == (expected_id, 2.0)
    finally:
        ray.shutdown()


def test_actor_process_restarts_and_reloads_npu_model() -> None:
    expected_id = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    assert expected_id and "," not in expected_id

    @ray.remote(resources={"NPU": 1}, num_cpus=1, max_restarts=1, max_task_retries=-1)
    class Predictor:
        def __init__(self):
            import torch
            import torch_npu  # noqa: F401

            ids = ray.get_runtime_context().get_accelerator_ids()["NPU"]
            assert len(ids) == 1
            self.npu_id = str(ids[0])
            assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == self.npu_id
            self.model = torch.nn.Linear(1, 1).eval().to("npu:0")
            with torch.no_grad():
                self.model.weight.fill_(3.0)
                self.model.bias.fill_(0.25)
            self.instance = uuid.uuid4().hex

        def predict(self, value: float) -> dict:
            import torch

            assert next(self.model.parameters()).device.type == "npu"
            with torch.inference_mode():
                result = (
                    self.model(torch.tensor([[value]], device="npu:0")).cpu().item()
                )
            return {
                "pid": os.getpid(),
                "instance": self.instance,
                "npu_id": self.npu_id,
                "value": result,
            }

        @ray.method(max_task_retries=0)
        def crash(self):
            # Do not replay this call on the restarted actor.
            os._exit(7)

    ray.init(include_dashboard=False, log_to_driver=False)
    actor = None
    try:
        assert int(ray.cluster_resources().get("NPU", 0)) == 1
        actor = Predictor.remote()
        before = ray.get(actor.predict.remote(2.0), timeout=60)
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(actor.crash.remote(), timeout=45)
        after = ray.get(actor.predict.remote(2.0), timeout=90)
        assert after["pid"] != before["pid"]
        assert after["instance"] != before["instance"]
        assert before["npu_id"] == after["npu_id"] == expected_id
        assert before["value"] == after["value"] == 6.25
    finally:
        if actor is not None:
            ray.kill(actor, no_restart=True)
        ray.shutdown()
