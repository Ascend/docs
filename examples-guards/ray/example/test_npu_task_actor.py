"""A persistent actor and a task must use distinct physical NPUs."""

import os

import ray


def test_task_and_actor_have_isolated_npus() -> None:
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    expected = sorted(visible.split(",")) if visible else []
    assert len(expected) == 2, expected

    ray.init(include_dashboard=False, log_to_driver=False)

    def npu_result() -> tuple[str, float]:
        import torch
        import torch_npu  # noqa: F401

        ids = ray.get_runtime_context().get_accelerator_ids()["NPU"]
        assert len(ids) == 1, ids
        physical_id = str(ids[0])
        assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == physical_id
        assert torch.npu.is_available()
        value = torch.ones(4, device="npu:0").sum().cpu().item()
        return physical_id, float(value)

    @ray.remote(resources={"NPU": 1})
    class Holder:
        def run(self) -> tuple[str, float]:
            return npu_result()

    @ray.remote(resources={"NPU": 1})
    def task() -> tuple[str, float]:
        return npu_result()

    actor = None
    try:
        assert int(ray.cluster_resources().get("NPU", 0)) == 2
        actor = Holder.remote()
        actor_result = ray.get(actor.run.remote(), timeout=60)
        task_result = ray.get(task.remote(), timeout=60)
        assert sorted((actor_result[0], task_result[0])) == expected
        assert (actor_result[1], task_result[1]) == (4.0, 4.0)
    finally:
        if actor is not None:
            ray.kill(actor)
        ray.shutdown()
