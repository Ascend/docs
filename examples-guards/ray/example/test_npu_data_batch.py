"""Ray Data must schedule a batch transform on a real Ascend NPU."""

import os

import ray


def test_map_batches_uses_assigned_npu() -> None:
    expected_id = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    assert expected_id and "," not in expected_id

    def add_on_npu(batch: dict) -> dict:
        import torch
        import torch_npu  # noqa: F401

        ids = ray.get_runtime_context().get_accelerator_ids()["NPU"]
        assert len(ids) == 1, ids
        assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == str(ids[0])
        values = torch.as_tensor(batch["id"], device="npu:0")
        result = (values + 1).cpu().numpy()
        return {"id": result, "npu_id": [str(ids[0])] * len(result)}

    ray.init(include_dashboard=False, log_to_driver=False)
    try:
        assert int(ray.cluster_resources().get("NPU", 0)) == 1
        rows = (
            ray.data.range(4, override_num_blocks=1)
            .map_batches(add_on_npu, batch_size=4, resources={"NPU": 1})
            .take_all()
        )
        assert sorted(int(row["id"]) for row in rows) == [1, 2, 3, 4]
        assert {row["npu_id"] for row in rows} == {expected_id}
    finally:
        ray.shutdown()
