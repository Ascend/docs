# Ray NPU application coverage

These are project-owned conformance tests inspired by the upstream documentation.
They run against the installed Ray version selected by the workflow. Small linear
models and synthetic data keep the results deterministic and avoid model downloads.
Model tests assert NPU placement explicitly; CPU references only calculate expected
answers. The suite requires at most two NPUs on one host.

| Test file | Upstream pattern | What must be true |
| --- | --- | --- |
| `test_npu_train_ddp.py` | [Distributed PyTorch training](https://docs.ray.io/en/latest/train/getting-started-pytorch.html) | `prepare_model` wraps a real NPU model in DDP; `prepare_data_loader` produces disjoint complete shards on NPUs; three optimizer steps on two different NPUs agree with a full-dataset CPU reference. |
| `test_npu_train_resume.py` | [Saving and loading checkpoints](https://docs.ray.io/en/latest/train/user-guides/checkpoints.html) | Three separate Trainer runs compare continuous training with a two-step checkpoint followed by two resumed steps. Model weights, SGD momentum, optimizer settings, step count, and final loss must match. |
| `test_npu_data_actor.py` | [Offline batch inference](https://docs.ray.io/en/latest/data/batch_inference.html) | A fixed one-actor pool loads a model once and processes multiple batches on its assigned NPU. All input IDs occur exactly once, actor identity persists, call count advances, and predictions match the CPU formula. |
| `test_npu_serve_batching.py` | [Dynamic request batching](https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html) | Concurrent HTTP requests form at least one multi-request batch. A persistent NPU model produces correct per-request responses across two waves; responses preserve IDs and the same model instance. |
| `test_npu_worker_recovery.py` | [Actor fault tolerance](https://docs.ray.io/en/latest/ray-core/fault_tolerance/actors.html) | In addition to task-error resource reuse, an actor process exits abruptly, is restarted by Ray, reloads its NPU model, and reproduces the result on the same assigned card with a new PID and instance ID. The crashing method is never retried. |
| `test_npu_fractional.py` | [Fractional accelerators](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html#fractional-accelerators) | Four tasks hold 0.25 NPU each concurrently; a fifth cannot enter until resources are released. Zero CPU requests prevent CPU contention from creating a false positive. This checks scheduler capacity, not device memory isolation. |
| `test_npu_tune_trials.py` | [Tune with PyTorch](https://docs.ray.io/en/latest/tune/examples/tune-pytorch-cifar.html) | Two grid-search trials reserve different NPUs concurrently, train a model with different learning rates, report metrics, and select the analytically expected best configuration. |

The existing discovery, accelerator-manager, single-worker Train, HCCL collective,
task/actor isolation, resource lifecycle, function-based Data, and two-application
Serve tests remain separate checks. In particular, the new DDP case complements
the collective test, and the actor-based Data case retains the function-based path.

Checkpoint data lives in temporary run storage until all comparisons finish.
Distributed workers all report; only rank 0 supplies the checkpoint when needed.
This keeps final metrics available under Ray Train V2. Restore uses an explicitly
supplied checkpoint through `train_loop_config`, not the deprecated Trainer
`resume_from_checkpoint` argument.

Local collection and CPU numerical checks do not establish NPU compatibility.
The application tests must pass on the declared Ascend runners. RLlib, multi-host
training, performance benchmarks, and large-model framework integrations are not
covered by this matrix.
