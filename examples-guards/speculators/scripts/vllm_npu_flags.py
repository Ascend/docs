"""vLLM serve flags shared by the evaluate PATH shim and the train launcher patch.

Keep this list in one place. shims/vllm and patch_launch_vllm.py both read it.
"""

VLLM_NPU_FLAGS = (
    '--no-enable-chunked-prefill',
    '--enforce-eager',
    '--api-server-count',
    '1',
    '--renderer-num-workers',
    '1',
    '--max-model-len',
    '2048',
)
