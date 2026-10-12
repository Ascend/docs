"""vLLM serve flags for the evaluate PATH shim.

Train examples call python -m vllm via scripts/launch_vllm.py and never
hit this list. That launcher is not edited.
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
