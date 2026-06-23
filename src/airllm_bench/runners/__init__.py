"""Runner layer — model backends (AirLLM, etc.).

Imported only by experiment scripts, never by the package root, so the heavy
ml stack stays out of the lint/test CI.
"""

from .airllm_runner import AirLLMRunner, ensure_safetensors_index

__all__ = ["AirLLMRunner", "ensure_safetensors_index"]
