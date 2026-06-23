"""airllm_bench — benchmarking harness for running a massive LLM locally (EX05).

Public API is exposed here so external consumers import from the package root
rather than reaching into internal modules (V3 S5 / S12).
"""

from .shared.version import __version__

__all__ = ["__version__"]
