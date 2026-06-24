"""Ollama runner — stream tokens from a local Ollama model via its HTTP API.

Used for the FR-b baseline "clean optimised run": Ollama serves an already-
quantized GGUF model with llama.cpp CPU kernels. Yields token strings so the
StreamMeasurer can time TTFT / TPOT. Uses ``urllib`` (stdlib) — no torch, no
extra deps, so it never pulls the ml stack into CI.

Caveat: Ollama runs in its **own** process, so our in-process ``psutil`` RAM
sampler does not capture the model's memory here — for this runner the timing
metrics (TTFT/TPOT/throughput) are the meaningful ones, not peak RAM.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterator

_OLLAMA_URL = "http://localhost:11434/api/generate"


class OllamaRunner:
    """Stream a generation from a local Ollama model.

    Setup:  ``model`` tag (e.g. ``qwen2.5:7b``), API ``url``, ``num_predict``.
    Input:  a prompt string.
    Output: an iterator of token strings.
    """

    def __init__(
        self,
        model: str,
        *,
        url: str = _OLLAMA_URL,
        num_predict: int = 128,
    ) -> None:
        self.model = model
        self.url = url
        self.num_predict = num_predict

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield response tokens from Ollama's streaming generate endpoint."""
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {"num_predict": self.num_predict},
            }
        ).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 - fixed localhost URL
            self.url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request) as response:  # noqa: S310 - localhost
            for raw in response:
                raw = raw.strip()
                if not raw:
                    continue
                chunk = json.loads(raw)
                token = chunk.get("response")
                if token:
                    yield token
                if chunk.get("done"):
                    break
