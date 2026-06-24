"""HuggingFace direct-load runner — the baseline that loads the whole model into RAM.

This is the FR-b *bottleneck* case: loading a model larger than available RAM via
plain ``transformers`` either OOMs or swaps unbearably. Unlike the AirLLM and
Ollama runners, the model lives in **this** process, so the harness ``psutil``
sampler captures its true peak RAM. torch/transformers are imported lazily, so the
module stays out of the lint/test CI; it is coverage-omitted integration code.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator


class HFDirectRunner:
    """Load a full model into memory and stream a generation.

    Setup:  ``model_id``, ``device``, ``dtype`` (e.g. 'float16'), ``max_new_tokens``.
    Input:  a prompt string.
    Output: an iterator of generated token strings.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cpu",
        dtype: str = "float16",
        max_new_tokens: int = 64,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def load(self) -> HFDirectRunner:
        """Load the tokenizer and full model into memory (this is the OOM step)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=getattr(torch, self.dtype)
        ).to(self.device)
        return self

    def stream(self, prompt: str) -> Iterator[str]:
        """Generate token-by-token via a background thread + streamer for timing."""
        from transformers import TextIteratorStreamer

        if self._model is None:
            self.load()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = threading.Thread(
            target=self._model.generate,
            kwargs={**inputs, "max_new_tokens": self.max_new_tokens, "streamer": streamer},
        )
        thread.start()
        yield from streamer
        thread.join()
