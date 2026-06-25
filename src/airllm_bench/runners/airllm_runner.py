"""AirLLM runner — load a model layer-sharded on disk and generate (PRD_airllm_sharding).

Wraps AirLLM's ``AutoModel`` behind a simple ``load()`` / ``generate()`` interface
and forces ``device="cpu"`` (this machine has no usable CUDA GPU). Includes a
helper that writes a SafeTensors index for single-file checkpoints, which AirLLM
requires before it will shard a model.

Caveat (see ``results/`` smoke finding): AirLLM targets large, multi-shard models
with a separate ``lm_head``. Small single-file models with *tied* embeddings
(e.g. Qwen2.5-0.5B) shard successfully but cannot generate. The 7B main model is
natively compatible. This module imports torch/airllm lazily inside methods so it
is never pulled in by the lint/test CI (which has no ml stack).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_safetensors_index(snapshot_dir: str | Path) -> bool:
    """Write ``model.safetensors.index.json`` for a single-file model.

    AirLLM only shards checkpoints that expose a weight-map index. Single-file
    models (just ``model.safetensors``) lack one, so we synthesise it. Returns
    True if an index was created, False if one already existed or is not needed.
    """
    snap = Path(snapshot_dir)
    index = snap / "model.safetensors.index.json"
    weights = snap / "model.safetensors"
    if index.exists() or not weights.exists():
        return False
    # Imported lazily, only on the create path, so the guard branches above are
    # importable/testable without the heavy ml stack (e.g. in CI).
    from safetensors import safe_open

    with safe_open(str(weights), framework="pt") as handle:
        keys = list(handle.keys())
    index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": weights.stat().st_size},
                "weight_map": dict.fromkeys(keys, "model.safetensors"),
            }
        ),
        encoding="utf-8",
    )
    return True


class AirLLMRunner:
    """Load an AirLLM model and generate text, layer-sharded on disk.

    Setup:  ``model_id``, ``shards_path`` (where per-layer files are written),
            ``device``, ``compression`` (None / '4bit' / '8bit'), ``max_seq_len``.
    Input:  a prompt string.
    Output: the decoded generated text.
    """

    def __init__(
        self,
        model_id: str,
        shards_path: str | Path,
        *,
        device: str = "cpu",
        compression: str | None = None,
        max_seq_len: int = 128,
    ) -> None:
        self.model_id = model_id
        self.shards_path = str(shards_path)
        self.device = device
        self.compression = compression
        self.max_seq_len = max_seq_len
        self._model: Any | None = None

    def load(self) -> AirLLMRunner:
        """Ensure the index exists, then load the AirLLM model (shards on first use)."""
        from airllm import AutoModel
        from huggingface_hub import snapshot_download

        # AirLLM's disk-space check reads the shards path before creating it,
        # so it must already exist.
        Path(self.shards_path).mkdir(parents=True, exist_ok=True)
        ensure_safetensors_index(snapshot_download(self.model_id))
        self._model = AutoModel.from_pretrained(
            self.model_id,
            device=self.device,
            layer_shards_saving_path=self.shards_path,
            compression=self.compression,
            max_seq_len=self.max_seq_len,
        )
        return self

    def generate(self, prompt: str, max_new_tokens: int = 20) -> str:
        """Generate a continuation of ``prompt`` (loads the model on first call)."""
        if self._model is None:
            self.load()
        model = self._model
        tokens = model.tokenizer([prompt], return_tensors="pt", return_attention_mask=False)
        output = model.generate(
            tokens["input_ids"],
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
        )
        return model.tokenizer.decode(output.sequences[0])
