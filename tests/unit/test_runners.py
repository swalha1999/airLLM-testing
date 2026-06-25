"""Tests for the runner layer's pure helpers (V3 S9).

The runner modules are mostly thin wrappers around torch/airllm (exercised by the
integration experiments, not CI). But ``ensure_safetensors_index`` has pure guard
logic that runs before any heavy import, so its branches are unit-testable here.
"""

from pathlib import Path

from airllm_bench.runners.airllm_runner import ensure_safetensors_index


def test_returns_false_when_index_already_exists(tmp_path: Path) -> None:
    """If an index is already present, nothing is created."""
    (tmp_path / "model.safetensors").write_bytes(b"x")
    (tmp_path / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    assert ensure_safetensors_index(tmp_path) is False


def test_returns_false_when_no_single_file_weights(tmp_path: Path) -> None:
    """If there is no single-file ``model.safetensors``, there is nothing to index."""
    assert ensure_safetensors_index(tmp_path) is False
