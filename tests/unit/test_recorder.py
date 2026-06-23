"""Tests for the result recorder (V3 S9) — writes to a tmp dir, fixed clock."""

import json
from datetime import UTC, datetime
from pathlib import Path

from airllm_bench.shared.recorder import SCHEMA_VERSION, ResultRecorder
from airllm_bench.shared.version import __version__


def _fixed_now() -> datetime:
    return datetime(2026, 6, 23, 12, 0, 0, tzinfo=UTC)


def test_save_writes_enriched_json(tmp_path: Path) -> None:
    """The saved file round-trips and carries schema/code versions + timestamp."""
    recorder = ResultRecorder(tmp_path, now_fn=_fixed_now)
    path = recorder.save({"scenario": "airllm_q4", "metrics": {"ttft_s": 1.2}})

    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["code_version"] == __version__
    assert data["timestamp"] == "2026-06-23T12:00:00+00:00"
    assert data["scenario"] == "airllm_q4"
    assert data["metrics"]["ttft_s"] == 1.2


def test_filename_is_scenario_prefixed_and_safe(tmp_path: Path) -> None:
    """Filename starts with the scenario and contains no colons (Windows-safe)."""
    path = ResultRecorder(tmp_path, now_fn=_fixed_now).save({"scenario": "baseline"})
    assert path.name.startswith("baseline_")
    assert ":" not in path.name
    assert path.name.endswith(".json")


def test_creates_results_dir_when_missing(tmp_path: Path) -> None:
    """A missing results directory is created on first save."""
    target = tmp_path / "nested" / "results"
    assert not target.exists()
    ResultRecorder(target, now_fn=_fixed_now).save({"scenario": "s"})
    assert target.is_dir()


def test_missing_scenario_defaults_filename(tmp_path: Path) -> None:
    """A record without a scenario still saves under a default name."""
    path = ResultRecorder(tmp_path, now_fn=_fixed_now).save({"metrics": {}})
    assert path.name.startswith("result_")
