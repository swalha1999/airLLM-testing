"""Tests for the results loader (V3 S9)."""

import json
from pathlib import Path

from airllm_bench.analysis.loader import latest_by_scenario, load_results


def _write(dir_path: Path, name: str, payload: dict) -> None:
    (dir_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_results_reads_all_json(tmp_path: Path) -> None:
    """Every JSON in the directory is loaded."""
    _write(tmp_path, "a", {"scenario": "x", "timestamp": "2026-01-01"})
    _write(tmp_path, "b", {"scenario": "y", "timestamp": "2026-01-02"})
    records = load_results(tmp_path)
    assert len(records) == 2
    assert {r["scenario"] for r in records} == {"x", "y"}


def test_load_results_missing_dir_returns_empty(tmp_path: Path) -> None:
    """A non-existent results directory yields an empty list, not an error."""
    assert load_results(tmp_path / "nope") == []


def test_latest_by_scenario_picks_newest(tmp_path: Path) -> None:
    """The most recent record (by timestamp) for a scenario is returned."""
    records = [
        {"scenario": "run", "timestamp": "2026-06-01", "v": 1},
        {"scenario": "run", "timestamp": "2026-06-24", "v": 2},
        {"scenario": "other", "timestamp": "2026-06-30", "v": 3},
    ]
    assert latest_by_scenario(records, "run")["v"] == 2


def test_latest_by_scenario_absent_returns_none() -> None:
    """An unknown scenario returns None."""
    assert latest_by_scenario([{"scenario": "a", "timestamp": "x"}], "missing") is None
