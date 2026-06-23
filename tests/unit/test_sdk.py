"""Tests for the BenchmarkSDK entry point (V3 S9).

Uses in-memory fakes for the measurer and recorder so the SDK is exercised
without a real model — proving the orchestration, not the heavy backends.
"""

import pytest

from airllm_bench.sdk import BenchmarkSDK, ScenarioResult


class FakeMeasurer:
    """Returns canned metrics and echoes the generate() output."""

    def __init__(self) -> None:
        self.calls = 0

    def measure(self, generate):
        self.calls += 1
        output = generate()
        return {"metrics": {"throughput_tok_s": 1.5}, "output": output}


class FakeRecorder:
    """Captures saved records in memory."""

    def __init__(self) -> None:
        self.saved: list[dict] = []

    def save(self, record):
        self.saved.append(record)
        return len(self.saved)


def _sdk() -> tuple[BenchmarkSDK, FakeMeasurer, FakeRecorder]:
    measurer, recorder = FakeMeasurer(), FakeRecorder()
    return BenchmarkSDK(measurer, recorder), measurer, recorder


def test_run_scenario_measures_records_and_returns_result() -> None:
    """A scenario is measured once, saved once, and returned with its metrics."""
    sdk, measurer, recorder = _sdk()
    result = sdk.run_scenario("baseline", lambda: "hello", metadata={"model": "qwen"})
    assert isinstance(result, ScenarioResult)
    assert result.scenario == "baseline"
    assert result.output == "hello"
    assert result.metrics["throughput_tok_s"] == 1.5
    assert measurer.calls == 1
    assert recorder.saved[0]["scenario"] == "baseline"
    assert recorder.saved[0]["model"] == "qwen"


def test_run_scenario_defaults_metadata_to_empty() -> None:
    """Metadata is optional and defaults to an empty mapping."""
    sdk, _, recorder = _sdk()
    result = sdk.run_scenario("airllm_q4", lambda: "x")
    assert result.metadata == {}
    assert recorder.saved[0] == {"scenario": "airllm_q4", "metrics": {"throughput_tok_s": 1.5}}


def test_empty_scenario_name_rejected() -> None:
    """A blank scenario name is a defensive ValueError (V3 S13)."""
    sdk, _, _ = _sdk()
    with pytest.raises(ValueError, match="non-empty"):
        sdk.run_scenario("", lambda: "x")


def test_measure_and_save_result_delegate() -> None:
    """The thin measure() and save_result() helpers delegate to collaborators."""
    sdk, measurer, recorder = _sdk()
    assert sdk.measure(lambda: "out")["output"] == "out"
    assert measurer.calls == 1
    assert sdk.save_result({"scenario": "s", "metrics": {}}) == 1
    assert recorder.saved[-1]["scenario"] == "s"
