"""Benchmark SDK — the single entry point for all experiment logic (V3 S5).

Experiment scripts and notebooks talk only to ``BenchmarkSDK``; they never reach
into the runner, measurement, or recorder modules directly. The SDK orchestrates
the three steps of every scenario — *run a generation, measure it, persist the
result* — using collaborators injected at construction (V3 S13) so it stays
decoupled and unit-testable without a real model.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


class Measurer(Protocol):
    """Times a generation and returns ``{"metrics": {...}, "output": ...}``."""

    def measure(self, generate: Callable[[], Any]) -> dict: ...


class Recorder(Protocol):
    """Persists a finished result record (e.g. as JSON under ``results/``)."""

    def save(self, record: dict) -> Any: ...


@dataclass(frozen=True)
class ScenarioResult:
    """Outcome of one measured scenario."""

    scenario: str
    metrics: dict
    output: Any = None
    metadata: dict = field(default_factory=dict)


class BenchmarkSDK:
    """Single entry point: run a scenario, measure it, and save the result.

    Setup:  a ``Measurer`` (produces metrics) and a ``Recorder`` (persists
            results), both injected so the SDK is decoupled from how either works.
    Input:  a scenario name, a zero-arg ``generate`` callable, optional metadata.
    Output: a :class:`ScenarioResult`.
    """

    def __init__(self, measurer: Measurer, recorder: Recorder) -> None:
        self._measurer = measurer
        self._recorder = recorder

    def measure(self, generate: Callable[[], Any]) -> dict:
        """Measure a single generation, delegating to the injected measurer."""
        return self._measurer.measure(generate)

    def save_result(self, record: dict) -> Any:
        """Persist a result record, delegating to the injected recorder."""
        return self._recorder.save(record)

    def run_scenario(
        self,
        scenario: str,
        generate: Callable[[], Any],
        metadata: dict | None = None,
    ) -> ScenarioResult:
        """Run, measure, and record one scenario end to end.

        ``generate`` is the zero-arg callable the measurer invokes (and times) to
        produce the model output. Its measured metrics and the output are bundled
        into a :class:`ScenarioResult`, and a flat record is handed to the recorder.
        """
        if not scenario:
            raise ValueError("scenario name must be a non-empty string")
        measured = self._measurer.measure(generate)
        metrics = measured.get("metrics", {})
        output = measured.get("output")
        meta = metadata or {}
        self.save_result({"scenario": scenario, "metrics": metrics, "output": output, **meta})
        return ScenarioResult(scenario=scenario, metrics=metrics, output=output, metadata=meta)
