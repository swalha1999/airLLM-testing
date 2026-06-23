"""Tests for the API gatekeeper (V3 S7/S9).

A controllable fake clock drives the rate-limit window so tests never sleep
for real and the wait/drain path is deterministic.
"""

from pathlib import Path

import pytest

from airllm_bench.shared.config import ConfigLoader
from airllm_bench.shared.gatekeeper import (
    ApiGatekeeper,
    QueueFullError,
    ServiceLimits,
    limits_for,
)

_REPO_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


class Clock:
    """Fake monotonic clock; sleeping advances virtual time."""

    def __init__(self) -> None:
        self.t = 0.0

    def time(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def _gk(rpm: int = 2, retries: int = 0, depth: int = 10) -> tuple[ApiGatekeeper, Clock]:
    clock = Clock()
    limits = ServiceLimits(rpm, retry_after_seconds=5.0, max_retries=retries, max_queue_depth=depth)
    return ApiGatekeeper(limits, time_fn=clock.time, sleep_fn=clock.sleep), clock


def test_under_limit_executes_immediately() -> None:
    """Calls below the rate limit run without any waiting."""
    gk, clock = _gk(rpm=2)
    assert gk.execute(lambda x: x + 1, 41) == 42
    assert clock.t == 0.0


def test_over_limit_waits_then_runs() -> None:
    """The call that exceeds the per-minute limit waits for the window to drain."""
    gk, clock = _gk(rpm=2)
    gk.execute(lambda: "a")
    gk.execute(lambda: "b")
    assert gk.execute(lambda: "c") == "c"  # third call must wait
    assert clock.t == pytest.approx(60.0)  # advanced by one full window


def test_retry_then_succeed() -> None:
    """A transient failure is retried up to max_retries."""
    gk, _ = _gk(retries=2)
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient")
        return "ok"

    assert gk.execute(flaky) == "ok"
    assert calls["n"] == 2


def test_retry_exhausted_raises() -> None:
    """When retries run out, the last error propagates."""
    gk, _ = _gk(retries=1)
    with pytest.raises(RuntimeError, match="boom"):
        gk.execute(lambda: (_ for _ in ()).throw(RuntimeError("boom")))


def test_queue_full_raises_backpressure() -> None:
    """A zero-depth queue signals backpressure immediately."""
    gk, _ = _gk(depth=0)
    with pytest.raises(QueueFullError):
        gk.execute(lambda: "x")


def test_queue_status_at_rest() -> None:
    """At rest the queue is empty and reports the configured max depth."""
    gk, _ = _gk(depth=7)
    status = gk.get_queue_status()
    assert status.pending == 0
    assert status.max_depth == 7


def test_limits_for_reads_real_config() -> None:
    """limits_for resolves a service from the shipped rate_limits.json."""
    data = (
        ConfigLoader(_REPO_CONFIG_DIR)
        .load("rate_limits", "1.00", version_key="rate_limits.version")
        .data
    )
    limits = limits_for(data, "huggingface_hub")
    assert limits.requests_per_minute == 60
    assert limits.max_queue_depth == 100
    # Unknown service falls back to default.
    assert limits_for(data, "missing").requests_per_minute == 30
