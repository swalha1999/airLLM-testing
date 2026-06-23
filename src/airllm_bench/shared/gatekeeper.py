"""Centralized API gatekeeper — rate limiting, overflow queue, retries (V3 S7).

Every external call (cost-analysis provider calls, any scripted HF downloads)
goes through :class:`ApiGatekeeper`. It enforces a per-minute rate limit before
each call; when the limit is hit, callers wait in a bounded FIFO queue
(backpressure when full) instead of being dropped, and transient failures are
retried. Limits come from ``config/rate_limits.json`` (V3 S3) — never hard-coded.
Thread-safe (V3 S14): all shared state is guarded by a single lock.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class QueueFullError(RuntimeError):
    """Raised when the overflow queue is at capacity (backpressure signal)."""


@dataclass(frozen=True)
class ServiceLimits:
    """Resolved rate-limit settings for one service."""

    requests_per_minute: int
    retry_after_seconds: float
    max_retries: int
    max_queue_depth: int

    @classmethod
    def from_config(cls, service: dict, max_queue_depth: int) -> ServiceLimits:
        """Build limits from a service entry of ``rate_limits.json``."""
        return cls(
            requests_per_minute=int(service["requests_per_minute"]),
            retry_after_seconds=float(service.get("retry_after_seconds", 1)),
            max_retries=int(service.get("max_retries", 0)),
            max_queue_depth=int(max_queue_depth),
        )


@dataclass(frozen=True)
class QueueStatus:
    """Snapshot of the gatekeeper's pending-request queue."""

    pending: int
    max_depth: int


def limits_for(rate_limits_config: dict, service: str = "default") -> ServiceLimits:
    """Extract a service's limits from a loaded ``rate_limits.json`` mapping."""
    section = rate_limits_config["rate_limits"]
    services = section["services"]
    chosen = services.get(service, services["default"])
    max_depth = section.get("queue", {}).get("max_depth", 100)
    return ServiceLimits.from_config(chosen, max_depth)


class ApiGatekeeper:
    """All external calls pass through here (V3 S7).

    Setup:  resolved :class:`ServiceLimits`; ``time_fn``/``sleep_fn`` are
            injectable so the rate-limit clock is controllable in tests.
    Input:  any callable + its args, via :meth:`execute`.
    Output: the callable's return value (after rate-limit wait + retries).
    """

    def __init__(
        self,
        limits: ServiceLimits,
        *,
        time_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._limits = limits
        self._time = time_fn
        self._sleep = sleep_fn
        self._calls: deque[float] = deque()
        self._pending = 0
        self._lock = threading.Lock()

    def get_queue_status(self) -> QueueStatus:
        """Return the current pending count and configured max depth."""
        with self._lock:
            return QueueStatus(self._pending, self._limits.max_queue_depth)

    def _prune(self, now: float) -> None:
        """Drop call timestamps older than the 60-second window."""
        while self._calls and self._calls[0] <= now - 60.0:
            self._calls.popleft()

    def _reserve_slot(self) -> None:
        """Block until under the rate limit; raise if the queue is full."""
        with self._lock:
            if self._pending >= self._limits.max_queue_depth:
                raise QueueFullError("gatekeeper queue is full")
            self._pending += 1
        try:
            while True:
                with self._lock:
                    now = self._time()
                    self._prune(now)
                    if len(self._calls) < self._limits.requests_per_minute:
                        self._calls.append(now)
                        return
                    wait = 60.0 - (now - self._calls[0])
                self._sleep(max(wait, 0.0))
        finally:
            with self._lock:
                self._pending -= 1

    def execute(self, api_call: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run an external call through the gatekeeper, with retries on failure."""
        self._reserve_slot()
        attempt = 0
        while True:
            try:
                logger.info("gatekeeper executing (attempt %d)", attempt + 1)
                return api_call(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - retry any transient failure
                if attempt >= self._limits.max_retries:
                    logger.error("gatekeeper call failed after %d attempts", attempt + 1)
                    raise
                attempt += 1
                logger.warning("gatekeeper retry %d after error: %s", attempt, exc)
                self._sleep(self._limits.retry_after_seconds)
