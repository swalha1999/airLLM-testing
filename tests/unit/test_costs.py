"""Tests for the cost model (V3 S9) — known inputs -> known break-even."""

import pytest

from airllm_bench.services.costs import (
    api_cost_per_request,
    break_even_requests,
    electricity_per_request,
    onprem_fixed_monthly,
)


def test_api_cost_per_request() -> None:
    """500 in @ $2.5/M + 500 out @ $10/M = $0.00625."""
    assert api_cost_per_request(500, 500, 2.5, 10.0) == pytest.approx(0.00625)


def test_onprem_fixed_monthly() -> None:
    """$1200 over 36 months + $5 maintenance = $38.33/month."""
    assert onprem_fixed_monthly(1200.0, 36, 5.0) == pytest.approx(38.333, abs=1e-3)


def test_electricity_per_request() -> None:
    """3600 s at 45 W and $0.16/kWh = $0.0072."""
    assert electricity_per_request(3600, 45, 0.16) == pytest.approx(0.0072)


def test_break_even_requests() -> None:
    """Fixed / (api - var) gives the crossover volume."""
    be = break_even_requests(38.333, 0.00625, 0.00126)
    assert be == pytest.approx(38.333 / (0.00625 - 0.00126), rel=1e-3)


def test_break_even_none_when_onprem_not_cheaper() -> None:
    """If on-prem's per-request cost isn't below the API's, the lines never cross."""
    assert break_even_requests(38.0, 0.001, 0.001) is None
    assert break_even_requests(38.0, 0.001, 0.002) is None
