"""Tests for chart generation (V3 S9) — assert a PNG is written."""

from pathlib import Path

from airllm_bench.analysis.charts import bar_chart, line_chart


def test_bar_chart_writes_png(tmp_path: Path) -> None:
    """bar_chart writes a non-empty PNG at the requested path."""
    out = tmp_path / "figs" / "throughput.png"
    result = bar_chart(["a", "b"], [0.79, 0.01], "Throughput", "tok/s", out, log=True)
    assert result == out
    assert out.is_file()
    assert out.stat().st_size > 0


def test_line_chart_writes_png_with_vline(tmp_path: Path) -> None:
    """line_chart writes a non-empty PNG, including the break-even marker path."""
    out = tmp_path / "figs" / "break_even.png"
    series = {"API": ([0, 1, 2], [0, 1, 2]), "On-Prem": ([0, 1, 2], [1, 1.5, 2.0])}
    result = line_chart(series, "Cost vs volume", "requests", "$", out, vline=1.0)
    assert result == out
    assert out.stat().st_size > 0
