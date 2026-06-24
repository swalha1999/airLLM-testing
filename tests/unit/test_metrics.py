"""Tests for metric consolidation (V3 S9)."""

from airllm_bench.analysis.metrics import metrics_row, metrics_table


def test_metrics_row_fills_missing_with_none() -> None:
    """Present fields are read; absent ones become None."""
    record = {"metrics": {"peak_ram_gb": 7.74, "throughput_tok_s": 0.01}}
    row = metrics_row(record)
    assert row["peak_ram_gb"] == 7.74
    assert row["throughput_tok_s"] == 0.01
    assert row["ttft_s"] is None  # not captured in this record


def test_metrics_table_one_row_per_present_scenario() -> None:
    """A row is built per scenario that has a record; absent ones are skipped."""
    records = [
        {"scenario": "a", "timestamp": "1", "metrics": {"peak_ram_gb": 1.0}},
        {"scenario": "a", "timestamp": "2", "metrics": {"peak_ram_gb": 2.0}},
        {"scenario": "b", "timestamp": "1", "metrics": {"throughput_tok_s": 0.5}},
    ]
    table = metrics_table(records, ["a", "b", "missing"])
    assert [r["scenario"] for r in table] == ["a", "b"]
    assert table[0]["peak_ram_gb"] == 2.0  # latest 'a' record
    assert table[1]["throughput_tok_s"] == 0.5
