"""Consolidate scenario metrics into comparable rows (FR-d, #19).

Pulls the standard inference-metric set out of saved result records so every
scenario lines up in one table. Missing measurements stay ``None`` (not every
scenario captures every metric — e.g. the direct-load baseline recorded peak RAM
and runtime but not per-token timing), which the report shows honestly as N/A.
"""

from .loader import latest_by_scenario

# The standard FR-d metric set (PLAN §5/§6).
METRIC_FIELDS = (
    "peak_ram_gb",
    "peak_vram_gb",
    "ttft_s",
    "tpot_ms",
    "throughput_tok_s",
    "total_runtime_s",
    "est_power_wh",
)


def metrics_row(record: dict, fields: tuple[str, ...] = METRIC_FIELDS) -> dict:
    """Extract the metric fields from one record's ``metrics`` block."""
    metrics = record.get("metrics", {})
    return {field: metrics.get(field) for field in fields}


def metrics_table(
    records: list[dict],
    scenarios: list[str],
    fields: tuple[str, ...] = METRIC_FIELDS,
) -> list[dict]:
    """Build one row per scenario (its latest record), skipping absent scenarios."""
    table = []
    for scenario in scenarios:
        record = latest_by_scenario(records, scenario)
        if record is not None:
            table.append({"scenario": scenario, **metrics_row(record, fields)})
    return table
