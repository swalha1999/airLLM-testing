"""FR-d — consolidate metrics across all scenarios into one table (#19).

Reads the saved result records and lines up every scenario's inference metrics
(peak RAM/VRAM, TTFT, TPOT, throughput, runtime, power) in a single table, then
records it. Built straight from ``results/`` so it is reproducible (NFR1).

Run with: uv run python experiments/consolidate_metrics.py
"""

from airllm_bench.analysis import METRIC_FIELDS, load_results, metrics_table
from airllm_bench.shared.recorder import ResultRecorder

SCENARIOS = ["baseline_ollama_q4", "baseline_hf_direct_fp16", "airllm_7b_fp16"]


def _fmt(value: object) -> str:
    return "N/A" if value is None else str(value)


def main() -> None:
    """Build, print, and record the consolidated metrics table."""
    table = metrics_table(load_results(), SCENARIOS)

    header = ["scenario", *METRIC_FIELDS]
    print(" | ".join(header))
    for row in table:
        print(" | ".join(_fmt(row.get(col)) for col in header))

    path = ResultRecorder().save({"scenario": "metrics_summary", "rows": table})
    print(f"\nrecorded: {path}")


if __name__ == "__main__":
    main()
