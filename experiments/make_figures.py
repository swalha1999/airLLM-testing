"""Generate the Phase 4 comparison figures from saved results (#22).

Builds bar charts for throughput, peak RAM, and total runtime straight from the
recorded metrics (no re-running), writing PNGs to ``figures/``.

Run with: uv run python experiments/make_figures.py
"""

from airllm_bench.analysis import latest_by_scenario, load_results
from airllm_bench.analysis.charts import bar_chart

FIGURES_DIR = "figures"


def _metric(record: dict | None, field: str) -> float:
    return (record or {}).get("metrics", {}).get(field) or 0.0


def main() -> None:
    """Write the three comparison figures to figures/."""
    records = load_results()
    ollama = latest_by_scenario(records, "baseline_ollama_q4")
    airllm = latest_by_scenario(records, "airllm_7b_fp16")
    hf = latest_by_scenario(records, "baseline_hf_direct_fp16")

    paths = [
        bar_chart(
            ["Ollama Q4", "AirLLM FP16"],
            [_metric(ollama, "throughput_tok_s"), _metric(airllm, "throughput_tok_s")],
            "Throughput (higher is better)",
            "tokens/sec (log)",
            f"{FIGURES_DIR}/throughput_comparison.png",
            log=True,
        ),
        bar_chart(
            ["HF FP16 direct", "AirLLM FP16"],
            [_metric(hf, "peak_ram_gb"), _metric(airllm, "peak_ram_gb")],
            "Peak process RAM — same FP16 7B (direct adds ~15 GB swap)",
            "GB",
            f"{FIGURES_DIR}/peak_ram_comparison.png",
        ),
        bar_chart(
            ["Ollama Q4", "AirLLM FP16", "HF FP16 direct"],
            [
                _metric(ollama, "total_runtime_s"),
                _metric(airllm, "total_runtime_s"),
                _metric(hf, "total_runtime_s"),
            ],
            "Total runtime (log scale; differing token counts)",
            "seconds (log)",
            f"{FIGURES_DIR}/runtime_comparison.png",
            log=True,
        ),
    ]
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
