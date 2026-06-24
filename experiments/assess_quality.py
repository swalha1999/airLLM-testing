"""FR-d — qualitative output-quality assessment per scenario (#20).

Pairs each scenario's captured output with a documented qualitative rubric
(coherence / relevance / fluency, 1-5) and records the assessment. Quality is a
human judgement; the scores + rationale are stated here so the result is
transparent and reproducible from the saved outputs.

Run with: uv run python experiments/assess_quality.py
"""

from airllm_bench.analysis import latest_by_scenario, load_results
from airllm_bench.shared.recorder import ResultRecorder

# Rubric scores (1-5) with rationale, applied to the captured outputs.
ASSESSMENT: dict[str, dict] = {
    "baseline_ollama_q4": {
        "precision": "~Q4 (GGUF)",
        "coherence": 5,
        "relevance": 5,
        "fluency": 5,
        "note": "Complete, accurate, fluent answer; no visible quantization degradation.",
    },
    "baseline_hf_direct_fp16": {
        "precision": "FP16",
        "coherence": 5,
        "relevance": 5,
        "fluency": 5,
        "note": "Complete, accurate, fluent (truncated at the 32-token cap).",
    },
    "airllm_7b_fp16": {
        "precision": "FP16",
        "coherence": 5,
        "relevance": 5,
        "fluency": 4,
        "note": "On-topic and correct, but only 8 tokens (cap) so partial; echoes the prompt.",
    },
}

RED_LINE = (
    "Not reached: only FP16 and ~Q4 were runnable (bitsandbytes Q8/Q4 are CUDA-gated, #15). "
    "Both produced coherent output, so no quality red line was crossed in the testable range."
)


def main() -> None:
    """Build, print, and record the per-scenario quality assessment."""
    records = load_results()
    rows = []
    for scenario, rubric in ASSESSMENT.items():
        record = latest_by_scenario(records, scenario) or {}
        rows.append(
            {"scenario": scenario, "output": (record.get("output") or "").strip(), **rubric}
        )

    for row in rows:
        print(
            f"{row['scenario']} ({row['precision']}): "
            f"coherence={row['coherence']} relevance={row['relevance']} fluency={row['fluency']}"
        )

    assessment = {"scenario": "quality_assessment", "rows": rows, "red_line": RED_LINE}
    print(f"\nrecorded: {ResultRecorder().save(assessment)}")


if __name__ == "__main__":
    main()
