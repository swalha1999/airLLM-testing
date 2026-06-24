"""FR-c quantization comparison: FP16 (AirLLM) vs ~Q4 (Ollama GGUF) (#18).

Builds the comparison straight from saved result JSONs — no re-running. Q4/Q8 via
bitsandbytes is unavailable on this CPU machine (#15), so the runnable comparison
is FP16 (AirLLM, on-prem) vs ~Q4 (Ollama / llama.cpp GGUF). Writes a consolidated
comparison record to ``results/``.

Run with: uv run python experiments/compare_quantization.py
"""

from airllm_bench.analysis import latest_by_scenario, load_results
from airllm_bench.shared.recorder import ResultRecorder

# Model on-disk sizes (GB) — the direct memory effect of quantization.
DISK_GB = {"fp16": 15.0, "q4": 4.7}


def _metrics(record: dict | None) -> dict:
    return record.get("metrics", {}) if record else {}


def main() -> None:
    """Assemble and record the FP16-vs-Q4 comparison."""
    records = load_results()
    q4 = latest_by_scenario(records, "baseline_ollama_q4")
    fp16 = latest_by_scenario(records, "airllm_7b_fp16")
    bnb = latest_by_scenario(records, "bnb_check_q4_0_5b")

    q4_tps = _metrics(q4).get("throughput_tok_s") or 0
    fp16_tps = _metrics(fp16).get("throughput_tok_s") or 0
    speedup = round(q4_tps / fp16_tps, 1) if fp16_tps else None

    comparison = {
        "scenario": "quantization_comparison",
        "regimes": {
            "fp16_airllm": {"disk_gb": DISK_GB["fp16"], **_metrics(fp16)},
            "q4_ollama_gguf": {"disk_gb": DISK_GB["q4"], **_metrics(q4)},
        },
        "q4_vs_fp16_speedup_x": speedup,
        "disk_shrink_x": round(DISK_GB["fp16"] / DISK_GB["q4"], 1),
        "bitsandbytes_q4_q8": {
            "status": (bnb or {}).get("status", "not_run"),
            "finding": (bnb or {}).get("finding", ""),
        },
    }

    print("=== Quantization comparison (FP16 AirLLM vs ~Q4 Ollama GGUF) ===")
    print(f"FP16 (AirLLM):  {DISK_GB['fp16']} GB on disk, {fp16_tps} tok/s")
    print(f"~Q4 (Ollama):   {DISK_GB['q4']} GB on disk, {q4_tps} tok/s")
    print(f"Q4 is ~{speedup}x faster and ~{comparison['disk_shrink_x']}x smaller on disk")
    print(f"bitsandbytes Q4/Q8: {comparison['bitsandbytes_q4_q8']['status']}")
    print(f"\nrecorded: {ResultRecorder().save(comparison)}")


if __name__ == "__main__":
    main()
