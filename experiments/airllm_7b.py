"""FR-c — the AirLLM main run: Qwen2.5-7B, layer-sharded on CPU (#17).

Runs the *same* 7B that thrashed in the FR-b baseline, but through AirLLM, which
loads one transformer layer at a time from disk. The point is NOT speed — it is
**feasibility**: AirLLM keeps peak RAM low (manual paging of layers) so a model
that could not fit now runs, at the cost of disk-bound, layer-by-layer latency.
Records peak RAM, total runtime, and throughput.

Run with: uv run --extra ml python experiments/airllm_7b.py
"""

import time
import traceback

from airllm_bench.measure.resources import ResourceSampler
from airllm_bench.runners import AirLLMRunner
from airllm_bench.shared.recorder import ResultRecorder

MODEL = "Qwen/Qwen2.5-7B"
SHARDS = "C:/airllm_shards"
PROMPT = "Explain what a transformer model is in two sentences."
MAX_NEW_TOKENS = 8


def main() -> None:
    """Run the 7B through AirLLM and record memory + timing."""
    recorder = ResultRecorder()
    record: dict = {
        "scenario": "airllm_7b_fp16",
        "model": MODEL,
        "engine": "airllm-fp16-cpu",
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    sampler = ResourceSampler()
    sampler.start()
    start = time.perf_counter()
    try:
        output = AirLLMRunner(MODEL, SHARDS, max_seq_len=128).generate(
            PROMPT, max_new_tokens=MAX_NEW_TOKENS
        )
        record["status"] = "success"
        record["output"] = output
    except Exception as exc:  # noqa: BLE001 - capture the outcome, never crash
        record["status"] = "documented_failure"
        record["error"] = f"{type(exc).__name__}: {exc}"[:300]
        traceback.print_exc()
    runtime = time.perf_counter() - start
    peak_ram, peak_vram = sampler.stop()
    ok = record["status"] == "success"
    record["metrics"] = {
        "peak_ram_gb": round(peak_ram, 3),
        "peak_vram_gb": round(peak_vram, 3),
        "total_runtime_s": round(runtime, 3),
        "throughput_tok_s": round(MAX_NEW_TOKENS / runtime, 5) if ok and runtime else 0,
    }
    print(f"\nrecorded: {recorder.save(record)}")
    print(
        f"status: {record['status']} | peak_ram_gb: {round(peak_ram, 2)} | runtime_s: {round(runtime, 1)}"
    )


if __name__ == "__main__":
    main()
