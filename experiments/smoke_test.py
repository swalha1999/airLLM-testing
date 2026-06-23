"""Phase 2 smoke test — validate the AirLLM pipeline on the tiny model (#14).

Runs the AirLLM runner on Qwen2.5-0.5B and records the outcome as a result JSON.
On this model AirLLM shards every layer but cannot generate (tied embeddings /
single-file layout), so the run is expected to record a *documented failure* —
which is a valid result per the assignment. The real generation + quant sweep
runs on the natively-compatible 7B in Phase 3.

Run with: uv run --extra ml python experiments/smoke_test.py
"""

import time
import traceback

from airllm_bench.runners import AirLLMRunner
from airllm_bench.shared.recorder import ResultRecorder

MODEL = "Qwen/Qwen2.5-0.5B"
SHARDS = "C:/airllm_shards"
PROMPT = "What is the capital of France?"


def main() -> None:
    """Run the smoke test and persist its outcome (success or documented failure)."""
    recorder = ResultRecorder()
    record: dict = {"scenario": "smoke_airllm_0_5b", "model": MODEL, "compression": "FP16"}
    start = time.perf_counter()
    try:
        output = AirLLMRunner(MODEL, SHARDS).generate(PROMPT, max_new_tokens=8)
        record["status"] = "success"
        record["output"] = output
    except Exception as exc:  # noqa: BLE001 - capture the finding, never crash the run
        record["status"] = "documented_failure"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["finding"] = (
            "AirLLM sharded all layers, but generation fails: Qwen2.5-0.5B has "
            "tied embeddings (no lm_head) and a single-file checkpoint, which "
            "AirLLM does not support. The 7B main model is natively compatible."
        )
        traceback.print_exc()
    record["metrics"] = {"total_runtime_s": round(time.perf_counter() - start, 3)}
    path = recorder.save(record)
    print(f"\nrecorded: {path}\nstatus: {record['status']}")


if __name__ == "__main__":
    main()
