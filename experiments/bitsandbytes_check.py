"""bitsandbytes quantization check (#15).

Runs AirLLM with compression='4bit' on Qwen2.5-0.5B to confirm whether
bitsandbytes quantization runs on this Windows / CPU-only machine. AirLLM
applies the compression per layer *during sharding*, which isolates the
bitsandbytes operation from the later (0.5B-specific) generation limitation.
The outcome — success or a documented failure — is recorded as a result.

Run with: uv run --extra ml python experiments/bitsandbytes_check.py
"""

import time
import traceback

from airllm_bench.runners import AirLLMRunner
from airllm_bench.shared.recorder import ResultRecorder

MODEL = "Qwen/Qwen2.5-0.5B"
SHARDS = "C:/airllm_shards_q4"


def main() -> None:
    """Attempt 4-bit compression and record whether bitsandbytes runs here."""
    recorder = ResultRecorder()
    record: dict = {"scenario": "bnb_check_q4_0_5b", "model": MODEL, "compression": "Q4(4bit)"}
    start = time.perf_counter()
    try:
        AirLLMRunner(MODEL, SHARDS, compression="4bit").load()
        record["status"] = "compression_ran"
        record["finding"] = "bitsandbytes 4-bit compression ran during layer sharding."
    except Exception as exc:  # noqa: BLE001 - capture the finding, never crash
        record["status"] = "documented_failure"
        record["error"] = f"{type(exc).__name__}: {exc}"
        if "CUDA" in str(exc):
            record["finding"] = (
                "bitsandbytes 4-bit quantization requires a CUDA GPU. On this "
                "CPU-only machine (torch 2.12.1+cpu, no usable CUDA) AirLLM's "
                "compression='4bit' fails during the layer-compression step with "
                "'Torch not compiled with CUDA enabled'. Q4/Q8 are unavailable on "
                "this hardware; the experiment runs FP16 via AirLLM CPU sharding."
            )
        else:
            record["finding"] = f"bitsandbytes 4-bit compression failed: {record['error']}"
        traceback.print_exc()
    record["metrics"] = {"total_runtime_s": round(time.perf_counter() - start, 3)}
    path = recorder.save(record)
    print(f"\nrecorded: {path}\nstatus: {record['status']}")


if __name__ == "__main__":
    main()
