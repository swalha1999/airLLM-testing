"""FR-b baseline — the BOTTLENECK run: direct HF FP16 load of the 7B (#16).

Attempts to load the full ~15 GB FP16 model into ~15.7 GB of RAM with plain
transformers. This is expected to OOM or swap unbearably — the memory-bound
bottleneck the assignment is about. Peak RAM is sampled in-process (the model
lives here, unlike Ollama), and the outcome (success or documented failure) is
recorded either way.

>>> Take the Task Manager / RAM screen-recording WHILE this run executes. <<<

Run with: uv run --extra ml python experiments/baseline_hf_direct.py
"""

import time
import traceback

from airllm_bench.measure.resources import ResourceSampler
from airllm_bench.runners.hf_runner import HFDirectRunner
from airllm_bench.shared.recorder import ResultRecorder

MODEL = "Qwen/Qwen2.5-7B"
PROMPT = "Explain what a transformer model is in two sentences."


def main() -> None:
    """Attempt the direct load and record the bottleneck outcome + peak RAM."""
    recorder = ResultRecorder()
    record: dict = {"scenario": "baseline_hf_direct_fp16", "model": MODEL, "engine": "hf-fp16"}
    sampler = ResourceSampler()
    sampler.start()
    start = time.perf_counter()
    try:
        tokens = list(HFDirectRunner(MODEL, dtype="float16", max_new_tokens=32).stream(PROMPT))
        record["status"] = "success"
        record["output"] = "".join(tokens)
    except Exception as exc:  # noqa: BLE001 - capture the bottleneck, never crash
        record["status"] = "documented_failure"
        record["error"] = f"{type(exc).__name__}: {exc}"[:300]
        record["finding"] = (
            "Direct HF FP16 load of the 7B (~15 GB) on a 15.7 GB-RAM machine exceeded "
            "available memory (memory-bound bottleneck) — the baseline cannot run the "
            "model the normal way. AirLLM's layer-sharding is what makes it runnable."
        )
        traceback.print_exc()
    peak_ram, peak_vram = sampler.stop()
    record["metrics"] = {
        "peak_ram_gb": round(peak_ram, 3),
        "peak_vram_gb": round(peak_vram, 3),
        "total_runtime_s": round(time.perf_counter() - start, 3),
    }
    print(f"\nrecorded: {recorder.save(record)}\nstatus: {record['status']}")


if __name__ == "__main__":
    main()
