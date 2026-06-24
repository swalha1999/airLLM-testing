"""FR-b baseline — the clean optimised run: Ollama qwen2.5:7b, measured (#16).

Streams a generation from the already-pulled Ollama GGUF model through the full
harness (SDK -> StreamMeasurer -> ResultRecorder) and records TTFT / TPOT /
throughput. This is the "well-behaved" contrast to the HF direct-load bottleneck.

Run with: uv run python experiments/baseline_ollama.py
"""

from airllm_bench.measure import StreamMeasurer
from airllm_bench.runners.ollama_runner import OllamaRunner
from airllm_bench.sdk import BenchmarkSDK
from airllm_bench.shared.recorder import ResultRecorder

MODEL = "qwen2.5:7b"
PROMPT = "Explain what a transformer model is in two sentences."


def main() -> None:
    """Run and record the Ollama baseline."""
    runner = OllamaRunner(MODEL, num_predict=128)
    sdk = BenchmarkSDK(StreamMeasurer(power_watts=45), ResultRecorder())
    result = sdk.run_scenario(
        "baseline_ollama_q4",
        lambda: runner.stream(PROMPT),
        metadata={"model": MODEL, "engine": "ollama-gguf-q4", "prompt": PROMPT},
    )
    print("metrics:", result.metrics)
    print("output:", result.output[:200])


if __name__ == "__main__":
    main()
