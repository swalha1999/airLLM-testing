"""Tests for StreamMeasurer (V3 S9) — deterministic via a scripted clock."""

from airllm_bench.measure.measurer import StreamMeasurer


class FakeClock:
    """Returns pre-scripted timestamps on each call."""

    def __init__(self, times: list[float]) -> None:
        self._times = list(times)

    def __call__(self) -> float:
        return self._times.pop(0)


class FakeSampler:
    """Stand-in for ResourceSampler returning canned peaks."""

    def __init__(self, ram: float = 6.0, vram: float = 0.1) -> None:
        self.ram, self.vram = ram, vram
        self.started = self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> tuple[float, float]:
        self.stopped = True
        return self.ram, self.vram


def test_measure_computes_all_metrics() -> None:
    """Three tokens give TTFT, TPOT, throughput, memory, runtime, power, output."""
    clock = FakeClock([0.0, 1.0, 1.5, 2.0, 2.5])  # start, t1, t2, t3, end
    sampler = FakeSampler(ram=6.0, vram=0.1)
    measurer = StreamMeasurer(lambda: sampler, time_fn=clock, power_watts=45)

    result = measurer.measure(lambda: ["a", "b", "c"])
    m = result["metrics"]

    assert result["output"] == "abc"
    assert m["ttft_s"] == 1.0
    assert m["tpot_ms"] == 500.0  # mean of two 0.5s gaps
    assert m["throughput_tok_s"] == 1.2  # 3 tokens / 2.5s
    assert m["total_runtime_s"] == 2.5
    assert m["output_tokens"] == 3
    assert m["peak_ram_gb"] == 6.0
    assert m["peak_vram_gb"] == 0.1
    assert m["est_power_wh"] == round(2.5 / 3600 * 45, 6)
    assert sampler.started and sampler.stopped


def test_single_token_has_zero_tpot() -> None:
    """With one token there are no inter-token gaps, so TPOT is 0."""
    clock = FakeClock([0.0, 1.0, 2.0])  # start, t1, end
    measurer = StreamMeasurer(FakeSampler, time_fn=clock)
    m = measurer.measure(lambda: ["only"])["metrics"]
    assert m["ttft_s"] == 1.0
    assert m["tpot_ms"] == 0.0
    assert m["throughput_tok_s"] == 0.5


def test_zero_tokens_and_no_power() -> None:
    """No tokens → zeroed timing; no power_watts → no energy estimate."""
    clock = FakeClock([3.0, 3.0])  # start == end, runtime 0
    measurer = StreamMeasurer(FakeSampler, time_fn=clock)
    m = measurer.measure(lambda: [])["metrics"]
    assert m["ttft_s"] == 0.0
    assert m["throughput_tok_s"] == 0.0
    assert m["output_tokens"] == 0
    assert "est_power_wh" not in m
