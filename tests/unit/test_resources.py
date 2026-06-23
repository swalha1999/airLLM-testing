"""Tests for ResourceSampler (V3 S9/S14) — fakes avoid real hardware reads."""

from airllm_bench.measure.resources import ResourceSampler, _default_ram_gb


def test_sample_once_keeps_the_peak() -> None:
    """Repeated samples retain the maximum RAM/VRAM seen."""
    rams = iter([1.0, 3.0, 2.0])
    sampler = ResourceSampler(
        ram_fn=lambda: next(rams), vram_fn=lambda: 0.5, sleep_fn=lambda _: None
    )
    sampler._sample_once()
    sampler._sample_once()
    sampler._sample_once()
    peak_ram, peak_vram = sampler.stop()  # stop without start: returns peaks
    assert peak_ram == 3.0
    assert peak_vram == 0.5


def test_loop_samples_until_stopped() -> None:
    """The poll loop keeps sampling until the stop event is set."""
    calls = {"n": 0}
    sampler = ResourceSampler(vram_fn=lambda: 0.0, sleep_fn=lambda _: None)

    def ram() -> float:
        calls["n"] += 1
        if calls["n"] >= 3:
            sampler._stop.set()
        return float(calls["n"])

    sampler._ram_fn = ram
    sampler._loop()
    assert sampler._peak_ram == 3.0


def test_start_then_stop_returns_peaks() -> None:
    """A real start/stop thread cycle returns the sampled peaks."""
    sampler = ResourceSampler(ram_fn=lambda: 2.0, vram_fn=lambda: 0.5, interval=0.001)
    sampler.start()
    ram, vram = sampler.stop()
    assert ram == 2.0
    assert vram == 0.5


def test_default_ram_is_positive() -> None:
    """The real psutil-backed RAM reading is a positive number of GB."""
    assert _default_ram_gb() > 0
