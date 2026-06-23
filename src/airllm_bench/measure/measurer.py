"""Stream measurer — TTFT, ITL/TPOT, throughput, runtime, peak memory (PLAN §5).

Consumes a token stream (the ``generate`` callable yields tokens) and times it:
the first token gives TTFT (a Prefill proxy), the gaps between later tokens give
TPOT (a Decode proxy), and total tokens over total time give throughput. Peak
RAM/VRAM are sampled in the background via :class:`ResourceSampler`. Pure-Python
and model-agnostic, so it is unit-tested without any real model.
"""

import time
from collections.abc import Callable, Iterable
from typing import Any

from .resources import ResourceSampler


class StreamMeasurer:
    """Measure one streamed generation and emit a metrics record.

    Setup:  a ``sampler_factory`` (defaults to :class:`ResourceSampler`), an
            injectable ``time_fn``, and optional ``power_watts`` for an energy
            estimate.
    Input:  ``generate`` — a zero-arg callable returning an iterable of tokens.
    Output: ``{"metrics": {...}, "output": str}``.
    """

    def __init__(
        self,
        sampler_factory: Callable[[], Any] = ResourceSampler,
        time_fn: Callable[[], float] = time.perf_counter,
        power_watts: float | None = None,
    ) -> None:
        self._sampler_factory = sampler_factory
        self._time = time_fn
        self._power_watts = power_watts

    def measure(self, generate: Callable[[], Iterable[Any]]) -> dict:
        """Run and time ``generate``'s token stream, sampling memory throughout."""
        sampler = self._sampler_factory()
        sampler.start()
        start = self._time()
        first: float | None = None
        last = start
        gaps: list[float] = []
        tokens: list[Any] = []
        for token in generate():
            now = self._time()
            if first is None:
                first = now
            else:
                gaps.append(now - last)
            last = now
            tokens.append(token)
        end = self._time()
        peak_ram, peak_vram = sampler.stop()
        metrics = self._metrics(start, first, end, gaps, len(tokens), peak_ram, peak_vram)
        return {"metrics": metrics, "output": "".join(str(t) for t in tokens)}

    def _metrics(
        self,
        start: float,
        first: float | None,
        end: float,
        gaps: list[float],
        n_tokens: int,
        peak_ram: float,
        peak_vram: float,
    ) -> dict:
        """Turn raw timestamps into the PLAN §6 metrics record."""
        runtime = end - start
        ttft = (first - start) if first is not None else 0.0
        tpot_ms = (sum(gaps) / len(gaps) * 1000.0) if gaps else 0.0
        throughput = (n_tokens / runtime) if runtime > 0 else 0.0
        metrics = {
            "ttft_s": round(ttft, 6),
            "tpot_ms": round(tpot_ms, 6),
            "throughput_tok_s": round(throughput, 6),
            "peak_ram_gb": round(peak_ram, 6),
            "peak_vram_gb": round(peak_vram, 6),
            "total_runtime_s": round(runtime, 6),
            "output_tokens": n_tokens,
        }
        if self._power_watts is not None:
            metrics["est_power_wh"] = round(runtime / 3600.0 * self._power_watts, 6)
        return metrics
