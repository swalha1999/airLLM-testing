"""Peak RAM / VRAM sampling during a run (PLAN §5, V3 S14).

A background thread polls memory at a fixed interval and keeps the peak. RAM is
read with ``psutil`` (a core dependency); VRAM is read with ``pynvml`` lazily and
best-effort, returning 0.0 when no NVIDIA GPU / library is available — so this
module imports cleanly even where the heavy GPU stack is absent (e.g. CI).
"""

import threading
import time
from collections.abc import Callable

import psutil


def _default_ram_gb() -> float:
    """Resident set size of this process, in GB."""
    return psutil.Process().memory_info().rss / 1e9


def _default_vram_gb() -> float:  # pragma: no cover - hardware/driver dependent
    """Used VRAM of GPU 0 in GB, or 0.0 if unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        used = pynvml.nvmlDeviceGetMemoryInfo(handle).used
        pynvml.nvmlShutdown()
        return used / 1e9
    except Exception:
        return 0.0


class ResourceSampler:
    """Track peak RAM and VRAM across a run.

    Setup:  ``ram_fn``/``vram_fn`` (injected for tests), poll ``interval``.
    Output: ``stop()`` returns ``(peak_ram_gb, peak_vram_gb)``.
    """

    def __init__(
        self,
        ram_fn: Callable[[], float] = _default_ram_gb,
        vram_fn: Callable[[], float] = _default_vram_gb,
        interval: float = 0.05,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._ram_fn = ram_fn
        self._vram_fn = vram_fn
        self._interval = interval
        self._sleep = sleep_fn
        self._peak_ram = 0.0
        self._peak_vram = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> None:
        """Take one reading and update the running peaks."""
        self._peak_ram = max(self._peak_ram, self._ram_fn())
        self._peak_vram = max(self._peak_vram, self._vram_fn())

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._sleep(self._interval)

    def start(self) -> None:
        """Take an initial reading and begin background sampling."""
        self._sample_once()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[float, float]:
        """Stop sampling and return ``(peak_ram_gb, peak_vram_gb)``."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return self._peak_ram, self._peak_vram
