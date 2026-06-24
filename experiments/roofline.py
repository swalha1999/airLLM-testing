"""Roofline visualization — where the workload sits vs the hardware limits (#25).

A roofline plots attainable performance (GFLOP/s) against arithmetic intensity
(FLOP/byte): the roof is min(compute peak, bandwidth x intensity). Batch-1 LLM
*decode* has very low intensity (each weight is read once per token), so it lands
deep on the **bandwidth-limited slope**, far below the compute ceiling — the
visual proof that every scenario here is memory/IO-bound, not compute-bound (FR-f).

Hardware ceilings are illustrative approximations for this i7-1195G7 / DDR4-3200 /
NVMe machine (physical constants, stated here). Writes figures/roofline.png.

Run with: uv run python experiments/roofline.py
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - must follow the backend selection

PEAK_GFLOPS = 150.0  # CPU FP32 peak (approx)
BW_RAM_GBS = 50.0  # dual-channel DDR4-3200 (approx)
BW_SSD_GBS = 2.0  # NVMe sequential read (approx) — AirLLM streams layers from here
DECODE_INTENSITY = 1.0  # ~FLOP/byte for batch-1 decode (very low)
FIGURE = "figures/roofline.png"


def main() -> None:
    """Plot the roofline with the decode operating point marked."""
    intensity = np.logspace(-1, 2.5, 200)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        intensity,
        np.minimum(PEAK_GFLOPS, BW_RAM_GBS * intensity),
        label="RAM-bound roof (~50 GB/s)",
    )
    ax.plot(
        intensity,
        np.minimum(PEAK_GFLOPS, BW_SSD_GBS * intensity),
        label="SSD-bound roof (~2 GB/s, AirLLM)",
    )
    ax.axhline(
        PEAK_GFLOPS,
        color="#888",
        linestyle=":",
        label=f"compute ceiling (~{PEAK_GFLOPS:.0f} GFLOP/s)",
    )
    ax.axvline(
        DECODE_INTENSITY,
        color="#c44e52",
        linestyle="--",
        label="LLM decode intensity (~1 FLOP/byte)",
    )

    # Operating points where decode intensity meets each roof.
    ax.scatter([DECODE_INTENSITY], [BW_RAM_GBS * DECODE_INTENSITY], color="#4c72b0", zorder=5)
    ax.scatter([DECODE_INTENSITY], [BW_SSD_GBS * DECODE_INTENSITY], color="#dd8452", zorder=5)
    ax.annotate(
        "memory-bound\n(far below ceiling)",
        (DECODE_INTENSITY, BW_RAM_GBS),
        textcoords="offset points",
        xytext=(12, -4),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("attainable performance (GFLOP/s)")
    ax.set_title("Roofline — batch-1 decode is bandwidth-bound, not compute-bound")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=150)
    plt.close(fig)
    print(f"wrote {FIGURE}")


if __name__ == "__main__":
    main()
