"""Generate comparison figures from saved metrics (#22).

Uses a headless matplotlib backend (Agg) so charts render in CI and without a
display. Each helper writes a high-resolution PNG and returns its path. Colours
and labels are kept consistent so the figures read as a set (V3 §9.3).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - must follow the backend selection

_PALETTE = ("#4c72b0", "#dd8452", "#55a868", "#c44e52")


def bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    out_path: str | Path,
    *,
    log: bool = False,
) -> Path:
    """Write a labelled bar chart to ``out_path`` and return the path."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=_PALETTE[: len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    ax.bar_label(bars, fmt="%.3g", padding=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
