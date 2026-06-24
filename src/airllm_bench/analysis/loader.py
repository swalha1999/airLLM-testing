"""Load recorded result JSONs from ``results/`` for analysis (PLAN §6).

A small, pure-Python helper so analysis scripts and notebooks regenerate tables
and figures straight from the saved raw numbers (NFR1) — no re-running models.
Reused across the quantization comparison (#18), metric consolidation (#19), and
the figures (#22).
"""

import json
from pathlib import Path

# Default: <repo-root>/results (loader.py is under src/<pkg>/analysis).
_DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"


def load_results(results_dir: Path | None = None) -> list[dict]:
    """Load every result record from ``results/``, sorted by filename."""
    directory = Path(results_dir) if results_dir else _DEFAULT_RESULTS_DIR
    if not directory.is_dir():
        return []
    return [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))
    ]


def latest_by_scenario(records: list[dict], scenario: str) -> dict | None:
    """Return the most recent record for ``scenario`` (by timestamp), or None."""
    matches = [r for r in records if r.get("scenario") == scenario]
    if not matches:
        return None
    return max(matches, key=lambda record: record.get("timestamp", ""))
