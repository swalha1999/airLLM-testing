"""Result recorder — persist a versioned result record to ``results/`` (PLAN §6).

Every scenario's raw numbers are saved as JSON so figures can be regenerated
without re-running models (NFR1). Each record is stamped with the result schema
version and the harness code version (V3 S8) plus a UTC timestamp, then written
under ``results/`` with a filesystem-safe, scenario-prefixed filename.
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .version import __version__

SCHEMA_VERSION = "1.00"

# Default output dir: <repo-root>/results (recorder.py is under src/<pkg>/shared).
_DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"


class ResultRecorder:
    """Write result records to disk as JSON.

    Setup:  ``results_dir`` (injectable; defaults to <repo-root>/results) and a
            ``now_fn`` clock injected so filenames/timestamps are deterministic
            in tests.
    Input:  a record mapping (scenario, model, quant, prompt_id, metrics, ...).
    Output: the :class:`~pathlib.Path` written.
    """

    def __init__(
        self,
        results_dir: Path | None = None,
        *,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.results_dir = results_dir or _DEFAULT_RESULTS_DIR
        self._now = now_fn or (lambda: datetime.now(UTC))

    def save(self, record: dict) -> Path:
        """Enrich, then write ``record`` as JSON; return the file path."""
        enriched = self._enrich(record)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        path = self.results_dir / self._filename(enriched)
        path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
        return path

    def _enrich(self, record: dict) -> dict[str, Any]:
        """Stamp the record with schema/code versions and a UTC timestamp."""
        return {
            "schema_version": SCHEMA_VERSION,
            "code_version": __version__,
            "timestamp": self._now().isoformat(),
            **record,
        }

    def _filename(self, record: dict) -> str:
        """Build a filesystem-safe, scenario-prefixed filename."""
        scenario = record.get("scenario", "result")
        stamp = record["timestamp"].replace(":", "-")
        return f"{scenario}_{stamp}.json"
