#!/usr/bin/env python
"""Generate an automated test report into reports/ (#32, V3 §6.4).

Runs the test suite in-process (so it works under ``uv run`` without nesting
another ``uv``), captures the pytest + coverage output, and writes it to
``reports/test_report.md`` with a pass/fail headline. Exits with pytest's code,
so a failing suite fails the report generation too.

    uv run python scripts/test_report.py
"""

import contextlib
import io
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPORT = Path("reports/test_report.md")


def main() -> None:
    """Run the suite, capture output, and write the markdown report."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exit_code = int(pytest.main(["-q", "tests"]))
    output = buffer.getvalue()

    status = "PASS ✅" if exit_code == 0 else "FAIL ❌"
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        f"# Automated Test Report\n\n"
        f"- **Result:** {status} (pytest exit code {exit_code})\n"
        f"- **Generated:** {datetime.now(UTC).isoformat()}\n"
        f"- **Command:** `uv run python scripts/test_report.py`\n\n"
        f"The coverage gate (≥85%, config-driven) and the full suite output:\n\n"
        f"```\n{output.strip()}\n```\n",
        encoding="utf-8",
    )
    print(f"wrote {REPORT} (pytest exit code {exit_code})")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
