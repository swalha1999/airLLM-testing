#!/usr/bin/env python
"""150-line file checker (#31, V3 S1).

Every Python file must be at most 150 *code* lines — blank lines and full-line
``#`` comments are not counted (V3 S1). Scans git-tracked ``.py`` files and exits
non-zero if any exceed the limit. Wired into CI (.github/workflows/ci.yml); run
locally with:

    uv run python scripts/check_line_limit.py
"""

import subprocess
import sys
from pathlib import Path

LIMIT = 150  # V3 S1 rule constant (max code lines per file)


def code_line_count(path: Path) -> int:
    """Count non-blank, non-comment lines in a file (V3's 'lines of code')."""
    count = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        count += 1
    return count


def python_files() -> list[str]:
    """Return git-tracked ``.py`` file paths."""
    result = subprocess.run(["git", "ls-files", "*.py"], capture_output=True, text=True, check=True)
    return [line for line in result.stdout.splitlines() if line]


def main() -> None:
    """Check every tracked .py file; exit 1 if any exceed the limit."""
    violations = [
        (path, count) for path in python_files() if (count := code_line_count(Path(path))) > LIMIT
    ]
    if violations:
        print(f"FILES OVER {LIMIT} CODE LINES:")
        for path, count in violations:
            print(f"  {path}: {count}")
        sys.exit(1)
    print(f"line-limit check: all tracked .py files <= {LIMIT} code lines")


if __name__ == "__main__":
    main()
