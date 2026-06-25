#!/usr/bin/env python
"""Secret scanner (#30) — fail if a likely secret is committed (V3 S4).

Scans git-tracked text files for high-signal secret patterns (HuggingFace /
OpenAI / AWS / GitHub tokens, private-key blocks) and exits non-zero if any are
found. Wired into CI (.github/workflows/ci.yml); run locally with:

    uv run python scripts/secret_scan.py

Secrets live only in ``.env`` (git-ignored, never scanned) — see ``.env-example``.
"""

import re
import subprocess
import sys
from pathlib import Path

PATTERNS = {
    "HuggingFace token": re.compile(r"hf_[A-Za-z0-9]{34,}"),
    "OpenAI key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    "Private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".lock"}
SKIP_FILES = {"scripts/secret_scan.py", ".env-example"}


def tracked_files() -> list[str]:
    """Return the repo's git-tracked file paths."""
    result = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
    return [line for line in result.stdout.splitlines() if line]


def scan_file(path: str) -> list[tuple[str, int, str]]:
    """Return (path, line, pattern-name) for each secret-looking match in a file."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    findings = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for name, pattern in PATTERNS.items():
            if pattern.search(line):
                findings.append((path, lineno, name))
    return findings


def main() -> None:
    """Scan all tracked text files; exit 1 if any secret is found."""
    findings: list[tuple[str, int, str]] = []
    for path in tracked_files():
        if path in SKIP_FILES or Path(path).suffix.lower() in SKIP_SUFFIXES:
            continue
        findings.extend(scan_file(path))

    if findings:
        print("SECRETS DETECTED:")
        for path, lineno, name in findings:
            print(f"  {path}:{lineno}  {name}")
        sys.exit(1)
    print("secret scan: clean (no secrets in tracked files)")


if __name__ == "__main__":
    main()
