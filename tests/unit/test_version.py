"""Tests for harness version tracking (V3 S8)."""

import re

from airllm_bench import __version__
from airllm_bench.shared.version import get_version


def test_get_version_matches_package_export() -> None:
    """The package-level export and the module accessor agree."""
    assert get_version() == __version__


def test_version_follows_major_minor_format() -> None:
    """Version is a ``MAJOR.MINOR`` string starting at 1.00 (V3 S8)."""
    assert re.fullmatch(r"\d+\.\d{2}", __version__)
