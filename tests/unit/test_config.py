"""Tests for the versioned config loader (V3 S8/S9)."""

import json
from pathlib import Path

import pytest

from airllm_bench.shared.config import ConfigError, ConfigLoader, LoadedConfig

# Real config dir shipped with the repo, used by the integration test below.
_REPO_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def _write(dir_path: Path, name: str, payload: dict) -> None:
    """Write a config payload to ``<dir>/<name>.json``."""
    (dir_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_valid_config_returns_data(tmp_path: Path) -> None:
    """A well-formed, correctly-versioned config loads and exposes its data."""
    _write(tmp_path, "setup", {"version": "1.00", "models": {"main": "x"}})
    loaded = ConfigLoader(tmp_path).load("setup", "1.00")
    assert isinstance(loaded, LoadedConfig)
    assert loaded.version == "1.00"
    assert loaded.data["models"]["main"] == "x"


def test_missing_file_raises(tmp_path: Path) -> None:
    """A missing config file is a fail-fast ConfigError."""
    with pytest.raises(ConfigError, match="not found"):
        ConfigLoader(tmp_path).load("nope", "1.00")


def test_invalid_json_raises(tmp_path: Path) -> None:
    """Malformed JSON is reported as a ConfigError, not a raw decode error."""
    (tmp_path / "broken.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid JSON"):
        ConfigLoader(tmp_path).load("broken", "1.00")


def test_version_mismatch_raises(tmp_path: Path) -> None:
    """A config whose version differs from expected is rejected."""
    _write(tmp_path, "setup", {"version": "2.00"})
    with pytest.raises(ConfigError, match="!= expected"):
        ConfigLoader(tmp_path).load("setup", "1.00")


def test_missing_version_key_raises(tmp_path: Path) -> None:
    """A config without the expected version key is rejected."""
    _write(tmp_path, "setup", {"models": {}})
    with pytest.raises(ConfigError, match="Missing version key"):
        ConfigLoader(tmp_path).load("setup", "1.00")


def test_nested_version_key(tmp_path: Path) -> None:
    """A dotted version_key reads a nested version (e.g. rate_limits.version)."""
    _write(tmp_path, "rate_limits", {"rate_limits": {"version": "1.00"}})
    loaded = ConfigLoader(tmp_path).load("rate_limits", "1.00", version_key="rate_limits.version")
    assert loaded.version == "1.00"


def test_real_repo_configs_load() -> None:
    """The actual shipped config files load and validate at version 1.00."""
    loader = ConfigLoader(_REPO_CONFIG_DIR)
    assert loader.load("setup", "1.00").data["models"]
    assert loader.load("costs", "1.10").data["on_prem"]
    rate = loader.load("rate_limits", "1.00", version_key="rate_limits.version")
    assert rate.data["rate_limits"]["services"]["default"]
