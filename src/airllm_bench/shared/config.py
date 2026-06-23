"""Versioned JSON config loader with startup version validation (V3 S8).

All configurable values live in ``config/*.json`` (V3 S3); this module is the
single place that reads them, so the rest of the harness never touches raw
files. Each config carries a version that is checked against the version the
code expects, failing fast on an incompatible config rather than misbehaving
later with stale settings.
"""

import json
from dataclasses import dataclass
from pathlib import Path

# Default config directory: <repo-root>/config, resolved relative to this file
# (src/airllm_bench/shared/config.py -> parents[3] is the repo root).
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


class ConfigError(Exception):
    """Raised when a config file is missing, invalid, or version-incompatible."""


@dataclass(frozen=True)
class LoadedConfig:
    """A successfully loaded, version-checked config file."""

    name: str
    version: str
    data: dict


def _read_json(path: Path) -> dict:
    """Read and parse a JSON file, raising ConfigError on any problem."""
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc


def _extract_version(data: dict, version_key: str) -> str:
    """Read a (possibly dotted) version key out of a config mapping."""
    node: object = data
    for part in version_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"Missing version key '{version_key}'")
        node = node[part]
    return str(node)


class ConfigLoader:
    """Load and version-check the project's JSON config files.

    Setup:  config_dir (defaults to <repo-root>/config; inject another for tests).
    Input:  a config name, the expected version, and where the version lives.
    Output: a LoadedConfig, or a ConfigError on any failure.
    """

    def __init__(self, config_dir: Path | None = None) -> None:
        self.config_dir = config_dir or _DEFAULT_CONFIG_DIR

    def load(
        self,
        name: str,
        expected_version: str,
        version_key: str = "version",
    ) -> LoadedConfig:
        """Load ``<name>.json``, validate its version, and return it.

        Raises ConfigError if the file is missing, not valid JSON, lacks the
        version key, or its version does not match ``expected_version``.
        """
        path = self.config_dir / f"{name}.json"
        data = _read_json(path)
        version = _extract_version(data, version_key)
        if version != expected_version:
            raise ConfigError(f"{name}.json version {version!r} != expected {expected_version!r}")
        return LoadedConfig(name=name, version=version, data=data)
