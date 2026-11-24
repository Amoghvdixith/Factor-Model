"""Configuration loading and portable project paths.

All paths are resolved relative to the project root, which is found by walking up from
the current working directory (or this file) to the folder containing ``pyproject.toml``
with ``name = "factor-model"``.  The ``FACTORS_ROOT`` environment variable overrides it.
No absolute machine-specific path is stored anywhere in the project.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised for a missing or invalid configuration value."""


def _is_root(p: Path) -> bool:
    f = p / "pyproject.toml"
    return f.exists() and 'name = "factor-model"' in f.read_text(encoding="utf-8")


def project_root() -> Path:
    """Return the project root directory."""
    env = os.environ.get("FACTORS_ROOT")
    if env:
        return Path(env).resolve()
    for start in (Path.cwd(), Path(__file__).resolve()):
        for p in (start, *start.parents):
            if _is_root(p):
                return p
    raise ConfigError("could not locate the factor-model project root; set FACTORS_ROOT")


def load_config(name: str, root: Path | None = None) -> dict[str, Any]:
    """Load ``configs/<name>.yaml`` as a dict."""
    root = root or project_root()
    path = root / "configs" / f"{name}.yaml"
    if not path.exists():
        raise ConfigError(f"missing config file configs/{name}.yaml")
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ConfigError(f"configs/{name}.yaml must contain a mapping")
    return cfg


def load_all(root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load every project configuration file."""
    return {n: load_config(n, root) for n in
            ("data", "timeseries", "characteristics", "risk_model", "portfolio")}


def rel(path: Path, root: Path | None = None) -> str:
    """Path relative to the project root, with forward slashes (for metadata/logs)."""
    root = root or project_root()
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return Path(path).name
