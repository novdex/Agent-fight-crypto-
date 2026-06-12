"""Config loading. FROZEN INTERFACE — see docs/INTERFACES.md."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from arena.models import ArenaConfig


def load_config(path: str | Path = "config.yaml") -> ArenaConfig:
    """Load config.yaml and .env (if present) from the working directory."""
    load_dotenv()
    p = Path(path)
    if not p.exists():
        return ArenaConfig()
    with open(p) as f:
        raw = yaml.safe_load(f) or {}
    return ArenaConfig.model_validate(raw)


def api_key_for(env_name: str) -> str:
    """Return the API key from the environment, or '' if unset."""
    if not env_name:
        return ""
    return os.environ.get(env_name, "")
