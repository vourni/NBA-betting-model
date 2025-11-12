from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict


@lru_cache(maxsize=1)
def load_env(dotenv_path: str | Path | None = None) -> Dict[str, str]:
    """
    Lightweight .env reader that populates os.environ while returning parsed values.
    Prefers caller-supplied paths but defaults to the project root .env file.
    """
    path = Path(dotenv_path).expanduser() if dotenv_path else Path(__file__).resolve().parent / ".env"
    if not path.exists():
        return {}

    parsed: Dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        parsed[key] = value
        os.environ.setdefault(key, value)
    return parsed


def get_env(name: str, *, required: bool = True, default: str | None = None) -> str | None:
    """
    Lookup helper that optionally enforces presence of required secrets.
    """
    load_env()
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value
