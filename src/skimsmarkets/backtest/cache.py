"""Tiny on-disk cache for backtest data — JSON files keyed by name.

Why not sqlite/parquet here: the cache holds whatever the API returns, often
nested and ragged (events with N markets, price series of varying length).
JSON keeps it inspectable with `cat | jq`. Heavy aggregation lives downstream
in pandas frames written to parquet.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CACHE_ROOT = Path(__file__).resolve().parents[3] / "backtest_cache"


def cache_path(*parts: str) -> Path:
    p = CACHE_ROOT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load(*parts: str) -> Any | None:
    p = cache_path(*parts)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def save(value: Any, *parts: str) -> None:
    p = cache_path(*parts)
    # write to a tmp file then rename — atomic against ctrl-c mid-write.
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, separators=(",", ":")))
    os.replace(tmp, p)


def exists(*parts: str) -> bool:
    return cache_path(*parts).exists()
