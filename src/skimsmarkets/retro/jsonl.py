"""JSONL run-log reader for the retro layer.

`pipeline._persist_run` writes prediction rows inline as plain dicts
(no single Pydantic dump), so this is the canonical reader. Skips
`record_type="error"` rows silently — those represent dropped events
the director never produced a prediction for, so they have nothing to
retro against. Bad JSON / unknown shapes log a warning and are skipped
rather than aborting the read (a corrupted line shouldn't kill an
otherwise-valid retro run).

`run_id` is parsed from the filename rather than the row payload so
`<run_id>.jsonl` and the row's `run_id` field remain consistent — they
are the same value today but the filename is the authoritative source
in case of future divergence.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError

from skimsmarkets.retro.models import PredictionRow

log = logging.getLogger(__name__)

# Resolved at module load to keep behaviour stable regardless of cwd.
# `parents[2]` walks src/skimsmarkets/retro/jsonl.py → src/skimsmarkets
# → src → repo-root.
_LOG_ROOT = Path(__file__).resolve().parents[3] / "logs" / "runs"
_TRADES_ROOT = Path(__file__).resolve().parents[3] / "logs" / "trades"


def list_run_files() -> list[Path]:
    """All `*.jsonl` files under `logs/runs/`, sorted by mtime descending.

    Excludes the resolution sidecars (`*.resolutions.jsonl`) so callers
    walking the run history don't double-count them as run logs.
    """
    if not _LOG_ROOT.exists():
        return []
    files = [
        p for p in _LOG_ROOT.glob("*.jsonl")
        if not p.name.endswith(".resolutions.jsonl")
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def run_path_for_id(run_id: str) -> Path:
    """Return the canonical path for a run id (does not check existence)."""
    return _LOG_ROOT / f"{run_id}.jsonl"


def iter_predictions(path: Path) -> Iterator[PredictionRow]:
    """Yield `PredictionRow` objects from one JSONL file.

    Skips error rows, malformed JSON, and rows that fail Pydantic
    validation (with a warning). Caller decides whether to materialise
    into a list — most callers iterate once and consume.
    """
    if not path.exists():
        return
    with path.open() as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(
                    "retro reader: %s line %d: malformed JSON (%s)",
                    path.name, line_num, e,
                )
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("record_type") != "prediction":
                continue
            try:
                yield PredictionRow.model_validate(payload)
            except ValidationError as e:
                log.warning(
                    "retro reader: %s line %d: schema mismatch (%s)",
                    path.name, line_num, e,
                )
                continue


def iter_all_predictions() -> Iterator[tuple[Path, PredictionRow]]:
    """Yield every prediction row across every run log under `logs/runs/`.

    Tuple shape `(path, row)` so callers can attribute rows to source
    files (useful for joining against the per-run resolution sidecar).
    """
    for path in list_run_files():
        for row in iter_predictions(path):
            yield path, row


def resolutions_sidecar_path(run_path: Path) -> Path:
    """Sidecar path: `<run_id>.resolutions.jsonl` next to the source log."""
    return run_path.with_suffix(".resolutions.jsonl")


def log_root() -> Path:
    """Module-resolved path to `logs/runs/`. Exposed for callers that
    need to write outside the prediction file (e.g. resolution sidecars).
    """
    return _LOG_ROOT


def trades_log_path(run_id: str) -> Path:
    """Return `logs/trades/<run_id>.jsonl` for the execute audit log.

    Separate directory from `logs/runs/` because trades and predictions
    are different concerns (compare with the resolutions sidecar,
    which lives next to the prediction log because it's downstream of
    a single run). The trades directory is created on first write by
    `execute/audit.py`.
    """
    return _TRADES_ROOT / f"{run_id}.jsonl"


def trades_log_root() -> Path:
    """Module-resolved path to `logs/trades/`. Used by the calendar-day
    spend tally — it globs every `*.jsonl` here and sums today's fills.
    """
    return _TRADES_ROOT
