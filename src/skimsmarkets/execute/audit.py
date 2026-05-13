"""Trade audit log — write-append + per-run executed-event lookup.

One file per ranker run at `logs/trades/<run_id>.jsonl`, written
append-mode (one JSON object per line). The `audit_timestamp` is set
when the row is constructed (just before write), so a slow / failing
order placement that takes 30s still attributes the trade to the time
it was attempted on.

The previous calendar-day spend tally was retired on 2026-05-12 in
favour of reading live open exposure from Kalshi's
`/portfolio/positions` — see `trader._prefetch_open_exposure`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from skimsmarkets.retro.jsonl import trades_log_path
from skimsmarkets.retro.models import TradeRow

log = logging.getLogger(__name__)

# Audit `fill_status` values that mean "the order reached Kalshi and
# took (or probably took) effect" — used for intra-run idempotency.
# `submitted` is included on the safe side: an order_id was returned
# even if our parser couldn't read the fill counts, so a duplicate
# would risk double-execution.
_EXECUTED_STATUSES: frozenset[str] = frozenset(
    {"filled", "partial", "submitted"}
)


def write_trade_row(row: TradeRow, path: Path) -> None:
    """Append one `TradeRow` to `path` as a JSON line.

    Creates the parent directory on first write. Uses Pydantic's
    JSON serialisation (handles datetime → ISO 8601 cleanly) and
    forces a single line per row.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = row.model_dump_json()
    with path.open("a") as f:
        f.write(line + "\n")


def executed_event_ids(run_id: str) -> set[str]:
    """Return event_ids that already have an "executed" audit row in this run.

    Used by the trader to dedupe within a run_id: re-running `skims
    execute --live` against the same run shouldn't place a second
    order for a prediction we already acted on. Scoped to the single
    run (not cross-run) so two ranker runs that happen to share an
    event still produce two separate trades — they represent two
    fresh signals.

    Reads `logs/trades/<run_id>.jsonl`. Missing file = empty set.
    Malformed lines and unknown statuses are skipped silently — the
    same posture as the rest of the JSONL readers.
    """
    path = trades_log_path(run_id)
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("record_type") != "trade":
                continue
            if payload.get("fill_status") not in _EXECUTED_STATUSES:
                continue
            event_id = payload.get("event_id")
            if isinstance(event_id, str) and event_id:
                seen.add(event_id)
    return seen
