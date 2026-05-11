"""Trade audit log — write-append + calendar-day spend tally.

One file per ranker run at `logs/trades/<run_id>.jsonl`, written
append-mode (one JSON object per line). The calendar-day spend tally
globs every file under `logs/trades/` and sums `fill_total_cost_cents`
for rows whose `audit_timestamp` falls on today's UTC date — survives
multiple ranker runs in a single day without double-counting.

The `audit_timestamp` is set when the row is constructed (just before
write), so a slow / failing order placement that takes 30s still
attributes the trade to the day it was attempted on, not the day the
ranker JSONL was written.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from skimsmarkets.retro.jsonl import trades_log_path, trades_log_root
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


def today_spend_cents(*, now: datetime | None = None) -> int:
    """Sum filled `fill_total_cost_cents` for every trade row stamped today (UTC).

    Iterates every `*.jsonl` in `logs/trades/`. Malformed lines log a
    warning and are skipped — the same discipline as `iter_predictions`
    in the retro reader. Skipped / dry-run rows contribute 0 (their
    `fill_total_cost_cents` defaults to 0).

    `now` is injectable for tests; production calls pass nothing and
    we anchor on `datetime.now(UTC)`.
    """
    anchor = now or datetime.now(UTC)
    today = anchor.date()
    root = trades_log_root()
    if not root.exists():
        return 0
    total = 0
    for path in root.glob("*.jsonl"):
        with path.open() as f:
            for line_num, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as e:
                    log.warning(
                        "trades reader: %s line %d malformed (%s)",
                        path.name, line_num, e,
                    )
                    continue
                if not isinstance(payload, dict):
                    continue
                if payload.get("record_type") != "trade":
                    continue
                try:
                    row = TradeRow.model_validate(payload)
                except ValidationError as e:
                    log.warning(
                        "trades reader: %s line %d schema mismatch (%s)",
                        path.name, line_num, e,
                    )
                    continue
                if row.audit_timestamp.astimezone(UTC).date() != today:
                    continue
                total += row.fill_total_cost_cents
    return total
