"""MatchStat historical-rankings backfill for the tennis GBT corpus.

Hits the standard `/ranking/singles` endpoint with the `RankingDate`
filter to retrieve weekly historical snapshots and persists one row
per (tour, ranking_date, player_id) under
`data/tennis_gbt/rankings_history.parquet`.

The vendor's `?filter=RankingDate:YYYY-MM-DD` returns "the ranking
snapshot on or before this date." We walk Monday-by-Monday across the
match data's date range and dedup by the snapshot's own `date` field
so frozen periods (e.g. 2020 COVID rankings freeze, year-end gaps)
don't bloat the table — the same snapshot returned across several
weeks collapses to one row per player.

Reuses the production `_TokenBucket` (5 req/sec) from `matchstat.py`
so the backfill respects the same vendor budget the live pipeline
does — no separate quota policy to maintain.

Cold-start gating is NOT enforced here. The raw table carries every
ranked player; the algorithmic-lens feature builder is allowed to
drop rows whose denominators are too thin.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from skimsmarkets.tennis.matchstat import (
    _BURST_TOKENS,
    _REQUESTS_PER_SECOND,
    _RETRY_ATTEMPTS,
    _RETRY_BASE_S,
    _TokenBucket,
    _coerce_int,
    _parse_iso_date,
)

log = logging.getLogger(__name__)

_BASE = "https://tennis-api-atp-wta-itf.p.rapidapi.com"
_HOST = "tennis-api-atp-wta-itf.p.rapidapi.com"

_PAGE_SIZE = 100

# Output path. Sits next to `raw_matches.parquet` + `player_profiles.parquet`
# under the same gitignored data directory.
_DATA_DIR = Path("data/tennis_gbt")
RANKINGS_HISTORY_PATH = _DATA_DIR / "rankings_history.parquet"


def _headers(api_key: str) -> dict[str, str]:
    return {"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": _HOST}


async def _get_json(
    client: httpx.AsyncClient,
    bucket: _TokenBucket,
    path: str,
    params: dict[str, Any] | None = None,
) -> Any:
    """Token-bucket-throttled GET with exponential-backoff retry on 429.

    Inlined rather than imported from `gbt_backfill.py` so each backfill
    module is independently runnable without cross-imports.

    Detects the vendor's body-level throttle (HTTP 200 with
    `{"error": true, "statusCode": 429, "message": "ThrottlerException"}`)
    and treats it as a 429. Without this check the call returns
    "success" but with no `data` key, which the caller can't distinguish
    from a genuinely empty snapshot — silently dropping rows.
    """
    last_exc: Exception | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        await bucket.acquire()
        try:
            r = await client.get(path, params=params)
            if r.status_code == 429:
                wait = _RETRY_BASE_S * (attempt + 1)
                log.warning("429 from %s — sleeping %.1fs", path, wait)
                await asyncio.sleep(wait)
                continue
            r.raise_for_status()
            body = r.json()
            # Body-level throttle: HTTP 200 OK but the response carries
            # an embedded throttler error instead of the data payload.
            if (
                isinstance(body, dict)
                and body.get("error") is True
                and body.get("statusCode") == 429
            ):
                wait = _RETRY_BASE_S * (attempt + 1)
                log.warning(
                    "body-level 429 from %s (msg=%r) — sleeping %.1fs",
                    path, body.get("message"), wait,
                )
                await asyncio.sleep(wait)
                continue
            return body
        except Exception as e:  # noqa: BLE001
            last_exc = e
            log.warning(
                "attempt %d/%d %s: %s",
                attempt + 1, _RETRY_ATTEMPTS, path, type(e).__name__,
            )
            await asyncio.sleep(_RETRY_BASE_S * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    return None


def _mondays(start: date, end: date) -> Iterable[date]:
    """Yield every Monday between `start` and `end` inclusive. Snaps
    `start` forward to the next Monday if it lands mid-week.
    """
    days_to_monday = (7 - start.weekday()) % 7
    cur = start + timedelta(days=days_to_monday)
    while cur <= end:
        yield cur
        cur = cur + timedelta(days=7)


async def _fetch_snapshot_pages(
    client: httpx.AsyncClient,
    bucket: _TokenBucket,
    tour: str,
    ranking_date_query: date,
    top_n: int,
) -> list[dict[str, Any]]:
    """Fetch one weekly snapshot at the vendor's snap-to-on-or-before
    date, paginating up to `top_n` players. Returns flat row dicts.
    """
    out: list[dict[str, Any]] = []
    pages_needed = max(1, (top_n + _PAGE_SIZE - 1) // _PAGE_SIZE)
    for page in range(1, pages_needed + 1):
        body = await _get_json(
            client, bucket,
            f"/tennis/v2/{tour}/ranking/singles",
            params={
                "pageSize": _PAGE_SIZE,
                "pageNo": page,
                "filter": f"RankingDate:{ranking_date_query.isoformat()}",
            },
        )
        if not isinstance(body, dict):
            break
        rows = body.get("data")
        if not isinstance(rows, list) or not rows:
            break
        for row in rows:
            if not isinstance(row, dict):
                continue
            player = row.get("player")
            if not isinstance(player, dict):
                continue
            pid = _coerce_int(player.get("id"))
            pos = _coerce_int(row.get("position"))
            pts = _coerce_int(row.get("point"))
            # `date` on each row is the vendor's actual snapshot date.
            # Identical across many weekly queries during frozen periods.
            snap = _parse_iso_date(row.get("date"))
            if pid is None or pos is None or snap is None:
                continue
            out.append({
                "tour": tour,
                "ranking_date": snap,
                "player_id": pid,
                "rank": pos,
                "rank_points": pts,
            })
        if len(rows) < _PAGE_SIZE:
            break
    return out


async def backfill_rankings(
    api_key: str,
    *,
    tours: Iterable[str] = ("atp", "wta"),
    start_date: date = date(2008, 12, 1),
    end_date: date | None = None,
    top_n: int = 500,
    timeout: float = 30.0,
) -> Path:
    """Run the historical-rankings backfill end-to-end.

    Returns the parquet path. Idempotent — overwrites the file on each
    invocation. Default `top_n=500` covers ~99%+ of match participants
    in the GBT past-matches corpus while keeping the backfill under
    ~30 minutes at the production rate-bucket budget.
    """
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    if end_date is None:
        end_date = date.today()

    bucket = _TokenBucket(_REQUESTS_PER_SECOND, _BURST_TOKENS)
    all_rows: list[dict[str, Any]] = []

    async with httpx.AsyncClient(
        base_url=_BASE,
        headers=_headers(api_key),
        timeout=timeout,
    ) as client:
        for tour in tours:
            mondays = list(_mondays(start_date, end_date))
            log.info(
                "backfill rankings %s — %d Mondays %s → %s, top %d",
                tour, len(mondays), mondays[0], mondays[-1], top_n,
            )
            for i, monday in enumerate(mondays, 1):
                rows = await _fetch_snapshot_pages(
                    client, bucket, tour, monday, top_n
                )
                all_rows.extend(rows)
                if i % 50 == 0 or i == len(mondays):
                    log.info(
                        "backfill rankings %s — %d/%d Mondays, %d rows",
                        tour, i, len(mondays), len(all_rows),
                    )

    if not all_rows:
        raise RuntimeError(
            "rankings backfill produced zero rows — check API key / network"
        )

    df = pd.DataFrame(all_rows)
    n_before = len(df)
    # Dedup by (tour, ranking_date, player_id). Frozen periods produce
    # the same snapshot across many weekly queries; one row per actual
    # snapshot is what we want.
    df = df.drop_duplicates(
        subset=["tour", "ranking_date", "player_id"], keep="first"
    )
    df = df.sort_values(["tour", "ranking_date", "rank"]).reset_index(drop=True)
    df["ranking_date"] = pd.to_datetime(df["ranking_date"])
    log.info(
        "deduped rankings rows: %d → %d (unique snapshot-rows)",
        n_before, len(df),
    )
    df.to_parquet(RANKINGS_HISTORY_PATH, index=False)
    log.info(
        "wrote %s (%.1f KB)",
        RANKINGS_HISTORY_PATH,
        RANKINGS_HISTORY_PATH.stat().st_size / 1024,
    )
    return RANKINGS_HISTORY_PATH


async def run_rankings_backfill_cli(
    *,
    tours: list[str] | None = None,
    top_n: int = 500,
    start_year: int | None = None,
) -> Path:
    """CLI entry — reads API key from env, runs backfill, returns the
    parquet path. Surfaces a loud RuntimeError when the key is missing
    so the operator sees the cause rather than a silent zero-row run.
    """
    from skimsmarkets import config as cfg
    config = cfg.Config.from_env(require_llm=False)
    if not config.tennis_stats_api_key:
        raise RuntimeError(
            "TENNIS_STATS_API_KEY not set; cannot backfill rankings."
        )
    start_date = date(start_year or 2008, 12, 1)
    return await backfill_rankings(
        config.tennis_stats_api_key,
        tours=tours or ["atp", "wta"],
        start_date=start_date,
        top_n=top_n,
    )
