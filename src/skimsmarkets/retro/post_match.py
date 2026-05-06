"""Step 3 fetcher — pull per-match box scores for resolved tennis events.

Iterates settled tennis predictions, calls the active
`TennisStatsProvider.fetch_post_match_stats` for both players, and
persists the result to a per-event cache so reruns don't re-hit the
vendor. Cache miss → vendor call. Cache hit → no HTTP. Vendor miss
→ cached as the empty pair (`player_a=None`, `player_b=None`) so
subsequent runs don't re-attempt.

Match date inference: the `wta-/atp-…-YYYY-MM-DD` slug carries the
calendar date the match is scheduled for, which matches the vendor's
`past-matches.date` for completed matches in 99%+ of cases. When the
slug date is malformed or missing we skip the event with a warning.

Concurrency cap: pulls both players in parallel inside one event, plus
fans out across events with a small semaphore. The MatchStat token
bucket inside the provider enforces the 5 req/sec ceiling, so this
just prevents queue bloat.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date
from pathlib import Path

from pydantic import ValidationError

from skimsmarkets.retro.jsonl import (
    iter_predictions,
    list_run_files,
    log_root,
    resolutions_sidecar_path,
)
from skimsmarkets.retro.models import (
    PredictionRow,
    ResolvedOutcome,
    RetroPostMatchPair,
)
from skimsmarkets.tennis.provider import TennisStatsProvider

log = logging.getLogger(__name__)

# Tennis slug shape (pre-LLM example): atp-zhang-altmaie-2026-05-06.
# Date is always last; tour prefix is `atp-` or `wta-`. Regex pinned
# to those because the retro layer only does post-match fetch for
# tennis (the only sport with a vendor adapter today).
_SLUG_DATE_RE = re.compile(r"^(atp|wta)-.+-(\d{4}-\d{2}-\d{2})$")

# Cache root: logs/retro/post_match/<tour>/<player_a_id>_<event_date>.json.
# `event_date` is the slug date (calendar day), not the vendor date —
# the vendor occasionally drops a row past midnight UTC for an
# evening match in a Western tz. Slug date is the source of truth.
_CACHE_ROOT = log_root().parent / "retro" / "post_match"

# Per-event concurrency. Provider's token bucket throttles below this;
# this just prevents the asyncio.gather queue from ballooning when
# someone runs retro across hundreds of past events.
_FETCH_CONCURRENCY = 8


def _parse_slug_date(slug: str) -> tuple[str, date] | None:
    """Return `(tour, match_date)` or None when slug doesn't match the
    canonical tennis pattern (non-tennis events fall through cleanly).
    """
    m = _SLUG_DATE_RE.match(slug)
    if not m:
        return None
    tour, ymd = m.group(1), m.group(2)
    try:
        return tour, date.fromisoformat(ymd)
    except ValueError:
        return None


def _cache_path(tour: str, player_a_id: int, on_date: date) -> Path:
    """Cache path includes the tour because player IDs aren't unique
    across tours (vendor IDs are tour-scoped). Also includes the date
    so two matches of the same player on different days don't
    collide.
    """
    return (
        _CACHE_ROOT
        / tour
        / f"{player_a_id}_{on_date.isoformat()}.json"
    )


def _read_cache(path: Path) -> RetroPostMatchPair | None:
    if not path.exists():
        return None
    try:
        return RetroPostMatchPair.model_validate_json(path.read_text())
    except (ValidationError, json.JSONDecodeError) as e:
        log.warning("retro post-match: corrupt cache %s: %s", path, e)
        return None


def _write_cache(path: Path, pair: RetroPostMatchPair) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pair.model_dump_json(indent=2))


async def fetch_post_match_for_row(
    provider: TennisStatsProvider,
    row: PredictionRow,
    sem: asyncio.Semaphore,
) -> RetroPostMatchPair | None:
    """Fetch (or cache-hit) the post-match pair for one resolved tennis
    prediction. Returns None when the event isn't tennis-shaped (slug
    doesn't match the regex), the prediction lacks tennis_stats, or
    one/both vendor calls miss.

    Cache-write semantics: even when one or both vendor calls return
    None (vendor has no row for that match), we WRITE the cache with
    None values so reruns don't re-attempt — these are persistent
    misses, not transient. The operator can `rm` a specific cache
    file to force a re-fetch.
    """
    if row.tennis_stats is None:
        return None
    parsed = _parse_slug_date(row.market_slug)
    if parsed is None:
        return None
    tour, on_date = parsed
    pa = row.tennis_stats.player_a
    pb = row.tennis_stats.player_b
    pa_id_str = pa.api_player_id
    pb_id_str = pb.api_player_id
    if pa_id_str is None or pb_id_str is None:
        return None
    try:
        pa_id = int(pa_id_str)
        pb_id = int(pb_id_str)
    except ValueError:
        log.warning(
            "retro post-match: non-int player ids on %s (a=%r b=%r)",
            row.event_id, pa_id_str, pb_id_str,
        )
        return None

    cache_path = _cache_path(tour, pa_id, on_date)
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached

    async with sem:
        # Fetch both sides in parallel inside one event. Provider's
        # token bucket serialises across all in-flight calls.
        actual_a, actual_b = await asyncio.gather(
            provider.fetch_post_match_stats(tour, pa_id, on_date, pb.name),
            provider.fetch_post_match_stats(tour, pb_id, on_date, pa.name),
        )
    pair = RetroPostMatchPair(
        event_id=row.event_id,
        on_date=on_date,
        player_a_name=pa.name,
        player_b_name=pb.name,
        player_a=actual_a,
        player_b=actual_b,
    )
    _write_cache(cache_path, pair)
    return pair


async def fetch_post_match_for_settled(
    provider: TennisStatsProvider,
) -> dict[str, RetroPostMatchPair]:
    """Fetch post-match pairs for every settled tennis prediction across
    all run logs. Returns a dict keyed by `event_id`.

    Only settled events are fetched — unsettled means the match either
    hasn't been played, was cancelled (50-50), or had a label-mismatch
    we can't grade. Either way, no useful divergence to compute.
    """
    sem = asyncio.Semaphore(_FETCH_CONCURRENCY)
    out: dict[str, RetroPostMatchPair] = {}
    tasks: list[asyncio.Task[RetroPostMatchPair | None]] = []

    for run_path in list_run_files():
        sidecar = resolutions_sidecar_path(run_path)
        outcomes_by_slug: dict[str, ResolvedOutcome] = {}
        if sidecar.exists():
            with sidecar.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        outcome = ResolvedOutcome.model_validate_json(line)
                    except Exception:  # noqa: BLE001
                        continue
                    outcomes_by_slug[outcome.slug] = outcome

        for row in iter_predictions(run_path):
            outcome = outcomes_by_slug.get(row.market_slug)
            if outcome is None or not outcome.settled:
                continue
            if row.sport_type != "tennis":
                continue
            tasks.append(
                asyncio.create_task(fetch_post_match_for_row(provider, row, sem))
            )

    if not tasks:
        return out
    results = await asyncio.gather(*tasks)
    for pair in results:
        if pair is None:
            continue
        out[pair.event_id] = pair
    return out
