"""MatchStat backfill for the tennis GBT spike.

Hits three vendor endpoints per tour and persists two parquet files
under `data/tennis_gbt/`:

  - `raw_matches.parquet` — one row per UNIQUE match (dedup'd by
    `match.id`, since each match appears in both players' past-matches
    feeds). Every box-score field the vendor ships is preserved as
    raw counters (numerator + denominator pairs); the feature builder
    converts to ratios at training time.
  - `player_profiles.parquet` — `(tour, player_id) → name, birthdate`.
    Joined into training rows to compute age-at-match-date without
    recapturing it per match.

Why two files: the match table is the corpus; the profile table is a
small static lookup. Splitting them keeps each parquet's schema
purpose-built and lets re-runs of the profile fetch be cheap when the
match table is already populated.

Reuses the production `_TokenBucket` (5 req/sec) from `matchstat.py`
so the backfill respects the same vendor budget the live pipeline
does — no separate quota policy to maintain. Reuses
`_normalize_name` likewise so the backfill table cross-joins cleanly
with the live identity normaliser.

Cold-start gate (≥ 20 priors per side) is enforced LATER in
`gbt_features.py`, not here. The raw table carries everything; the
feature builder is allowed to drop rows.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Iterable
from datetime import date
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
    _normalize_name,
    _parse_iso_date,
)

log = logging.getLogger(__name__)

_BASE = "https://tennis-api-atp-wta-itf.p.rapidapi.com/tennis/v2"
_HOST = "tennis-api-atp-wta-itf.p.rapidapi.com"

_RANKING_PAGE_SIZE = 100
_DEFAULT_PAST_MATCH_PAGE_SIZE = 100

# Output paths. `models/` is repo-tracked (the artifact is committed);
# `data/tennis_gbt/` has a `.gitignore` that drops `*.parquet` so the
# backfill output is regenerable without polluting git.
_DATA_DIR = Path("data/tennis_gbt")
RAW_MATCHES_PATH = _DATA_DIR / "raw_matches.parquet"
PLAYER_PROFILES_PATH = _DATA_DIR / "player_profiles.parquet"


def _headers(api_key: str) -> dict[str, str]:
    return {"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": _HOST}


async def _get_json(
    client: httpx.AsyncClient,
    bucket: _TokenBucket,
    path: str,
    params: dict[str, Any] | None = None,
) -> Any:
    """Token-bucket-throttled GET with exponential-backoff retry on 429.

    Mirrors the retry posture in `matchstat.py:_get` but standalone so
    the backfill doesn't drag in the full provider lifecycle (lazy
    rankings warmup, per-tour locks, etc. — none of which the backfill
    needs).
    """
    last_exc: Exception | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        await bucket.acquire()
        try:
            r = await client.get(path, params=params)
            if r.status_code == 429:
                # Back off + retry. Linear ramp is fine — the token
                # bucket already throttles steady-state.
                wait = _RETRY_BASE_S * (attempt + 1)
                log.warning("429 from %s — sleeping %.1fs", path, wait)
                await asyncio.sleep(wait)
                continue
            r.raise_for_status()
            body = r.json()
            # Body-level throttle: vendor sometimes returns HTTP 200 OK
            # but the response carries an embedded throttler error
            # instead of the data payload. Without this check the call
            # returns "success" but with no `data` key, silently
            # dropping rows on the caller side.
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
        except httpx.HTTPError as e:
            last_exc = e
            await asyncio.sleep(_RETRY_BASE_S * (attempt + 1))
    raise RuntimeError(f"GET {path} failed after {_RETRY_ATTEMPTS} attempts") from last_exc


async def _fetch_top_ids(
    client: httpx.AsyncClient,
    bucket: _TokenBucket,
    tour: str,
    top_n: int,
) -> list[tuple[int, str]]:
    """Fetch top-N (player_id, name) pairs for a tour from the rankings."""
    out: list[tuple[int, str]] = []
    pages = (top_n + _RANKING_PAGE_SIZE - 1) // _RANKING_PAGE_SIZE
    for page in range(1, pages + 1):
        body = await _get_json(
            client,
            bucket,
            f"{_BASE}/{tour}/ranking/singles",
            params={"pageSize": _RANKING_PAGE_SIZE, "pageNo": page},
        )
        for row in body.get("data", []):
            player = row.get("player") or {}
            pid = _coerce_int(player.get("id"))
            name = player.get("name")
            if pid is None or not name:
                continue
            out.append((pid, name))
            if len(out) >= top_n:
                return out
    return out


async def _fetch_profile(
    client: httpx.AsyncClient,
    bucket: _TokenBucket,
    tour: str,
    player_id: int,
) -> dict[str, Any]:
    """Fetch the profile payload for one player. Returns {} on failure
    so a single missing profile doesn't take down the backfill — the
    feature builder treats missing birthdate as missing-age.
    """
    try:
        body = await _get_json(
            client,
            bucket,
            f"{_BASE}/{tour}/player/profile/{player_id}",
            params={"include": "information"},
        )
    except Exception as e:  # noqa: BLE001
        log.warning("profile fetch failed for %s/%s: %s", tour, player_id, e)
        return {}
    return (body.get("data") or {}) if isinstance(body, dict) else {}


async def _fetch_past_matches(
    client: httpx.AsyncClient,
    bucket: _TokenBucket,
    tour: str,
    player_id: int,
    pages: int,
    page_size: int,
) -> list[dict[str, Any]]:
    """Fetch up to `pages × page_size` past matches for one player.

    Stops early if a page returns < page_size rows (vendor's signal
    that we've reached the end of available history). The vendor
    sometimes ships a final partial page; the empty-page check is
    `not rows`, so a partial page completes the loop next iteration.
    """
    out: list[dict[str, Any]] = []
    for page in range(1, pages + 1):
        body = await _get_json(
            client,
            bucket,
            f"{_BASE}/{tour}/player/past-matches/{player_id}",
            params={
                "pageSize": page_size,
                "pageNo": page,
                "include": "stat,tournament,round",
            },
        )
        rows = body.get("data") or []
        if not rows:
            break
        out.extend(rows)
        if len(rows) < page_size:
            break
    return out


def _flatten_match(row: dict[str, Any], tour: str) -> dict[str, Any] | None:
    """Project one vendor past-matches row into the parquet schema.

    Returns None when the row is missing a usable id, both player ids,
    a date, or a winner — those rows can't anchor a training example.
    Stats counters are kept as raw numerator/denominator pairs (vendor
    shape) so the feature builder owns the ratio computation in one
    place; the model file only contains COUNTS, never derived
    percentages.
    """
    mid = _coerce_int(row.get("id"))
    p1_id = _coerce_int(row.get("player1Id"))
    p2_id = _coerce_int(row.get("player2Id"))
    when = _parse_iso_date(row.get("date"))
    # `match_winner` is the WINNING PLAYER'S ID (probed 2026-05-06), NOT
    # a 1-or-2 side index. Map back to side relative to player1/player2.
    winner_id = _coerce_int(row.get("match_winner"))
    if mid is None or p1_id is None or p2_id is None or when is None:
        return None
    if winner_id == p1_id:
        winner_side = 1
    elif winner_id == p2_id:
        winner_side = 2
    else:
        # Walkover / abandonment / vendor inconsistency — no
        # labelled target.
        return None
    p1 = (row.get("player1") or {}) if isinstance(row.get("player1"), dict) else {}
    p2 = (row.get("player2") or {}) if isinstance(row.get("player2"), dict) else {}
    tournament = (row.get("tournament") or {}) if isinstance(row.get("tournament"), dict) else {}
    rnd = (row.get("round") or {}) if isinstance(row.get("round"), dict) else {}
    stats = row.get("stats") if isinstance(row.get("stats"), dict) else {}
    s1 = (stats.get("player1") or {}) if isinstance(stats, dict) else {}
    s2 = (stats.get("player2") or {}) if isinstance(stats, dict) else {}

    # Score string (e.g. "7-6(3) 6-4"). Persisted raw so the feature
    # builder can call `parse_score_details` and derive
    # tiebreak/decider/comeback/close-match aggregates point-in-time.
    # Older rows occasionally drop the field — the feature builder
    # treats None as "no clutch signal for this match" and skips it
    # without bumping any denominators.
    score_raw = row.get("result")
    score = (
        score_raw.strip()
        if isinstance(score_raw, str) and score_raw.strip()
        else None
    )
    out: dict[str, Any] = {
        "match_id": mid,
        "tour": tour,
        "match_date": when,
        "best_of": _coerce_int(row.get("best_of")),
        "court_id": _coerce_int(tournament.get("courtId")),
        "rank_id": _coerce_int(tournament.get("rankId")),
        "round_id": _coerce_int(rnd.get("id") or row.get("roundId")),
        "p1_id": p1_id,
        "p1_name": p1.get("name"),
        "p2_id": p2_id,
        "p2_name": p2.get("name"),
        "winner_side": winner_side,  # 1 if p1 won, 2 if p2 won
        "score": score,
    }
    # Per-side raw counters. Counts (aces, dfs, total points won) and
    # the numerator/denominator pairs that the feature builder
    # converts to percentages with point-in-time aggregation.
    for side, src in (("p1", s1), ("p2", s2)):
        out[f"{side}_first_serve"] = _coerce_int(src.get("firstServe"))
        out[f"{side}_first_serve_of"] = _coerce_int(src.get("firstServeOf"))
        out[f"{side}_won_first_serve"] = _coerce_int(src.get("winningOnFirstServe"))
        out[f"{side}_won_first_serve_of"] = _coerce_int(src.get("winningOnFirstServeOf"))
        out[f"{side}_won_second_serve"] = _coerce_int(src.get("winningOnSecondServe"))
        out[f"{side}_won_second_serve_of"] = _coerce_int(src.get("winningOnSecondServeOf"))
        out[f"{side}_bp_converted"] = _coerce_int(src.get("breakPointsConverted"))
        out[f"{side}_bp_converted_of"] = _coerce_int(src.get("breakPointsConvertedOf"))
        out[f"{side}_aces"] = _coerce_int(src.get("aces"))
        out[f"{side}_double_faults"] = _coerce_int(src.get("doubleFaults"))
        out[f"{side}_total_points_won"] = _coerce_int(src.get("totalPointsWon"))
    return out


async def _backfill_tour(
    client: httpx.AsyncClient,
    bucket: _TokenBucket,
    tour: str,
    top_n: int,
    pages: int,
    page_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (match_rows, profile_rows) for one tour. Profile rows
    are 1:1 with the top-N list; match rows are not yet deduped
    across players (caller does the cross-tour dedup).
    """
    log.info("backfill %s — fetching top %d", tour, top_n)
    pairs = await _fetch_top_ids(client, bucket, tour, top_n)
    log.info("backfill %s — got %d player ids", tour, len(pairs))

    profiles: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    for i, (pid, name) in enumerate(pairs, 1):
        # Profile (cheap; one call per player).
        prof = await _fetch_profile(client, bucket, tour, pid)
        info = (prof.get("information") or {}) if isinstance(prof, dict) else {}
        birth = _parse_iso_date(info.get("birthdate"))
        if birth is None:
            # MatchStat dropped `information.birthdate` from this
            # endpoint as of mid-2025 (verified by probing 2026-05-06).
            # Synthesise from `turnedPro` (year-only string) using the
            # ATP/WTA average turn-pro age of ~17. Coarse — actual
            # birth year drifts ±2y around this — but a noisy age
            # signal beats catboost's all-NaN read of the column.
            turned_pro = info.get("turnedPro")
            if isinstance(turned_pro, str) and turned_pro.isdigit():
                # Mid-year (Jul 1) so the years-elapsed math doesn't
                # systematically bias to a Jan 1 underestimate.
                birth = date(int(turned_pro) - 17, 7, 1)
        profiles.append({
            "tour": tour,
            "player_id": pid,
            "name": name,
            "name_normalized": _normalize_name(name),
            "birthdate": birth,
            "plays": info.get("plays"),
        })
        # Past matches (the bulk of the API budget).
        rows = await _fetch_past_matches(
            client, bucket, tour, pid, pages, page_size
        )
        for r in rows:
            flat = _flatten_match(r, tour)
            if flat is not None:
                matches.append(flat)
        if i % 10 == 0 or i == len(pairs):
            log.info("backfill %s — processed %d/%d players, %d match rows",
                     tour, i, len(pairs), len(matches))
    return matches, profiles


async def backfill(
    api_key: str,
    *,
    tours: Iterable[str] = ("atp", "wta"),
    top_n: int = 50,
    pages: int = 3,
    page_size: int = _DEFAULT_PAST_MATCH_PAGE_SIZE,
    timeout: float = 30.0,
) -> tuple[Path, Path]:
    """Run the backfill end-to-end. Returns (matches_path, profiles_path).

    Idempotent on output paths — overwrites both files. The vendor
    response is point-in-time so back-to-back runs on the same day
    give identical content (modulo the most recent matches).
    """
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    bucket = _TokenBucket(_REQUESTS_PER_SECOND, _BURST_TOKENS)
    all_matches: list[dict[str, Any]] = []
    all_profiles: list[dict[str, Any]] = []

    async with httpx.AsyncClient(headers=_headers(api_key), timeout=timeout) as client:
        for tour in tours:
            m, p = await _backfill_tour(
                client, bucket, tour, top_n=top_n, pages=pages, page_size=page_size
            )
            all_matches.extend(m)
            all_profiles.extend(p)

    if not all_matches:
        raise RuntimeError("backfill produced zero match rows — check API key / network")

    # Cross-player dedup. A match between two top-N players appears in
    # both feeds; we keep one row per `match_id`.
    matches_df = pd.DataFrame(all_matches)
    n_before = len(matches_df)
    matches_df = matches_df.drop_duplicates(subset=["match_id"], keep="first")
    matches_df = matches_df.sort_values("match_date").reset_index(drop=True)
    log.info(
        "deduped match rows: %d → %d (kept first occurrence)",
        n_before, len(matches_df),
    )

    profiles_df = pd.DataFrame(all_profiles)
    profiles_df = profiles_df.drop_duplicates(
        subset=["tour", "player_id"], keep="first"
    ).reset_index(drop=True)

    # Coerce date columns to a uniform Pyarrow-friendly dtype. pandas
    # writes mixed-type object columns as variable-length strings in
    # parquet, which the reader can't load back as dates.
    matches_df["match_date"] = pd.to_datetime(matches_df["match_date"])
    profiles_df["birthdate"] = pd.to_datetime(
        profiles_df["birthdate"], errors="coerce"
    )

    matches_df.to_parquet(RAW_MATCHES_PATH, index=False)
    profiles_df.to_parquet(PLAYER_PROFILES_PATH, index=False)
    log.info(
        "wrote %d matches → %s, %d profiles → %s",
        len(matches_df), RAW_MATCHES_PATH,
        len(profiles_df), PLAYER_PROFILES_PATH,
    )
    return RAW_MATCHES_PATH, PLAYER_PROFILES_PATH


def _resolve_api_key() -> str:
    """Same env contract as `tennis/provider.py:build_tennis_provider` —
    uses `TENNIS_STATS_API_KEY`. Raises with a hint when missing.
    """
    key = os.environ.get("TENNIS_STATS_API_KEY")
    if not key:
        raise RuntimeError(
            "TENNIS_STATS_API_KEY not set; backfill needs the live "
            "MatchStat key (the stub provider has no historical data)"
        )
    return key


async def run_backfill_cli(
    *, tours: list[str], top_n: int, pages: int, page_size: int
) -> tuple[Path, Path]:
    """CLI entry point. Resolves the API key from the env and dispatches
    to `backfill`. Kept thin so the underlying function stays usable
    from notebooks and tests with a passed-in key.
    """
    return await backfill(
        api_key=_resolve_api_key(),
        tours=tours,
        top_n=top_n,
        pages=pages,
        page_size=page_size,
    )


__all__ = [
    "PLAYER_PROFILES_PATH",
    "RAW_MATCHES_PATH",
    "backfill",
    "run_backfill_cli",
]
