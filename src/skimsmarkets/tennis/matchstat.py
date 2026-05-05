"""Concrete `TennisStatsProvider` for the MatchStat tennis API.

Vendor: <https://tennisapidoc.matchstat.com>. Hosted on RapidAPI under
`tennis-api-atp-wta-itf.p.rapidapi.com`. Auth via two static headers
(`X-RapidAPI-Key`, `X-RapidAPI-Host`) on every request. Rate limit is
100 req/min/IP — generous for our usage (≤ 1 rankings call + ~7 per-player
calls × a handful of tennis matches per slate).

Why each endpoint:
  - `/{tour}/ranking/singles`        — name → player_id index. The vendor's
                                       /search endpoint omits IDs (probed),
                                       so rankings is the only path. Top
                                       500 covers virtually all
                                       Polymarket-traded tour singles.
  - `/{tour}/player/profile/{id}`    — `form` array (recent W/L), bio
                                       (`information.birthdate`, `plays`),
                                       and career-high rank (`bestRank`).
                                       Profile does NOT carry points; we
                                       read those from the rankings index
                                       hit instead.
  - `/{tour}/player/surface-summary/{id}` — yearly per-court win/loss.
                                       Most recent year's row gives YTD
                                       totals + per-surface splits in one
                                       payload.
  - `/{tour}/player/past-matches/{id}` — last-N matches with opp / score /
                                       round / surface / tier. Used both
                                       for `last_match_date` and the
                                       3-row recent-match digest.
  - `/{tour}/player/perf-breakdown/{id}` — current-year W/L matrix.
                                       We pull top-5, top-10, slam,
                                       masters cells.
  - `/{tour}/player/match-stats/{id}` — career serve + return + BP stats.
                                       The `rtnStats` block is opponent's
                                       serve performance against this
                                       player, so we invert it for
                                       return-points-won %.
  - `/{tour}/player/titles/{id}`     — career titles per tier. Career
                                       achievement baseline distinct from
                                       YTD records (a 28yo with 0 slam
                                       titles + 15 mainTour is a
                                       different player from a 22yo with
                                       4 slams, regardless of rank).
  - `/{tour}/h2h/info/{a}/{b}`       — per-surface H2H counts. Preserved
                                       per-surface (used to be summed)
                                       so surface-conditioned matchups
                                       read correctly.
  - `/{tour}/h2h/matches/{a}/{b}`    — reverse-chronological meeting list.
                                       PageSize=3 surfaces matchup
                                       trajectory across recent meetings,
                                       not just the latest one.
  - `/{tour}/h2h/stats/{a}/{b}`      — matchup-conditioned aggregates.
                                       We pull decider/tiebreak,
                                       bo3/bo5 split, set-1 win/lose →
                                       match conversion, and matchup
                                       1st-serve-won + BP-convert.

Naming normalization: vendor names sometimes carry diacritics
(e.g. "Cóbolli") that Polymarket strips. We index on a lowercase +
diacritic-stripped form so common labelings cross-match without exact
casing.
"""

from __future__ import annotations

import asyncio
import logging
import random
import unicodedata
from collections.abc import Iterable
from datetime import UTC, date, datetime
from types import TracebackType
from typing import Any, Self

import httpx

from skimsmarkets.tennis.identity import TennisMatchIdentity
from skimsmarkets.tennis.models import (
    TennisH2HMeeting,
    TennisHeadToHead,
    TennisInMatchupStats,
    TennisPlayerStats,
    TennisRecentMatch,
    TennisStatsContext,
)

log = logging.getLogger(__name__)

_BASE_URL = "https://tennis-api-atp-wta-itf.p.rapidapi.com"
_HOST = "tennis-api-atp-wta-itf.p.rapidapi.com"

# How many ranking entries to pull per tour. Top 500 covers ATP's full
# ranked field comfortably (the cutoff for tour-level singles
# entry is well inside top 250) and gives buffer for WTA's longer ranked
# tail. The vendor caps page sizes at 100, so this becomes 5 paginated
# calls per tour at boot — still trivial against the 100 req/min ceiling.
_RANKING_PAGE_SIZE = 100
_RANKING_MAX_PAGES = 5  # 5 × 100 = top 500

# Vendor courtId → our surface key. Per probing 2026-04-23:
#   1 = Hard, 2 = Clay, 3 = I.hard (indoor hard), 5 = Grass.
# We collapse Hard + I.hard into "hard" so the prompt block stays compact
# and matches the "hard / clay / grass / carpet" surface vocabulary the
# tennis sport hint already uses.
_COURT_ID_TO_SURFACE: dict[int, str] = {
    1: "hard",
    2: "clay",
    3: "hard",
    4: "carpet",
    5: "grass",
}

# Vendor tournament rankId → tier label. Sourced from the /player/titles
# tourRankId enumeration (probed): 0 futures, 1 challenger, 2 main_tour,
# 3 masters, 4 grand_slam, 5 team_cup, 7 tour_finals. Only the upper
# tiers reach the prompt — Polymarket trades tour-level events almost
# exclusively, so a Challenger label on a recent match is a meaningful
# downgrade signal but Futures/team-cup labels are noise.
_RANK_ID_TO_TIER: dict[int, str] = {
    0: "futures",
    1: "challenger",
    2: "main_tour",
    3: "masters",
    4: "grand_slam",
    5: "team_cup",
    7: "tour_finals",
}

# Tier keys we actually surface in `career_titles`. The lower tiers
# (futures, challenger, team_cup) and Davis/Fed Cup are dropped — they
# are not load-bearing for tour-level Polymarket markets.
_TITLE_TIERS_KEEP: dict[int, str] = {
    2: "main_tour",
    3: "masters",
    4: "grand_slam",
    7: "tour_finals",
}

# `round` and `tournament` get added to `include=` on h2h/matches and
# past-matches calls so the vendor ships the joined `round.name`
# (e.g. "Final", "1/2", "1/4") and `tournament.courtId` /
# `tournament.rankId` inline. Centralised so the include string and the
# parser stay in sync.
_MATCH_INCLUDE = "tournament,round"

# Recent-match / recent-meeting page sizes. Kept conservative — the
# renderer caps at 3 rows for prompt-token reasons, but we pull a couple
# extra in case the vendor omits a row.
_RECENT_MATCH_PAGE_SIZE = 5
_RECENT_MEETING_PAGE_SIZE = 3

_RETRY_ATTEMPTS = 5
_RETRY_BASE_S = 1.0

# Per-provider HTTP rate limit. MatchStat's published limit is
# 5 requests per second; we enforce it client-side with a token bucket
# so we never depend on the retry-on-429 path under normal operation
# (the retry stays in place as a backstop for vendor-side hiccups).
# A semaphore alone can't enforce a per-second cap because throughput
# = concurrency / latency, and latency varies — at 100ms per call
# even concurrency=1 fires 10 req/sec. The token bucket decouples
# rate from latency. `_BURST_TOKENS` lets the first few calls fire
# back-to-back for snappy startup; subsequent calls drip out at the
# steady rate.
_REQUESTS_PER_SECOND = 5.0
_BURST_TOKENS = 5


def _normalize_name(name: str) -> str:
    """Lowercase, diacritic-stripped, single-spaced.

    Used for keying the rankings index. The vendor and Polymarket both
    ship names in roughly the same Latin form, but the vendor preserves
    diacritics (Cóbolli, Müller) while Polymarket question strings
    sometimes drop them. We normalize both sides so lookups don't miss.
    """
    nfkd = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(stripped.lower().split())


def _coerce_int(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        try:
            return int(v.strip())
        except ValueError:
            return None
    if isinstance(v, float):
        return int(v)
    return None


def _parse_iso_date(v: Any) -> date | None:
    """Vendor ships dates as `2026-04-20T00:00:00.000Z`. Trim to the
    date portion and parse — used for computing age from birthdate
    where the validator on the model isn't available.
    """
    if not isinstance(v, str) or not v:
        return None
    try:
        return date.fromisoformat(v[:10])
    except ValueError:
        return None


def _years_between(birth: date, today: date) -> int:
    """Whole-year age. Subtracts a year if the birthday hasn't landed
    yet this calendar year.
    """
    years = today.year - birth.year
    if (today.month, today.day) < (birth.month, birth.day):
        years -= 1
    return max(years, 0)


def _surface_from_court_id(cid: Any) -> str | None:
    n = _coerce_int(cid)
    if n is None:
        return None
    return _COURT_ID_TO_SURFACE.get(n)


def _tier_from_rank_id(rid: Any) -> str | None:
    n = _coerce_int(rid)
    if n is None:
        return None
    return _RANK_ID_TO_TIER.get(n)


class _TokenBucket:
    """Async token-bucket rate limiter for steady-rate HTTP throttling.

    Semaphores cap *concurrency*; this caps *rate*. The distinction
    matters when latency is short or unpredictable — a 100 req/min
    vendor with a Semaphore(2) and 50 ms response time would still
    fire ~40 req/sec, blowing the limit. The token bucket decouples
    rate from latency by handing out one token per `1 / rate` seconds
    regardless of how long each call takes.

    `capacity` is the burst budget — calls that arrive when the bucket
    is full fire immediately; the bucket then refills at `rate`
    tokens per second. A capacity equal to the rate gives roughly
    "one second's worth of burst" which is fine for startup snappiness
    without overshooting the steady-state limit for long.

    Single-process semantics; not durable across runs. Sufficient for
    our use case (one provider per pipeline run).
    """

    def __init__(self, rate_per_second: float, capacity: float) -> None:
        self._rate = rate_per_second
        self._capacity = capacity
        self._tokens = capacity
        self._last_refill: float | None = None
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until one token is available, then consume it.

        Lock-guarded so concurrent callers compute consistent token
        counts. Refills lazily on each acquire — no background task,
        no clock-tick overhead.
        """
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            if self._last_refill is None:
                self._last_refill = now
            else:
                elapsed = now - self._last_refill
                self._tokens = min(
                    self._capacity, self._tokens + elapsed * self._rate
                )
                self._last_refill = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            # Bucket empty: sleep until exactly one token will be
            # available, then consume it. Holding the lock during the
            # sleep serialises waiters in arrival order, which is the
            # behaviour we want for fairness.
            wait = (1.0 - self._tokens) / self._rate
            await asyncio.sleep(wait)
            self._tokens = 0.0


class MatchStatTennisProvider:
    """Async-context-managed adapter for the MatchStat tennis API.

    Lifecycle:
      `async with MatchStatTennisProvider(api_key) as p:`
          ... `await p.fetch(identity)` per match ...

    The first `fetch` triggers a one-shot rankings warmup (lazy, behind a
    lock so concurrent fetches don't trigger duplicate paginations).
    Subsequent fetches reuse the in-memory index for zero extra cost.
    """

    name = "matchstat"

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 20.0,
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        # Rankings index: tour → normalized-name → (player_id, position, points).
        self._index: dict[str, dict[str, tuple[int, int | None, int | None]]] = {}
        self._index_locks: dict[str, asyncio.Lock] = {
            "atp": asyncio.Lock(),
            "wta": asyncio.Lock(),
        }
        # Per-provider HTTP rate limiter. Every `_get` call acquires
        # one token before issuing the request; the bucket refills at
        # the vendor's published 5 req/sec limit. Bursts from
        # `asyncio.gather` callers (warm_form_for_selection,
        # _player_stats fan-out, H2H parallel block) get serialised
        # automatically without saturating the limit. Single shared
        # budget spans the whole provider lifetime.
        self._rate_limiter = _TokenBucket(
            rate_per_second=_REQUESTS_PER_SECOND,
            capacity=_BURST_TOKENS,
        )
        # Profile cache keyed by (tour, player_id) → the profile tuple.
        # Populated on the first `_player_profile` call for each player and
        # reused for the rest of the provider's lifetime. Two callers hit
        # this cache: (1) the selection-stage warmup
        # (`warm_form_for_selection`) pre-fetches form arrays for every
        # tennis identity in the raw slate; (2) the full enrichment path's
        # `_player_stats` re-uses those entries for players that survived
        # the cap, eliminating the duplicate /player/profile call. Keyed
        # by `(tour, pid)` — IDs are tour-scoped per the MatchStat URL
        # shape `/tennis/v2/{tour}/player/profile/{id}`, so collisions
        # across tours are theoretically possible.
        self._profile_cache: dict[
            tuple[str, int],
            tuple[list[str], int | None, int | None, str | None],
        ] = {}

    async def __aenter__(self) -> Self:
        headers = {
            "Accept": "application/json",
            "X-RapidAPI-Key": self._api_key,
            "X-RapidAPI-Host": _HOST,
        }
        self._client = httpx.AsyncClient(timeout=self._timeout, headers=headers)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ----- HTTP plumbing -----

    async def _get(
        self, path: str, params: dict[str, str | int] | None = None
    ) -> Any | None:
        """GET with 429-aware retry. Returns parsed JSON or None on failure.

        Mirrors the posture in `unusual_whales/client.py`: any failure
        (network, non-2xx, malformed JSON) returns None and lets the caller
        degrade gracefully — never raises through to abort the pipeline.

        Rate limit: every call acquires one token from
        `self._rate_limiter` before issuing the request. The bucket
        refills at the vendor's 5 req/sec ceiling so we never burst
        past it under normal operation. Retries on 429 use
        exponential backoff with jitter as a backstop for vendor-side
        hiccups (transient overload, IP-shared throttling) — without
        jitter, all N concurrent retries would wake at the same
        moment and re-429 immediately.
        """
        if self._client is None:
            raise RuntimeError(
                "MatchStatTennisProvider used outside of `async with` context"
            )
        url = f"{_BASE_URL}{path}"
        for attempt in range(_RETRY_ATTEMPTS):
            # Acquire one token per attempt — retries pay the rate
            # limit too, otherwise a 429 burst would re-saturate the
            # bucket on the way back up.
            await self._rate_limiter.acquire()
            try:
                resp = await self._client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status == 404:
                    log.debug("matchstat %s: 404", path)
                    return None
                if status == 429 and attempt + 1 < _RETRY_ATTEMPTS:
                    retry_after = e.response.headers.get("Retry-After")
                    # Exponential backoff with jitter: ±50% of base.
                    # Without jitter, gather()'d callers all get
                    # 429'd at once, all back off `2^n`, all retry
                    # at the same instant and 429 again. Jitter
                    # spreads the wakeups across a window so the
                    # rate-limit window drains.
                    try:
                        backoff = (
                            float(retry_after)
                            if retry_after
                            else _RETRY_BASE_S * (2 ** attempt)
                        )
                    except ValueError:
                        backoff = _RETRY_BASE_S * (2 ** attempt)
                    wait = backoff * random.uniform(0.5, 1.5)
                    log.debug(
                        "matchstat %s: 429 sleeping %.1fs (attempt %d/%d)",
                        path, wait, attempt + 1, _RETRY_ATTEMPTS,
                    )
                    await asyncio.sleep(wait)
                    continue
                log.warning("matchstat %s: HTTP %s", path, status)
                return None
            except Exception as e:  # noqa: BLE001
                log.warning("matchstat %s: %s", path, type(e).__name__)
                return None
        return None

    # ----- Rankings index (name → id) -----

    async def _ensure_index(self, tour: str) -> None:
        """Populate `self._index[tour]` once per process.

        Lock-guarded so concurrent fetches don't issue duplicate
        paginated requests. Failed page fetches just leave that page out
        of the index — partial coverage is better than no coverage when
        rate limits or intermittent vendor issues hit.
        """
        if tour in self._index:
            return
        async with self._index_locks[tour]:
            if tour in self._index:
                return
            mapping: dict[str, tuple[int, int | None, int | None]] = {}
            for page in range(1, _RANKING_MAX_PAGES + 1):
                body = await self._get(
                    f"/tennis/v2/{tour}/ranking/singles",
                    params={"pageSize": _RANKING_PAGE_SIZE, "pageNo": page},
                )
                if body is None:
                    break
                rows = body.get("data") if isinstance(body, dict) else None
                if not isinstance(rows, list) or not rows:
                    break
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    player = row.get("player")
                    if not isinstance(player, dict):
                        continue
                    pid = _coerce_int(player.get("id"))
                    name = player.get("name")
                    if pid is None or not isinstance(name, str) or not name:
                        continue
                    key = _normalize_name(name)
                    if key in mapping:
                        # First (highest-ranked) hit wins on duplicate
                        # normalized names — exotic but possible (two
                        # players with diacritics that strip to the same
                        # form). Highest rank is the more relevant one for
                        # Polymarket which only lists tour singles.
                        continue
                    mapping[key] = (
                        pid,
                        _coerce_int(row.get("position")),
                        _coerce_int(row.get("point")),
                    )
                if len(rows) < _RANKING_PAGE_SIZE:
                    break  # last page reached early
            self._index[tour] = mapping
            log.info(
                "matchstat: indexed %d %s players from rankings",
                len(mapping), tour,
            )

    def _resolve(
        self, tour: str, name: str
    ) -> tuple[int, int | None, int | None] | None:
        idx = self._index.get(tour, {})
        return idx.get(_normalize_name(name))

    # ----- Selection-stage helpers (called pre-cap, no HTTP per event) -----

    async def warm_for_selection(self, tours: Iterable[str]) -> None:
        """Warm the rankings index for one or more tours in parallel.

        Idempotent: `_ensure_index` early-returns when the tour is
        already cached. Pagination cost is one-time per tour
        (5 calls × 100 entries = top 500), shared across the entire
        process lifetime — every tennis event in the slate uses the
        warm index for free after that.
        """
        unique_tours = {t for t in tours if t in self._index_locks}
        if not unique_tours:
            return
        await asyncio.gather(*(self._ensure_index(t) for t in unique_tours))

    def lookup_player_rank(
        self, tour: str, name: str
    ) -> tuple[int, int] | None:
        """Sync `(rank_position, rank_points)` lookup against the warm index.

        Returns None when:
          - the index isn't warm for this tour
          - the player isn't in the top-N covered by the index
          - the matched record is missing position OR points
        Callers (selection scoring) treat None as "no rank signal"
        and degrade to other imbalance signals.
        """
        hit = self._resolve(tour, name)
        if hit is None:
            return None
        _pid, position, points = hit
        if position is None or points is None:
            return None
        return position, points

    async def warm_form_for_selection(
        self, identities: Iterable["TennisMatchIdentity"]
    ) -> None:
        """Pre-fetch `/player/profile` for every player across the given
        identities and populate `self._profile_cache`.

        Lets selection-stage scoring read form arrays via
        `lookup_player_form` without triggering an HTTP per call. Both
        sides of every identity are fanned out in parallel; players
        already in the cache (e.g. someone who appears in two slate
        events) skip re-fetching naturally via the cache check inside
        `_player_profile`.

        Pre-condition: the rankings index must already be warm for the
        tours represented in `identities`. Selection orchestrator calls
        `warm_for_selection(tours)` first, so this dependency holds in
        practice. Identities that fail rank resolution (player outside
        top-500) are silently skipped — the score cascade in
        `_tennis_imbalance` already requires both ranks to compute
        anything useful, so there's no point pre-fetching profiles for
        unranked players.

        Failure mode: any single `_player_profile` call that errors
        out lands an empty tuple in the cache and lets the form
        adjustment skip that event. We don't propagate exceptions so
        a partial vendor outage degrades gracefully.
        """
        # Collect the unique (tour, pid) pairs we need profiles for.
        # `_resolve` is sync and cheap (dict lookup); doing the
        # collection up front lets us issue exactly one fetch per
        # unique player even when an identity list contains repeats.
        pending: set[tuple[str, int]] = set()
        for ident in identities:
            for nm in (ident.player_a, ident.player_b):
                hit = self._resolve(ident.tour, nm)
                if hit is None:
                    continue
                pid = hit[0]
                if (ident.tour, pid) in self._profile_cache:
                    continue
                pending.add((ident.tour, pid))
        if not pending:
            return
        log.info(
            "matchstat: warming profile cache for %d players (selection)",
            len(pending),
        )
        await asyncio.gather(
            *(self._player_profile(tour, pid) for tour, pid in pending)
        )

    def lookup_player_form(
        self, tour: str, name: str
    ) -> tuple[str, int | None] | None:
        """Sync `(last_10_form_string, best_rank)` lookup from the cache.

        Returns None when:
          - the player isn't in the rankings index for this tour
          - `warm_form_for_selection` hasn't been called yet (cache miss)
          - the cached profile has an empty form array (vendor returned
            no recent matches for this player)
        Callers (selection scoring) treat None as "no form signal" and
        skip the form-alignment adjustment, falling back to the
        points-ratio base score.

        Form string is uppercased and capped at the last 10 entries
        for consistency with the renderer's prompt-time form string —
        same `"WWLWWLWWWL"` shape both layers consume.
        """
        hit = self._resolve(tour, name)
        if hit is None:
            return None
        pid = hit[0]
        cached = self._profile_cache.get((tour, pid))
        if cached is None:
            return None
        form_arr, best_rank, _age, _plays = cached
        if not form_arr:
            return None
        tail = form_arr[-10:]
        form_str = "".join(c.upper() for c in tail if c in ("w", "l"))
        if not form_str:
            return None
        return form_str, best_rank

    # ----- Per-player fetches -----

    async def _player_profile(
        self, tour: str, pid: int
    ) -> tuple[list[str], int | None, int | None, str | None]:
        """Returns (form_array, best_rank, age_years, plays).

        Cache-then-fetch: results land in `self._profile_cache` keyed by
        `(tour, pid)`. The selection-stage form warmup
        (`warm_form_for_selection`) and the per-event enrichment path
        (`_player_stats`) both call this method, so the cache turns
        what would be 2× duplicate HTTP per surviving event into one.
        Cache is provider-lifetime — a fresh `async with` block (one
        per pipeline run) gets a fresh cache, which is desirable: form
        data should not survive across runs.

        Single profile call with `include=form,ranking,information`
        covers four needs: the recent W/L array, career-high ranking
        (`bestRank.position`), birthdate (under `information.birthdate`,
        used to compute current age), and `information.plays`
        (handedness + backhand style). All free on the same response;
        no extra HTTP. The vendor ships `form` oldest → newest; we pass
        it through unchanged.
        """
        cached = self._profile_cache.get((tour, pid))
        if cached is not None:
            return cached
        body = await self._get(
            f"/tennis/v2/{tour}/player/profile/{pid}",
            params={"include": "form,ranking,information"},
        )
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, dict):
            empty: tuple[list[str], int | None, int | None, str | None] = (
                [], None, None, None,
            )
            self._profile_cache[(tour, pid)] = empty
            return empty
        form = data.get("form") or []
        if not isinstance(form, list):
            form = []
        best_rank = None
        best_block = data.get("bestRank")
        if isinstance(best_block, dict):
            best_rank = _coerce_int(best_block.get("position"))
        age_years: int | None = None
        plays: str | None = None
        info = data.get("information")
        if isinstance(info, dict):
            birth = _parse_iso_date(info.get("birthdate"))
            if birth is not None:
                age_years = _years_between(birth, datetime.now(UTC).date())
            raw_plays = info.get("plays")
            if isinstance(raw_plays, str) and raw_plays.strip():
                plays = raw_plays.strip()
        result: tuple[list[str], int | None, int | None, str | None] = (
            [str(x) for x in form if isinstance(x, str)],
            best_rank,
            age_years,
            plays,
        )
        self._profile_cache[(tour, pid)] = result
        return result

    async def _player_recent_matches(
        self, tour: str, pid: int, name: str
    ) -> tuple[date | None, list[TennisRecentMatch]]:
        """Pull the last N matches; return `(last_match_date, recent_matches)`.

        Single call replaces what used to be a pageSize=1 "just the date"
        lookup. The first row's date doubles as `last_match_date`; the
        rest of the rows feed `TennisPlayerStats.recent_matches` with
        opp / score / round / surface / tier so the prompt can show
        recent quality instead of just a W/L pattern.

        `pid` and `name` are the subject player; we pick opponent vs
        subject by comparing `player1Id`/`player2Id` against `pid`,
        which is robust to whichever side the vendor lists the subject
        on.
        """
        body = await self._get(
            f"/tennis/v2/{tour}/player/past-matches/{pid}",
            params={"pageSize": _RECENT_MATCH_PAGE_SIZE, "include": _MATCH_INCLUDE},
        )
        rows = body.get("data") if isinstance(body, dict) else None
        if not isinstance(rows, list) or not rows:
            return None, []
        last_date: date | None = None
        out: list[TennisRecentMatch] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_date = _parse_iso_date(row.get("date"))
            if last_date is None and row_date is not None:
                last_date = row_date
            p1 = row.get("player1") if isinstance(row.get("player1"), dict) else {}
            p2 = row.get("player2") if isinstance(row.get("player2"), dict) else {}
            p1_id = _coerce_int(row.get("player1Id"))
            is_subject_p1 = p1_id == pid
            opp_block = p2 if is_subject_p1 else p1
            opp_name = opp_block.get("name") if isinstance(opp_block, dict) else None
            if not isinstance(opp_name, str) or not opp_name:
                # Skip rows we can't attribute — pretty much never
                # happens on the live vendor but guard anyway.
                continue
            winner_id = _coerce_int(row.get("match_winner"))
            won = winner_id == pid
            tourn = row.get("tournament") if isinstance(row.get("tournament"), dict) else {}
            surface = _surface_from_court_id(tourn.get("courtId"))
            tier = _tier_from_rank_id(tourn.get("rankId"))
            tname = tourn.get("name")
            tournament_name = tname.strip() if isinstance(tname, str) and tname.strip() else None
            rnd = row.get("round") if isinstance(row.get("round"), dict) else {}
            rname = rnd.get("name")
            round_name = rname.strip() if isinstance(rname, str) and rname.strip() else None
            result = row.get("result")
            score = result.strip() if isinstance(result, str) and result.strip() else None
            out.append(
                TennisRecentMatch(
                    date=row_date,
                    opponent_name=opp_name.strip(),
                    won=won,
                    result=score,
                    surface=surface,
                    round=round_name,
                    tournament_name=tournament_name,
                    tournament_tier=tier,
                )
            )
        # `name` is unused at parse time but kept in the signature so the
        # caller can pass it explicitly — it makes the call site read as
        # "fetch recent matches for this player by name" and avoids a
        # silent reordering bug if the IDs ever flip.
        del name
        return last_date, out

    async def _player_tier_records(
        self, tour: str, pid: int
    ) -> dict[str, tuple[int, int] | None]:
        """Pull current-year W/L vs top-5/top-10 + at Slams + at Masters.

        Perf-breakdown ships a year-keyed dict whose value is a 4-axis
        matrix (`court`, `round`, `rank`, `level`). We deliberately
        consume only four cells of one year's slice — the full payload
        is enormous and most cells overlap signal we already have via
        surface-summary or h2h. Year selection: the largest numeric key
        present (vendor sorts unspecified, max() makes us robust to
        order changes). Cells use `aw`/`al` (all wins / all losses) per
        the vendor's convention; the bare `w`/`l` columns track only
        finals. `top5` is added alongside `top10` because the gap
        between "elite-tier slayer" and "merely beats top-10s" is
        material and the cell is on the same response.
        """
        body = await self._get(f"/tennis/v2/{tour}/player/perf-breakdown/{pid}")
        data = body.get("data") if isinstance(body, dict) else None
        out: dict[str, tuple[int, int] | None] = {
            "record_vs_top_5": None,
            "record_vs_top_10": None,
            "record_at_grand_slam": None,
            "record_at_masters": None,
        }
        if not isinstance(data, dict) or not data:
            return out

        # Year keys arrive as strings ("2026", "2025", ...). Pick the
        # largest by numeric value; defensively skip non-numeric keys.
        def _ykey(k: Any) -> int:
            return _coerce_int(k) or 0

        latest_year = max(data.keys(), key=_ykey, default=None)
        if latest_year is None:
            return out
        year_block = data.get(latest_year)
        if not isinstance(year_block, dict):
            return out

        def _cell(parent_key: str, child_key: str) -> tuple[int, int] | None:
            parent = year_block.get(parent_key)
            if not isinstance(parent, dict):
                return None
            child = parent.get(child_key)
            if not isinstance(child, dict):
                return None
            wins = _coerce_int(child.get("aw"))
            losses = _coerce_int(child.get("al"))
            if wins is None and losses is None:
                return None
            # Vendor returns 0 / 0 for cells the player hasn't appeared
            # in this year — suppress those rather than render "0-0",
            # which reads like a real but empty record.
            if (wins or 0) == 0 and (losses or 0) == 0:
                return None
            return (wins or 0, losses or 0)

        out["record_vs_top_5"] = _cell("rank", "top5")
        out["record_vs_top_10"] = _cell("rank", "top10")
        out["record_at_grand_slam"] = _cell("level", "grandSlam")
        out["record_at_masters"] = _cell("level", "masters")
        return out

    async def _player_match_stats(
        self, tour: str, pid: int
    ) -> dict[str, float | None]:
        """Career serve / return / break-point percentages.

        The vendor ships raw counters (numerator + denominator) under
        `serviceStats`, `rtnStats`, `breakPointsServeStats`, and
        `breakPointsRtnStats`. We compute the ratios here so the prompt
        block carries percentages directly — the reasoner shouldn't have
        to do arithmetic on raw counts in-context. Field naming:
        `<x>Gm` is the numerator (count of events meeting condition),
        `<x>OfGm` is the denominator (eligible events).

        `rtnStats` is the awkward one: the vendor stores it as opponent's
        serve performance against this player (same field shape as
        `serviceStats`, just from the other side of the net). So
        return-points-won % is `1 − (rtnStats.winningOnFirstServe /
        rtnStats.winningOnFirstServeOf)`. Doing the inversion here means
        the model field is the canonical "return-points-won" reading
        without further math by the renderer or reasoner.

        Returns a dict so the caller can spread the values directly into
        `TennisPlayerStats(...)` without N positional args. Keys mirror
        the model field names exactly.
        """
        body = await self._get(f"/tennis/v2/{tour}/player/match-stats/{pid}")
        data = body.get("data") if isinstance(body, dict) else None
        out: dict[str, float | None] = {
            "first_serve_in_pct": None,
            "first_serve_win_pct": None,
            "second_serve_win_pct": None,
            "first_serve_return_win_pct": None,
            "second_serve_return_win_pct": None,
            "break_point_save_pct": None,
            "break_point_convert_pct": None,
        }
        if not isinstance(data, dict):
            return out

        def _ratio(num: Any, den: Any) -> float | None:
            n = _coerce_int(num)
            d = _coerce_int(den)
            if n is None or d is None or d <= 0:
                return None
            return n / d

        srv = data.get("serviceStats") if isinstance(data.get("serviceStats"), dict) else {}
        out["first_serve_in_pct"] = _ratio(
            srv.get("firstServeGm"), srv.get("firstServeOfGm")
        )
        out["first_serve_win_pct"] = _ratio(
            srv.get("winningOnFirstServeGm"), srv.get("winningOnFirstServeOfGm")
        )
        out["second_serve_win_pct"] = _ratio(
            srv.get("winningOnSecondServeGm"), srv.get("winningOnSecondServeOfGm")
        )

        # Return side: `rtnStats` is "opponent's serve performance against
        # this player." Invert to get this player's return-points-won %.
        rtn = data.get("rtnStats") if isinstance(data.get("rtnStats"), dict) else {}
        opp_first_held = _ratio(
            rtn.get("winningOnFirstServeGm"), rtn.get("winningOnFirstServeOfGm")
        )
        if opp_first_held is not None:
            out["first_serve_return_win_pct"] = 1.0 - opp_first_held
        opp_second_held = _ratio(
            rtn.get("winningOnSecondServeGm"), rtn.get("winningOnSecondServeOfGm")
        )
        if opp_second_held is not None:
            out["second_serve_return_win_pct"] = 1.0 - opp_second_held

        bp_srv = data.get("breakPointsServeStats") if isinstance(data.get("breakPointsServeStats"), dict) else {}
        out["break_point_save_pct"] = _ratio(
            bp_srv.get("breakPointSavedGm"), bp_srv.get("breakPointFacedGm")
        )

        bp_rtn = data.get("breakPointsRtnStats") if isinstance(data.get("breakPointsRtnStats"), dict) else {}
        out["break_point_convert_pct"] = _ratio(
            bp_rtn.get("breakPointWonGm"), bp_rtn.get("breakPointChanceGm")
        )
        return out

    async def _player_career_titles(
        self, tour: str, pid: int
    ) -> dict[str, int] | None:
        """Career titles per tier from `/player/titles`.

        Distinct signal from YTD `record_at_grand_slam` etc.: those are
        current-year W/L; this is "career titles ever won." A 28yo with
        0 slam titles + 15 main-tour titles reads very differently from
        a 22yo with 4 slam titles. One extra HTTP call per player.

        Lower tiers (futures, challenger, team_cup) are dropped at parse
        time — Polymarket markets are tour-level and lower-tier title
        counts are noise that competes with prompt budget.
        """
        body = await self._get(f"/tennis/v2/{tour}/player/titles/{pid}")
        rows = body.get("data") if isinstance(body, dict) else None
        if not isinstance(rows, list):
            return None
        out: dict[str, int] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            tier_id = _coerce_int(row.get("tourRankId"))
            if tier_id is None:
                continue
            tier_key = _TITLE_TIERS_KEEP.get(tier_id)
            if tier_key is None:
                continue
            won = _coerce_int(row.get("titlesWon"))
            if won is None or won <= 0:
                continue
            out[tier_key] = won
        return out or None

    async def _player_surface_year_record(
        self, tour: str, pid: int
    ) -> tuple[
        tuple[int, int] | None,
        dict[str, tuple[int, int]] | None,
    ]:
        """Aggregate the most recent year's surface-summary into
        `(ytd_total, surface_dict)`.

        Vendor returns one row per year, each with a list of per-court
        win/loss counts. We take the FIRST year (vendor sorts
        descending; we double-check with `max(year)` as a fallback).
        Hard + I.hard collapse into a single "hard" entry to match the
        sport-hint vocabulary.
        """
        body = await self._get(f"/tennis/v2/{tour}/player/surface-summary/{pid}")
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or not data:
            return None, None

        def _year_of(row: Any) -> int:
            return _coerce_int(row.get("year") if isinstance(row, dict) else None) or 0

        latest = max((r for r in data if isinstance(r, dict)), key=_year_of, default=None)
        if not isinstance(latest, dict):
            return None, None
        surfaces = latest.get("surfaces")
        if not isinstance(surfaces, list):
            return None, None
        merged: dict[str, tuple[int, int]] = {}
        ytd_w = 0
        ytd_l = 0
        for s in surfaces:
            if not isinstance(s, dict):
                continue
            cid = _coerce_int(s.get("courtId"))
            wins = _coerce_int(s.get("courtWins")) or 0
            losses = _coerce_int(s.get("courtLosses")) or 0
            ytd_w += wins
            ytd_l += losses
            key = _COURT_ID_TO_SURFACE.get(cid) if cid is not None else None
            if key is None:
                continue
            prev_w, prev_l = merged.get(key, (0, 0))
            merged[key] = (prev_w + wins, prev_l + losses)
        return (ytd_w, ytd_l), merged or None

    async def _player_stats(
        self,
        tour: str,
        name: str,
        ranking_hit: tuple[int, int | None, int | None] | None,
    ) -> TennisPlayerStats:
        """Build a `TennisPlayerStats` for one player.

        When the rankings index has no hit, we still return a stats
        object carrying just the echoed name — `has_actionable_signal`
        will still trigger as long as the OTHER player or the H2H block
        has data. Better than dropping the whole context for one
        unranked player.
        """
        if ranking_hit is None:
            return TennisPlayerStats(name=name)
        pid, position, points = ranking_hit

        # All six per-player endpoints are independent — fan them out so
        # each player's full block lands in one round-trip's worth of
        # wall time rather than six sequential ones. The outer `fetch`
        # ALSO gathers across players, so two-player wall time is bounded
        # by the slowest single call. Total cost per match: 6 (this
        # player) + 6 (other player) + 3 (h2h info+matches+stats) = 15
        # calls. With the 100 req/min ceiling that's ~9s of vendor budget
        # per match, fully parallel → ~1s wall clock.
        (
            (form_arr, best_rank, age_years, plays),
            (ytd_pair, surfaces),
            match_stats,
            tier_records,
            (last_match_date_, recent_matches),
            career_titles,
        ) = await asyncio.gather(
            self._player_profile(tour, pid),
            self._player_surface_year_record(tour, pid),
            self._player_match_stats(tour, pid),
            self._player_tier_records(tour, pid),
            self._player_recent_matches(tour, pid, name),
            self._player_career_titles(tour, pid),
        )

        last_10_form: str | None = None
        if form_arr:
            # Vendor ships oldest → newest already; uppercase + cap to
            # the most recent 10 entries for a uniform width across
            # players regardless of how many matches they've played.
            tail = form_arr[-10:]
            last_10_form = "".join(c.upper() for c in tail if c in ("w", "l"))

        return TennisPlayerStats(
            name=name,
            api_player_id=str(pid),
            rank_singles=position,
            rank_points=points,
            best_rank_singles=best_rank,
            age_years=age_years,
            plays=plays,
            ytd_win_loss=ytd_pair,
            surface_win_loss=surfaces,
            last_10_form=last_10_form or None,
            recent_matches=recent_matches or None,
            last_match_date=last_match_date_,
            career_titles=career_titles,
            **match_stats,
            **tier_records,
        )

    # ----- H2H -----

    async def _head_to_head(
        self,
        tour: str,
        a_id: int,
        b_id: int,
        name_a: str,
        name_b: str,
    ) -> TennisHeadToHead | None:
        """Fetch overall H2H counts + recent meetings + matchup-conditioned stats.

        Three concurrent calls:
          /h2h/info — per-surface counts. Preserved per-surface (used to
            be summed) so surface-conditioned matchups read correctly.
          /h2h/matches — pageSize=3 reverse-chronological. Replaces the
            previous "last meeting only" with the matchup arc.
          /h2h/stats — matchup-conditioned aggregates (decider, tiebreak,
            bo3/bo5, set-1 win/lose conversions, in-matchup serve + BP).

        Empty H2H (no prior meetings) returns None and the renderer
        suppresses the section.
        """
        info_body, matches_body, stats_body = await asyncio.gather(
            self._get(f"/tennis/v2/{tour}/h2h/info/{a_id}/{b_id}"),
            self._get(
                f"/tennis/v2/{tour}/h2h/matches/{a_id}/{b_id}",
                params={
                    "pageSize": _RECENT_MEETING_PAGE_SIZE,
                    "include": _MATCH_INCLUDE,
                },
            ),
            self._get(f"/tennis/v2/{tour}/h2h/stats/{a_id}/{b_id}"),
        )

        # ----- /h2h/info: per-surface AND total counts -----
        a_wins_total = 0
        b_wins_total = 0
        surface_h2h: dict[str, tuple[int, int]] = {}
        info_rows = info_body.get("data") if isinstance(info_body, dict) else None
        if isinstance(info_rows, list):
            for row in info_rows:
                if not isinstance(row, dict):
                    continue
                a_w = _coerce_int(row.get("player1wins")) or 0
                b_w = _coerce_int(row.get("player2wins")) or 0
                a_wins_total += a_w
                b_wins_total += b_w
                surface = _surface_from_court_id(row.get("courtId"))
                if surface is None:
                    continue
                if a_w == 0 and b_w == 0:
                    # Suppress empty surface rows so the prompt doesn't
                    # render "grass=0-0" alongside meaningful entries.
                    continue
                prev_a, prev_b = surface_h2h.get(surface, (0, 0))
                surface_h2h[surface] = (prev_a + a_w, prev_b + b_w)

        # ----- /h2h/matches: list of recent meetings (newest first) -----
        recent_meetings: list[TennisH2HMeeting] = []
        match_rows = matches_body.get("data") if isinstance(matches_body, dict) else None
        if isinstance(match_rows, list):
            for row in match_rows[:_RECENT_MEETING_PAGE_SIZE]:
                if not isinstance(row, dict):
                    continue
                row_date = _parse_iso_date(row.get("date"))
                winner_id = _coerce_int(row.get("match_winner"))
                if winner_id == a_id:
                    winner_name = name_a
                elif winner_id == b_id:
                    winner_name = name_b
                else:
                    winner_name = None
                tourn = row.get("tournament") if isinstance(row.get("tournament"), dict) else {}
                surface = _surface_from_court_id(tourn.get("courtId"))
                tier = _tier_from_rank_id(tourn.get("rankId"))
                tname = tourn.get("name")
                tournament_name = tname.strip() if isinstance(tname, str) and tname.strip() else None
                rnd = row.get("round") if isinstance(row.get("round"), dict) else {}
                rname = rnd.get("name")
                round_name = rname.strip() if isinstance(rname, str) and rname.strip() else None
                result = row.get("result")
                score = result.strip() if isinstance(result, str) and result.strip() else None
                recent_meetings.append(
                    TennisH2HMeeting(
                        date=row_date,
                        winner_name=winner_name,
                        surface=surface,
                        round=round_name,
                        result=score,
                        tournament_name=tournament_name,
                        tournament_tier=tier,
                    )
                )

        # ----- /h2h/stats: matchup-conditioned per-player aggregates -----
        a_in_matchup = self._build_matchup_stats(stats_body, "player1Stats")
        b_in_matchup = self._build_matchup_stats(stats_body, "player2Stats")

        # Suppress only when literally nothing was found — a populated
        # stats block alone is enough signal even without h2h/info.
        if (
            a_wins_total == 0 and b_wins_total == 0
            and not recent_meetings
            and a_in_matchup is None and b_in_matchup is None
        ):
            return None

        return TennisHeadToHead(
            a_wins=a_wins_total,
            b_wins=b_wins_total,
            surface_h2h=surface_h2h or None,
            recent_meetings=recent_meetings or None,
            a_in_matchup=a_in_matchup,
            b_in_matchup=b_in_matchup,
        )

    @staticmethod
    def _build_matchup_stats(
        stats_body: Any, player_block_key: str
    ) -> TennisInMatchupStats | None:
        """Pull one player's matchup-conditioned aggregates from /h2h/stats.

        The vendor's `data.player1Stats` corresponds to player_a (the IDs
        in the URL path are positional). We extract:
          - decider/tiebreak (wins, total) — sample-size visible for the
            reasoner.
          - bo3/bo5 (wins, total) — format-conditioned record. Slams =
            bo5; everything else = bo3. The split tells the reasoner
            "this matchup is lopsided AT slams specifically" vs lopsided
            overall.
          - first-set-won → match-win pct AND first-set-lost → match-win
            pct. Together they characterise how this player handles set
            1 outcomes against this specific opponent.
          - first-serve-won pct + BP-convert pct, IN matchup. Distinct
            from the same fields on `TennisPlayerStats` (career across
            all opponents).

        Returns None when the response is missing or has no usable cells.
        """
        if not isinstance(stats_body, dict):
            return None
        data = stats_body.get("data")
        if not isinstance(data, dict):
            return None
        block = data.get(player_block_key)
        if not isinstance(block, dict):
            return None

        def _wins_total(win_key: str, total_key: str) -> tuple[int, int] | None:
            wins = _coerce_int(block.get(win_key))
            total = _coerce_int(block.get(total_key))
            if wins is None or total is None or total <= 0:
                return None
            return (wins, total)

        def _pct(key: str) -> float | None:
            # Vendor ships percentages as integers 0–100; we store them
            # as ratios in [0, 1] for consistency with career
            # serve/return percentages elsewhere.
            v = _coerce_int(block.get(key))
            if v is None:
                return None
            return v / 100.0

        decider = _wins_total("decidingSetWin", "decidingSetCount")
        tiebreak = _wins_total("tiebreakWon", "tiebreakCount")
        bo3 = _wins_total("bestOfThreeWon", "bestOfThreeCount")
        bo5 = _wins_total("bestOfFiveWon", "bestOfFiveCount")
        comeback_pct = _pct("firstSetLoseMatchWinPercentage")
        closeout_pct = _pct("firstSetWinMatchWinPercentage")
        first_serve_win = _pct("winningOnFirstServePercentage")
        bp_convert = _pct("breakpointsWonPercentage")

        # Suppress when the player has literally no matchup-conditioned
        # signal — caller will drop the whole H2H if both sides come
        # back None.
        if all(
            v is None
            for v in (
                decider, tiebreak, bo3, bo5,
                comeback_pct, closeout_pct,
                first_serve_win, bp_convert,
            )
        ):
            return None

        return TennisInMatchupStats(
            decider_record=decider,
            tiebreak_record=tiebreak,
            bo3_record=bo3,
            bo5_record=bo5,
            first_set_lost_match_won_pct=comeback_pct,
            first_set_won_match_won_pct=closeout_pct,
            first_serve_win_pct=first_serve_win,
            break_point_convert_pct=bp_convert,
        )

    # ----- Public entry point -----

    async def fetch(
        self, identity: TennisMatchIdentity
    ) -> TennisStatsContext | None:
        await self._ensure_index(identity.tour)
        a_hit = self._resolve(identity.tour, identity.player_a)
        b_hit = self._resolve(identity.tour, identity.player_b)
        if a_hit is None and b_hit is None:
            log.debug(
                "matchstat: neither player resolved (%s vs %s, tour=%s)",
                identity.player_a, identity.player_b, identity.tour,
            )
            return None

        # Player blocks fetched in parallel; H2H gated on both IDs being
        # known (the H2H endpoint requires two IDs).
        player_a, player_b = await asyncio.gather(
            self._player_stats(identity.tour, identity.player_a, a_hit),
            self._player_stats(identity.tour, identity.player_b, b_hit),
        )

        h2h: TennisHeadToHead | None = None
        if a_hit is not None and b_hit is not None:
            h2h = await self._head_to_head(
                identity.tour,
                a_hit[0],
                b_hit[0],
                identity.player_a,
                identity.player_b,
            )

        return TennisStatsContext(
            provider=self.name,
            fetched_at=datetime.now(UTC),
            tournament=identity.tournament_hint,
            player_a=player_a,
            player_b=player_b,
            head_to_head=h2h,
        )
