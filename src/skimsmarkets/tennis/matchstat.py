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
from datetime import UTC, date, datetime, timedelta
from types import TracebackType
from typing import Any, Self

import httpx

from skimsmarkets.tennis.identity import TennisMatchIdentity
from skimsmarkets.tennis.models import (
    PerMatchStats,
    TennisH2HMeeting,
    TennisHeadToHead,
    TennisInMatchupStats,
    TennisPlayerStats,
    TennisRecentMatch,
    TennisStatsContext,
)
from skimsmarkets.tennis.provider import (
    MatchHistoryRow,
    MatchStatsFixture,
    _match_completeness,
    _parse_set_scores,
    parse_score_details,
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

# How many rows to slice for the recent-matches prompt digest. Distinct
# from the FETCH page size below — the digest stays compact for prompt
# tokens, while the fetch pulls more to feed the variance estimator and
# career-aggregate clutch computation.
_RECENT_MATCH_DIGEST_SIZE = 5
_RECENT_MEETING_PAGE_SIZE = 3

# Unified past-matches fetch size. Sized for the longest-tail consumer:
# career-aggregate clutch (tiebreak / decider / comeback / close-match
# rates on `TennisPlayerStats`) needs ~50 matches for stable
# denominators. The variance estimator and recent-matches digest both
# slice from this same cached list — denominator-larger is monotonically
# better for the variance estimator and the digest still slices to
# `_RECENT_MATCH_DIGEST_SIZE` for the prompt. Vendor caps at 100/page;
# 50 stays well under and keeps response payloads manageable.
_PAST_MATCHES_FETCH_SIZE = 50
# Include relations for past-matches. `stat` brings per-match box-score
# (first-serve %, BP convert / face counts, used by the variance
# estimator and clutch aggregator); `tournament,round` brings surface /
# tier / round name for `TennisRecentMatch`. Always-on now — both the
# selector warmup and the live recent-matches path share this shape so
# the cache populated by either is immediately reusable by the other.
_PAST_MATCHES_INCLUDE = "stat,tournament,round"

# Recency window for the BP-save 180d feature on `TennisPlayerStats`.
# Layered over the existing career BP-save % (no time bound) so the
# lens / GBT can read recent-form arcs (e.g. 75% recent vs 65% career
# = upswing).
_BP_SAVE_RECENCY_DAYS = 180

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
    """Lowercase, diacritic-stripped, hyphen-collapsed, single-spaced.

    Used for keying the rankings index. The vendor and Polymarket both
    ship names in roughly the same Latin form, but the vendor preserves
    diacritics (Cóbolli, Müller) while Polymarket question strings
    sometimes drop them. Polymarket also occasionally hyphenates
    compound given names (e.g. 'En-Shuo Liang') where the vendor's
    rankings list ships them with a space ('En Shuo Liang'); collapsing
    hyphens to spaces makes both lookups land on the same key. The
    final split/join collapses any resulting double-spaces.
    """
    nfkd = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(stripped.replace("-", " ").lower().split())


def _surname_token(name: str | None) -> str | None:
    """Last whitespace-separated token of `_normalize_name(name)`,
    further stripped to alnum-only.

    Mirrors `kalshi/slate.py:_surname_from_yes_sub_title` so the
    fixtures-overlay index keys line up with the slug-synthesised
    surnames the slate adapter writes. Returns None for empty / null
    input or names that strip to empty.
    """
    if not name:
        return None
    norm = _normalize_name(name)
    if not norm:
        return None
    last = norm.split()[-1]
    cleaned = "".join(c for c in last if c.isalnum())
    return cleaned or None


def _surname_pair_key(
    name_a: str | None, name_b: str | None
) -> frozenset[str] | None:
    """`frozenset({surname_a, surname_b})` for fixture-index keys.

    Returns None when either side strips to empty or both surnames
    collide (a same-surname matchup would key to a singleton set,
    making lookup ambiguous — defer to the singleton handler upstream
    rather than risk wrong attribution).
    """
    a = _surname_token(name_a)
    b = _surname_token(name_b)
    if a is None or b is None or a == b:
        return None
    return frozenset({a, b})


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


def _safe_pct(numerator: Any, denominator: Any) -> float | None:
    """Vendor ships fraction pairs like (47, 65) → 0.7231. None when the
    denominator is missing, zero, or unparseable. Used by `PerMatchStats`
    construction to convert vendor counters into ratios that line up
    field-for-field with the career-baseline percentages on
    `TennisPlayerStats`.
    """
    n = _coerce_int(numerator)
    d = _coerce_int(denominator)
    if n is None or d is None or d == 0:
        return None
    return n / d


def _parse_match_stats_block(
    row: dict[str, Any], subject_player_id: int
) -> PerMatchStats | None:
    """Build `PerMatchStats` from a single past-matches row's `stats` block.

    The vendor structures `stats` as `{player1: {...}, player2: {...}}`
    keyed by side, NOT by player id — so we read whichever side
    corresponds to the subject (matched against `player1Id` on the row).
    Returns None when the row carries no `stats` block (live-suspended
    matches, walkovers, very old matches the vendor ingested without
    box scores).
    """
    stats = row.get("stats")
    if not isinstance(stats, dict):
        return None
    p1_id = _coerce_int(row.get("player1Id"))
    is_subject_p1 = p1_id == subject_player_id
    side = stats.get("player1" if is_subject_p1 else "player2")
    if not isinstance(side, dict):
        return None
    return PerMatchStats(
        first_serve_in_pct=_safe_pct(
            side.get("firstServe"), side.get("firstServeOf")
        ),
        first_serve_win_pct=_safe_pct(
            side.get("winningOnFirstServe"), side.get("winningOnFirstServeOf")
        ),
        second_serve_win_pct=_safe_pct(
            side.get("winningOnSecondServe"), side.get("winningOnSecondServeOf")
        ),
        break_point_convert_pct=_safe_pct(
            side.get("breakPointsConverted"),
            side.get("breakPointsConvertedOf"),
        ),
        aces=_coerce_int(side.get("aces")),
        double_faults=_coerce_int(side.get("doubleFaults")),
        total_points_won=_coerce_int(side.get("totalPointsWon")),
    )


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
        # Per-player past-matches cache keyed by (tour, pid) → parsed
        # `MatchHistoryRow` list. Populated by
        # `warm_match_history_for_selection` at slate-time pre-cap, read
        # synchronously by `lookup_player_match_history` at selector
        # scoring time, and re-read by `_player_recent_matches` at
        # post-cap enrichment time so the warmup HTTP isn't wasted.
        # Empty list (rather than missing key) marks a player whose
        # vendor lookup ran but returned no usable rows — distinguishes
        # "warmup didn't fire for this player" from "warmup fired but
        # found nothing", so the live fallback only runs for the former.
        self._past_matches_cache: dict[tuple[str, int], list[MatchHistoryRow]] = {}
        # Per-player surface-summary cache keyed by (tour, pid) → the
        # `(ytd_pair, surface_dict)` shape `_player_surface_year_record`
        # already returns. Populated by
        # `warm_surface_summary_for_selection` at slate-time pre-cap;
        # read synchronously by `lookup_player_surface_record` at
        # selector scoring time, and re-read by
        # `_player_surface_year_record` at post-cap enrichment so the
        # warmup HTTP isn't wasted on survivors. The presence of the
        # key (regardless of contents) marks "warmup ran for this
        # player" — distinguishes from a true cache miss.
        self._surface_cache: dict[
            tuple[str, int],
            tuple[
                tuple[int, int] | None,
                dict[str, tuple[int, int]] | None,
            ],
        ] = {}
        # Per-matchup H2H cache keyed by `(tour, a_id, b_id)` ordered by
        # identity convention (NOT sorted) → the constructed
        # `TennisHeadToHead` or None for empty H2H. Populated by
        # `warm_h2h_for_selection` at slate-time pre-cap, read
        # synchronously by `lookup_h2h` at selector scoring time, and
        # re-read by `_head_to_head` at post-cap enrichment so the
        # warmup HTTP isn't wasted on cap-survivor events. Identity
        # ordering matters: `TennisHeadToHead.a_wins` is positional to
        # the URL path's first ID, so a swapped lookup would invert
        # surface_h2h tuples.
        self._h2h_cache: dict[
            tuple[str, int, int], TennisHeadToHead | None
        ] = {}
        # Per-fixture cache keyed by `(tour, frozenset({surname_a,
        # surname_b}))` → the `MatchStatsFixture` returned for that
        # matchup. Populated as a side effect of
        # `fetch_fixtures_for_date` (the same call already runs at
        # slate-overlay time for the tipoff overlay) and read by
        # `fetch()` so `TennisStatsContext.surface` can pick up the
        # vendor's `tournament.court.name` for events the slug-prefix
        # surface map misses (lower tiers, the standard slug format
        # `{tour}-{lastA}-{lastB}-{date}` carries no tournament token).
        # Cached for the provider's lifetime; surname-pair collisions
        # across dates fall back to the latest-written fixture, which
        # is benign because surface is tournament-property-stable.
        self._fixture_cache: dict[
            tuple[str, frozenset[str]], MatchStatsFixture
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
            if birth is None:
                # MatchStat dropped `information.birthdate` from this
                # endpoint as of mid-2025 (verified by probing 2026-05-06
                # in `gbt_backfill.py`). Synthesise from `turnedPro` —
                # ATP/WTA average turn-pro age of ~17 — same shape the
                # backfill path uses. Coarse (actual birth year drifts
                # ±2y around this) but a noisy age signal beats every
                # downstream consumer reading None.
                turned_pro = info.get("turnedPro")
                if isinstance(turned_pro, str) and turned_pro.isdigit():
                    birth = date(int(turned_pro) - 17, 7, 1)
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

    @staticmethod
    def _compute_career_clutch(
        rows: Iterable[MatchHistoryRow], today: date
    ) -> dict[str, Any]:
        """Aggregate `MatchHistoryRow`s into the 5 career-aggregate
        clutch fields on `TennisPlayerStats`.

        Walks each row, parses its score via `parse_score_details`,
        rotates winner-relative facts onto the subject's perspective
        (the row already carries `won` for the subject), and folds into
        running counters.

        BP-save 180d counters use the row's pre-derived `bp_saved` /
        `bp_faced` (subject's perspective; computed at row-parse time
        from the OPPONENT's BP-convert counters since the vendor only
        ships convert per row). Filtered by `date >= today - 180d`.

        Each output record is suppressed (set to None in the dict) when
        its denominator is 0 — a player who's never been to a decider
        gets `career_decider_record=None`, not `(0, 0)`, so the lens
        renderer's "suppress empty lines" gate works without per-line
        null-checks downstream.
        """
        tb_won = tb_played = 0
        dec_won = dec_played = 0
        cb_won = cb_total = 0
        cm_won = cm_played = 0
        bp_saved_180d = bp_faced_180d = 0
        recency_cutoff = today - timedelta(days=_BP_SAVE_RECENCY_DAYS)
        for r in rows:
            details = parse_score_details(r.raw_score, r.best_of, r.winner_side)
            if details is not None:
                # Decider — winner of the decider IS the match winner, so
                # subject's decider win == subject's match win when the
                # match went the distance.
                if details.went_to_decider:
                    dec_played += 1
                    if r.won:
                        dec_won += 1
                # Tiebreaks — rotate winner-relative TB count onto subject.
                tb_played += details.n_tiebreaks_played
                subject_tbs_won = (
                    details.winner_tiebreaks_won
                    if r.won
                    else details.n_tiebreaks_played - details.winner_tiebreaks_won
                )
                tb_won += subject_tbs_won
                # Close matches.
                if details.is_close_match:
                    cm_played += 1
                    if r.won:
                        cm_won += 1
                # Comeback — subject lost set 1 AND won match. The
                # denominator is matches where subject lost set 1
                # (the "given a set-1 deficit" condition).
                subject_won_set_one = (
                    details.winner_won_set_one
                    if r.won
                    else not details.winner_won_set_one
                )
                if not subject_won_set_one:
                    cb_total += 1
                    if r.won:
                        cb_won += 1
            # BP-save recency window — independent of score parse;
            # derived at row-parse time from opponent's BP-convert
            # counters.
            if (
                r.date is not None
                and r.date >= recency_cutoff
                and r.bp_saved is not None
                and r.bp_faced is not None
            ):
                bp_saved_180d += r.bp_saved
                bp_faced_180d += r.bp_faced
        return {
            "career_tiebreak_record": (tb_won, tb_played) if tb_played else None,
            "career_decider_record": (dec_won, dec_played) if dec_played else None,
            "career_comeback_record": (cb_won, cb_total) if cb_total else None,
            "career_close_match_record": (cm_won, cm_played) if cm_played else None,
            "break_point_save_pct_180d": (
                bp_saved_180d / bp_faced_180d if bp_faced_180d else None
            ),
        }

    @staticmethod
    def _parse_match_history_row(
        row: dict[str, Any], pid: int
    ) -> MatchHistoryRow | None:
        """Vendor past-matches row → `MatchHistoryRow`.

        Single source of truth for past-matches parsing. Both the
        selector's warmup (`warm_match_history_for_selection`) and the
        live recent-matches path (`_player_recent_matches`) consume the
        same parsed rows, which guarantees `match_completeness` and
        per-row `first_serve_win_pct` are computed identically wherever
        the row was first seen.

        Returns None when the row can't be attributed (no opponent name)
        — pretty much never happens on the live vendor but guard
        anyway. `match_completeness` and `first_serve_win_pct` may be
        None on a returned row when the score is aborted or the stat
        block is absent; consumers filter at aggregation time, not
        here, because TennisRecentMatch construction still wants the
        row for its scoreline + surface + opponent info.
        """
        p1 = row.get("player1") if isinstance(row.get("player1"), dict) else {}
        p2 = row.get("player2") if isinstance(row.get("player2"), dict) else {}
        p1_id = _coerce_int(row.get("player1Id"))
        is_subject_p1 = p1_id == pid
        opp_block = p2 if is_subject_p1 else p1
        opp_name = opp_block.get("name") if isinstance(opp_block, dict) else None
        if not isinstance(opp_name, str) or not opp_name:
            return None
        winner_id = _coerce_int(row.get("match_winner"))
        won = winner_id == pid
        p2_id = _coerce_int(row.get("player2Id"))
        # Side index (1 or 2) of the match winner relative to row p1/p2.
        # `parse_score_details` needs this to identify which side's
        # set-1 / tiebreak counts to treat as "winner's". None on
        # walkover/abandoned rows where `match_winner` doesn't match
        # either player_id.
        if winner_id == p1_id:
            winner_side: int | None = 1
        elif winner_id == p2_id:
            winner_side = 2
        else:
            winner_side = None
        row_date = _parse_iso_date(row.get("date"))
        tourn = row.get("tournament") if isinstance(row.get("tournament"), dict) else {}
        surface = _surface_from_court_id(tourn.get("courtId"))
        tier = _tier_from_rank_id(tourn.get("rankId"))
        tname = tourn.get("name")
        tournament_name = (
            tname.strip() if isinstance(tname, str) and tname.strip() else None
        )
        rnd = row.get("round") if isinstance(row.get("round"), dict) else {}
        rname = rnd.get("name")
        round_name = (
            rname.strip() if isinstance(rname, str) and rname.strip() else None
        )
        result = row.get("result")
        score = result.strip() if isinstance(result, str) and result.strip() else None

        # Match completeness (winner's share of total games). Symmetric
        # in p1/p2 — independent of `won`.
        completeness: float | None = None
        score_pair = _parse_set_scores(score)
        if score_pair is not None:
            completeness = _match_completeness(*score_pair)

        # Per-match first-serve-win % from the stat block. Only present
        # when the warmup-include path fetched this row (live no-stat
        # path leaves the field absent and we degrade to None
        # silently). Subject's side: `player1` block when subject is
        # p1, else `player2`.
        first_serve_win_pct: float | None = None
        bp_saved: int | None = None
        bp_faced: int | None = None
        stats = row.get("stats")
        if isinstance(stats, dict):
            side = stats.get("player1" if is_subject_p1 else "player2")
            opp = stats.get("player2" if is_subject_p1 else "player1")
            if isinstance(side, dict):
                first_serve_win_pct = _safe_pct(
                    side.get("winningOnFirstServe"),
                    side.get("winningOnFirstServeOf"),
                )
            # Subject's BP-faced equals opponent's BP-convert
            # opportunities; subject's BP-saved is the count NOT
            # converted. Vendor ships only the converted side per row,
            # so we derive saved arithmetically from the opponent block.
            if isinstance(opp, dict):
                opp_conv = _coerce_int(opp.get("breakPointsConverted"))
                opp_conv_of = _coerce_int(opp.get("breakPointsConvertedOf"))
                if opp_conv is not None and opp_conv_of is not None:
                    bp_faced = opp_conv_of
                    bp_saved = opp_conv_of - opp_conv

        # Vendor's `best_of` is unpopulated on every probed row — empirically
        # 0/24 across the recent slate. `parse_score_details` requires
        # `best_of` to mark deciders (`went_to_decider = n_sets == best_of`),
        # so a None vendor field silently nukes `career_decider_record` on
        # every player. Mirror `simulation.detect_best_of`'s tier-based
        # inference per-row: Grand Slam → 5, everything else → 3. Most
        # tour-level matches are bo3, so the inferred value is right ≥99%
        # of the time; the alternative (None for every row) is wrong 100%
        # of the time.
        bo = _coerce_int(row.get("best_of"))
        if bo is None:
            bo = 5 if tier == "grand_slam" else 3
        return MatchHistoryRow(
            date=row_date,
            opponent_name=opp_name.strip(),
            won=won,
            raw_score=score,
            surface=surface,
            round=round_name,
            tournament_name=tournament_name,
            tournament_tier=tier,
            match_completeness=completeness,
            first_serve_win_pct=first_serve_win_pct,
            best_of=bo,
            winner_side=winner_side,
            bp_saved=bp_saved,
            bp_faced=bp_faced,
        )

    async def _fetch_past_matches_rows(
        self,
        tour: str,
        pid: int,
        *,
        page_size: int,
        include: str,
    ) -> list[MatchHistoryRow]:
        """One past-matches HTTP → list of parsed `MatchHistoryRow`s.

        Failure / empty response → `[]` so the caller can distinguish
        "fetched but found nothing" (stays empty in the cache) from
        "haven't fetched yet" (key absent in cache). Used by both the
        selection-stage warmup (`include=stat,...`, pageSize=10) and
        the post-cap fallback inside `_player_recent_matches`
        (lighter `tournament,round`-only include, pageSize=5) — same
        endpoint, same parser, different inclusiveness.
        """
        body = await self._get(
            f"/tennis/v2/{tour}/player/past-matches/{pid}",
            params={"pageSize": page_size, "include": include},
        )
        rows = body.get("data") if isinstance(body, dict) else None
        if not isinstance(rows, list) or not rows:
            return []
        out: list[MatchHistoryRow] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            parsed = self._parse_match_history_row(row, pid)
            if parsed is not None:
                out.append(parsed)
        return out

    async def warm_match_history_for_selection(
        self, identities: Iterable[TennisMatchIdentity]
    ) -> None:
        """Pre-fetch `/past-matches?include=stat,...` for every player
        across `identities` and populate `self._past_matches_cache`.

        Selection-stage scoring reads the cache synchronously via
        `lookup_player_match_history` to compute a consistency /
        variance metric. Same dedup-by-unique-player pattern as
        `warm_form_for_selection` — collect the unique (tour, pid)
        pairs after rank resolution, fan out one HTTP per unique
        player, skip pairs already cached.

        Pre-condition: rankings index warm. Identities that don't
        resolve are silently skipped — selector requires both ranks
        to compute a score, so unranked players don't need a history
        cache anyway.

        Cache reuse downstream: the post-cap `_player_recent_matches`
        path also reads this cache, so the warmup HTTP serves both the
        selector's variance estimator AND the eventual prompt's
        recent-matches digest.
        """
        pending: set[tuple[str, int]] = set()
        for ident in identities:
            for nm in (ident.player_a, ident.player_b):
                hit = self._resolve(ident.tour, nm)
                if hit is None:
                    continue
                pid = hit[0]
                if (ident.tour, pid) in self._past_matches_cache:
                    continue
                pending.add((ident.tour, pid))
        if not pending:
            return
        log.info(
            "matchstat: warming past-matches cache for %d players (selection)",
            len(pending),
        )

        async def _one(tour: str, pid: int) -> None:
            rows = await self._fetch_past_matches_rows(
                tour,
                pid,
                page_size=_PAST_MATCHES_FETCH_SIZE,
                include=_PAST_MATCHES_INCLUDE,
            )
            self._past_matches_cache[(tour, pid)] = rows

        await asyncio.gather(*(_one(tour, pid) for tour, pid in pending))

    def lookup_player_match_history(
        self, tour: str, name: str
    ) -> list[MatchHistoryRow] | None:
        """Sync per-match history lookup from the warmed cache.

        Returns None when:
          - the player isn't in the rankings index for this tour
          - `warm_match_history_for_selection` hasn't been called for
            their identity (cache miss — distinguished from a populated
            empty list)
          - the warmup ran but the vendor returned no rows
        Selection callers treat None / empty as "no consistency signal"
        and skip the consistency adjustment.
        """
        hit = self._resolve(tour, name)
        if hit is None:
            return None
        pid = hit[0]
        cached = self._past_matches_cache.get((tour, pid))
        if cached is None or not cached:
            return None
        return cached

    async def _player_recent_matches(
        self, tour: str, pid: int, name: str
    ) -> tuple[date | None, list[TennisRecentMatch]]:
        """Pull the last N matches; return `(last_match_date, recent_matches)`.

        Reads from `self._past_matches_cache` first when populated by
        the selection-stage warmup — same vendor data, parsed once,
        reused for the prompt block. Cache miss falls back to the
        original lighter HTTP (no `include=stat`) so small slates
        that didn't trigger selection warmup pay the same cost as
        before. The first row's date doubles as `last_match_date`;
        rows feed `TennisPlayerStats.recent_matches`.

        `name` is unused at parse time but kept in the signature so the
        call site reads as "fetch recent matches for this player by
        name" and avoids a silent reordering bug if the IDs ever flip.
        """
        del name  # see docstring — call-site readability only.

        rows = self._past_matches_cache.get((tour, pid))
        if rows is None:
            rows = await self._fetch_past_matches_rows(
                tour,
                pid,
                page_size=_PAST_MATCHES_FETCH_SIZE,
                include=_PAST_MATCHES_INCLUDE,
            )
            # Populate the cache so a downstream lookup doesn't re-fetch.
            # The selector warmup uses the same fetch shape, so a cache
            # populated by either path is immediately reusable by the
            # other (variance estimator, clutch aggregator, digest).
            self._past_matches_cache[(tour, pid)] = rows
        if not rows:
            return None, []

        last_date: date | None = next(
            (r.date for r in rows if r.date is not None), None
        )
        out: list[TennisRecentMatch] = [
            TennisRecentMatch(
                date=r.date,
                opponent_name=r.opponent_name,
                won=r.won,
                result=r.raw_score,
                surface=r.surface,
                round=r.round,
                tournament_name=r.tournament_name,
                tournament_tier=r.tournament_tier,
            )
            for r in rows[:_RECENT_MATCH_DIGEST_SIZE]
        ]
        return last_date, out

    async def fetch_post_match_stats(
        self,
        tour: str,
        player_id: int,
        on_date: date,
        opponent_name: str,
    ) -> PerMatchStats | None:
        """Pull box-score stats for one completed match (retro-only path).

        Same `past-matches` endpoint and include shape as the live
        pipeline (`_PAST_MATCHES_INCLUDE`) — both paths need the per-
        match `stats.player1` / `stats.player2` block, the live path for
        career-aggregate clutch + variance estimation and the retro
        path for the post-match `PerMatchStats` row.

        Match-row identification: `on_date` (UTC date the match was
        played) plus the opponent name. `opponent_name` is matched
        case-insensitively after diacritic-stripping (vendor preserves
        accents that gamma sometimes drops). Date match alone is the
        fallback for the rare day where a player only has one match
        — names from gamma occasionally differ enough from the vendor's
        canonical form that even normalisation misses (different
        transliteration). When multiple rows share the date and no
        name match lands, we log and return None rather than guess.

        Returns None on:
          - vendor 404 / network failure (per `_get` posture)
          - no row matching `on_date` (match not yet ingested by vendor;
            common for very recent results)
          - row found but `stats` block missing or empty (live-suspended
            matches, walkovers)
          - division by zero on every percentage (extremely rare —
            vendor ships zero counts for both numerator and denominator
            on aborted matches)

        pageSize=20 covers ~3 weeks of typical tour activity, which is
        plenty for retro fetches (we're always hitting matches that
        are at most a few weeks old, before the operator runs the
        retro analysis). Distinct from the live path's
        `_PAST_MATCHES_FETCH_SIZE` (50) because retro doesn't need
        career-aggregate denominators — just the one row matching
        `on_date`.
        """
        body = await self._get(
            f"/tennis/v2/{tour}/player/past-matches/{player_id}",
            params={"pageSize": 20, "include": _PAST_MATCHES_INCLUDE},
        )
        rows = body.get("data") if isinstance(body, dict) else None
        if not isinstance(rows, list) or not rows:
            return None

        target_opponent = _normalize_name(opponent_name)
        # Pass 1: exact (date AND opponent-name) match. The desired
        # outcome on every well-formed retro fetch.
        date_matches: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_date = _parse_iso_date(row.get("date"))
            if row_date != on_date:
                continue
            date_matches.append(row)
            # Pull opponent name from whichever side ISN'T this player.
            p1_id = _coerce_int(row.get("player1Id"))
            opp_block = (
                row.get("player2") if p1_id == player_id else row.get("player1")
            )
            if not isinstance(opp_block, dict):
                continue
            vendor_opp = opp_block.get("name")
            if not isinstance(vendor_opp, str):
                continue
            if _normalize_name(vendor_opp) == target_opponent:
                return _parse_match_stats_block(row, player_id)

        # Pass 2: name didn't match but exactly one row shares the date.
        # Vendor occasionally ships a transliteration we can't normalise
        # (e.g. "Mensik" vs "Menšík" with non-trivial diacritic shape).
        # Date alone is unique enough on a single player's match history
        # at single-day granularity — players rarely play two matches
        # on the same calendar day at tour level.
        if len(date_matches) == 1:
            return _parse_match_stats_block(date_matches[0], player_id)
        if len(date_matches) > 1:
            log.warning(
                "matchstat post-match: %d rows on %s for player_id=%d, "
                "no opponent-name match against %r — declining to guess",
                len(date_matches), on_date.isoformat(), player_id, opponent_name,
            )
        return None

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

    async def _fetch_surface_summary(
        self, tour: str, pid: int
    ) -> tuple[
        tuple[int, int] | None,
        dict[str, tuple[int, int]] | None,
    ]:
        """One surface-summary HTTP → `(ytd_pair, surface_dict)`.

        Pure parsing helper, separate from cache lookup. Used by both
        the selection-stage warmup (`warm_surface_summary_for_selection`)
        and the cache-miss fallback inside `_player_surface_year_record`.

        Vendor returns one row per year with a list of per-court
        win/loss counts. We take the most recent year (vendor sorts
        descending; we double-check with `max(year)` as a fallback).
        Hard + I.hard collapse into a single "hard" entry to match
        the sport-hint vocabulary.
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

    async def _player_surface_year_record(
        self, tour: str, pid: int
    ) -> tuple[
        tuple[int, int] | None,
        dict[str, tuple[int, int]] | None,
    ]:
        """Cache-then-fetch wrapper around `_fetch_surface_summary`.

        Consults `self._surface_cache` first when populated by the
        selection-stage warmup — same vendor data, parsed once,
        reused for the prompt block. Cache miss falls back to a
        direct fetch and populates the cache so a second consumer
        within the same run doesn't re-fetch.
        """
        cached = self._surface_cache.get((tour, pid))
        if cached is not None:
            return cached
        result = await self._fetch_surface_summary(tour, pid)
        self._surface_cache[(tour, pid)] = result
        return result

    async def warm_surface_summary_for_selection(
        self, identities: Iterable[TennisMatchIdentity]
    ) -> None:
        """Pre-fetch `/player/surface-summary/{id}` for every player
        across `identities` and populate `self._surface_cache`.

        Same dedup-by-unique-player pattern as
        `warm_form_for_selection` and
        `warm_match_history_for_selection`. Selection-stage scoring
        reads the cache via `lookup_player_surface_record` to compute
        a surface-specialism adjustment. Pre-condition: rankings
        index warm. Cache reuse downstream:
        `_player_surface_year_record` consults this cache before
        re-fetching, so the warmup HTTP serves both the selector's
        specialism estimator AND the eventual prompt's surface block.
        """
        pending: set[tuple[str, int]] = set()
        for ident in identities:
            for nm in (ident.player_a, ident.player_b):
                hit = self._resolve(ident.tour, nm)
                if hit is None:
                    continue
                pid = hit[0]
                if (ident.tour, pid) in self._surface_cache:
                    continue
                pending.add((ident.tour, pid))
        if not pending:
            return
        log.info(
            "matchstat: warming surface-summary cache for %d players (selection)",
            len(pending),
        )

        async def _one(tour: str, pid: int) -> None:
            self._surface_cache[(tour, pid)] = await self._fetch_surface_summary(
                tour, pid
            )

        await asyncio.gather(*(_one(tour, pid) for tour, pid in pending))

    def lookup_player_surface_record(
        self, tour: str, name: str
    ) -> tuple[
        tuple[int, int] | None, dict[str, tuple[int, int]] | None
    ] | None:
        """Sync `(ytd_pair, surface_dict)` lookup from the warmed cache.

        Returns None when the player isn't in the rankings index or
        the surface warmup hasn't run for them. Both inner pair
        components are independently None when the vendor returned
        an empty section. Selection callers treat outer-None as
        "no surface signal" and skip the surface tier.
        """
        hit = self._resolve(tour, name)
        if hit is None:
            return None
        pid = hit[0]
        cached = self._surface_cache.get((tour, pid))
        if cached is None:
            return None
        return cached

    def lookup_player_profile_extras(
        self, tour: str, name: str
    ) -> tuple[int | None, int | None] | None:
        """Sync `(age_years, best_rank)` lookup from the warmed
        profile cache.

        Reads the same cache slot `lookup_player_form` reads — no
        extra HTTP, just a different view of the warmed profile.
        Returns None when the player isn't in the rankings index or
        `warm_form_for_selection` hasn't been called for their
        identity. Inner pair components are independently None when
        the vendor's profile response was missing the underlying
        field (no birthdate, no career-high rank).
        """
        hit = self._resolve(tour, name)
        if hit is None:
            return None
        pid = hit[0]
        cached = self._profile_cache.get((tour, pid))
        if cached is None:
            return None
        _form_arr, best_rank, age_years, _plays = cached
        return age_years, best_rank

    async def warm_h2h_for_selection(
        self, identities: Iterable[TennisMatchIdentity]
    ) -> None:
        """Pre-fetch /h2h/{info,matches,stats} per matchup and populate
        `self._h2h_cache`.

        Selection-stage scoring reads the cache synchronously via
        `lookup_h2h` to compute an H2H sample-size + surface-conditioned
        bonus. Same dedup pattern as the per-player warmups, but keyed
        by **matchup pair** rather than by individual player — collect
        unique `(tour, a_id, b_id)` triples after rank resolution, fan
        out one `_head_to_head` call per unique triple, skip pairs
        already cached.

        Pre-condition: rankings index warm. Identities where either
        player doesn't resolve are silently skipped — the H2H endpoint
        requires both vendor IDs.

        Cache reuse downstream: the post-cap `fetch()` path's
        `_head_to_head` consult-cache-first behaviour means cap-survivor
        events re-use the warmed payload, so the warmup HTTP serves
        both the selector's H2H tier AND the eventual prompt's H2H
        block.
        """
        pending: list[tuple[str, int, int, str, str]] = []
        seen: set[tuple[str, int, int]] = set()
        for ident in identities:
            a_hit = self._resolve(ident.tour, ident.player_a)
            b_hit = self._resolve(ident.tour, ident.player_b)
            if a_hit is None or b_hit is None:
                continue
            a_id = a_hit[0]
            b_id = b_hit[0]
            key = (ident.tour, a_id, b_id)
            if key in seen or key in self._h2h_cache:
                continue
            seen.add(key)
            pending.append(
                (ident.tour, a_id, b_id, ident.player_a, ident.player_b)
            )
        if not pending:
            return
        log.info(
            "matchstat: warming H2H cache for %d matchups (selection)",
            len(pending),
        )
        await asyncio.gather(
            *(
                self._head_to_head(tour, a_id, b_id, name_a, name_b)
                for tour, a_id, b_id, name_a, name_b in pending
            )
        )

    def lookup_h2h(
        self, tour: str, name_a: str, name_b: str
    ) -> TennisHeadToHead | None:
        """Sync H2H lookup from the warmed matchup cache.

        Returns None when:
          - either player isn't in the rankings index for this tour
          - `warm_h2h_for_selection` hasn't been called for this matchup
            (cache key absent)
          - the warmup ran but the vendor returned no usable H2H rows
            (cache key present, value is None)
        Selection callers treat None as "no H2H signal" — no penalty,
        no bonus.

        Identity-ordered: caller passes `name_a` / `name_b` in the same
        order as `TennisMatchIdentity` (the order used at warmup time).
        Reordering bypasses the cache key.
        """
        a_hit = self._resolve(tour, name_a)
        b_hit = self._resolve(tour, name_b)
        if a_hit is None or b_hit is None:
            return None
        return self._h2h_cache.get((tour, a_hit[0], b_hit[0]))

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

        # Career-aggregate clutch — read from the past-matches cache the
        # `_player_recent_matches` call above just populated. Pure post-
        # processing of cached rows; no extra HTTP. Today() pinned to
        # UTC for the recency-window computation so the 180d cutoff is
        # stable across timezones.
        cached_rows = self._past_matches_cache.get((tour, pid)) or []
        clutch_aggregates = self._compute_career_clutch(
            cached_rows, datetime.now(UTC).date()
        )

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
            **clutch_aggregates,
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

        Cache shape: keyed by `(tour, a_id, b_id)` in identity-positional
        order (NOT sorted). Both selection-stage warmup
        (`warm_h2h_for_selection`) and post-cap `fetch()` route through
        this method, so the cache absorbs duplicate work — warmup
        populates for every matchup in the slate, runtime fetch reuses
        the cached payload for cap-survivor events without re-issuing
        the three /h2h/* HTTPs.
        """
        cache_key = (tour, a_id, b_id)
        if cache_key in self._h2h_cache:
            return self._h2h_cache[cache_key]
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
            self._h2h_cache[cache_key] = None
            return None

        result = TennisHeadToHead(
            a_wins=a_wins_total,
            b_wins=b_wins_total,
            surface_h2h=surface_h2h or None,
            recent_meetings=recent_meetings or None,
            a_in_matchup=a_in_matchup,
            b_in_matchup=b_in_matchup,
        )
        self._h2h_cache[cache_key] = result
        return result

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

    # ----- Slate-build helper (per-date scheduled fixtures) -----

    async def fetch_fixtures_for_date(
        self, *, tour: str, date_iso: str
    ) -> dict[frozenset[str], MatchStatsFixture]:
        """`/tennis/v2/{tour}/fixtures/{date}` → surname-pair →
        `MatchStatsFixture` (date, player IDs, tournament, surface,
        round).

        Pulls the fixtures payload with
        `include=tournament,tournament.court,round` so the response
        carries everything the slate-overlay stage needs in a single
        HTTP — tipoff, player IDs, tournament name, surface, round.

        **Side effect**: seeds the internal rankings index with
        fixture-derived `(player_id, None, None)` entries for any
        player not already present. Position and points stay None
        for these entries (we don't have rankings for ITF futures
        players), but the IDs themselves unlock every per-player
        stats endpoint downstream — `_player_profile`,
        `_player_recent_matches`, `_head_to_head`, etc. are
        ID-keyed, not rank-keyed. Existing ranked players are NEVER
        overwritten; the seeding only fills empty slots.

        Returns a dict keyed by `frozenset({surname_a, surname_b})`
        (lowercased, diacritic-stripped, alnum-only — same canonical
        form `kalshi/slate.py:_surname_from_yes_sub_title` produces).
        Empty dict on any failure or when the vendor has no fixtures
        for that date — overlay degrades silently per missing match.

        `pageSize=500` covers the busiest tournament-day combination
        (M-tier futures days can ship 100+ singles fixtures per tour);
        the endpoint paginates beyond that but the slate horizon
        filter upstream caps practical demand well below that ceiling.
        """
        # Ensure the index exists for this tour BEFORE seeding —
        # otherwise our setdefault writes would land in an empty dict
        # that then gets overwritten by `_ensure_index`'s populate.
        await self._ensure_index(tour)

        path = f"/tennis/v2/{tour}/fixtures/{date_iso}"
        payload = await self._get(
            path,
            params={
                "pageSize": 500,
                "include": "tournament,tournament.court,round",
            },
        )
        out: dict[frozenset[str], MatchStatsFixture] = {}
        if not isinstance(payload, dict):
            return out
        items = payload.get("data") or []
        if not isinstance(items, list):
            return out
        for item in items:
            if not isinstance(item, dict):
                continue
            p1_obj = item.get("player1") or {}
            p2_obj = item.get("player2") or {}
            p1_name = p1_obj.get("name") if isinstance(p1_obj, dict) else None
            p2_name = p2_obj.get("name") if isinstance(p2_obj, dict) else None
            key = _surname_pair_key(p1_name, p2_name)
            if key is None:
                continue
            # `date` may be null on early-round matches whose tipoff
            # tour officials haven't confirmed. Carry None through —
            # the overlay stage skips dateless rows (Kalshi's
            # `occurrence_datetime` stays). The non-date fields
            # (player IDs, tournament, surface, round) are still
            # useful and worth indexing.
            tipoff: datetime | None = None
            raw_date = item.get("date")
            if isinstance(raw_date, str) and raw_date:
                try:
                    tipoff = datetime.fromisoformat(
                        raw_date.replace("Z", "+00:00")
                    )
                except ValueError:
                    tipoff = None
            tourn_obj = item.get("tournament") or {}
            tourn_name = (
                tourn_obj.get("name")
                if isinstance(tourn_obj, dict) else None
            )
            court_obj = (
                tourn_obj.get("court")
                if isinstance(tourn_obj, dict) else None
            )
            # Prefer the courtId path so indoor-hard ("I.hard") collapses
            # to canonical "hard" via `_COURT_ID_TO_SURFACE` — matches the
            # vocabulary `TennisPlayerStats.surface_win_loss` and the
            # tennis sport hint already use. The `include=tournament.court`
            # expansion ships the court as a nested object with `id`/`name`,
            # but flat `courtId` exists on some payload shapes too —
            # check both before falling through to raw lowercased name.
            surface: str | None = None
            if isinstance(court_obj, dict):
                surface = _surface_from_court_id(court_obj.get("id"))
            if surface is None and isinstance(tourn_obj, dict):
                surface = _surface_from_court_id(tourn_obj.get("courtId"))
            if surface is None and isinstance(court_obj, dict):
                court_name = court_obj.get("name")
                if isinstance(court_name, str) and court_name.strip():
                    surface = court_name.strip().lower()
            round_obj = item.get("round") or {}
            round_name = (
                round_obj.get("name")
                if isinstance(round_obj, dict) else None
            )
            p1_id = _coerce_int(item.get("player1Id"))
            p2_id = _coerce_int(item.get("player2Id"))
            fixture = MatchStatsFixture(
                date=tipoff,
                player_a_id=p1_id,
                player_b_id=p2_id,
                player_a_name=p1_name,
                player_b_name=p2_name,
                tournament_name=tourn_name,
                surface=surface,
                round_name=round_name,
            )
            # First match wins on duplicate keys — the vendor occasionally
            # ships the same matchup twice within a tournament when both
            # rounds advance through the bracket; the first (typically
            # earliest) row is the right pick.
            out.setdefault(key, fixture)
            # Same first-write-wins for the cross-call fixture cache so
            # the same-matchup-twice case picks the same fixture both
            # `out` and the cache resolve to. The cache is keyed by
            # `(tour, surname-pair)` so `fetch()` can look up the
            # fixture for a matchup without knowing the date — identity
            # doesn't carry one.
            self._fixture_cache.setdefault((tour, key), fixture)
            # Seed the rankings index with fixture-derived IDs so ITF
            # players outside the top-500 ranking still resolve. Only
            # fill empty slots — never overwrite ranked entries (which
            # carry richer position+points data).
            tour_index = self._index.setdefault(tour, {})
            if p1_id is not None and p1_name:
                tour_index.setdefault(_normalize_name(p1_name), (p1_id, None, None))
            if p2_id is not None and p2_name:
                tour_index.setdefault(_normalize_name(p2_name), (p2_id, None, None))
        return out

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

        # `identity.surface` is set by `tennis_match_identity` from a
        # slug-prefix lookup that only matches Slams + Masters/1000s.
        # For everything else (250s, Challengers, ITF futures, WTA 125s
        # — and even Slams/Masters whose per-match slug format
        # `{tour}-{lastA}-{lastB}-{date}` carries no tournament token)
        # it returns None. The fixtures payload we already fetched at
        # slate-overlay time carries the vendor's `tournament.court.name`
        # for the same matchup — fall back to that. `tournament_hint`
        # gets a parallel upgrade from the cached fixture when the
        # gamma-question prefix didn't supply one.
        surface = identity.surface
        tournament = identity.tournament_hint
        if surface is None or tournament is None:
            pair = _surname_pair_key(identity.player_a, identity.player_b)
            if pair is not None:
                fixture = self._fixture_cache.get((identity.tour, pair))
                if fixture is not None:
                    if surface is None and fixture.surface is not None:
                        surface = fixture.surface
                    if tournament is None and fixture.tournament_name is not None:
                        tournament = fixture.tournament_name

        return TennisStatsContext(
            provider=self.name,
            fetched_at=datetime.now(UTC),
            surface=surface,
            tournament=tournament,
            player_a=player_a,
            player_b=player_b,
            head_to_head=h2h,
        )
