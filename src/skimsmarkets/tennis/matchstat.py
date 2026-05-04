"""Concrete `TennisStatsProvider` for the MatchStat tennis API.

Vendor: <https://tennisapidoc.matchstat.com>. Hosted on RapidAPI under
`tennis-api-atp-wta-itf.p.rapidapi.com`. Auth via two static headers
(`X-RapidAPI-Key`, `X-RapidAPI-Host`) on every request. Rate limit is
100 req/min/IP — generous for our usage (≤ 1 rankings call + 6 per-match
calls × a handful of tennis matches per slate).

Why each endpoint:
  - `/{tour}/ranking/singles`        — name → player_id index. The vendor's
                                       /search endpoint omits IDs (probed),
                                       so rankings is the only path. Top
                                       500 covers virtually all
                                       Polymarket-traded tour singles.
  - `/{tour}/player/profile/{id}`    — `form` array (recent W/L), bio.
                                       Profile does NOT carry points; we
                                       read those from the rankings index
                                       hit instead.
  - `/{tour}/player/surface-summary/{id}` — yearly per-court win/loss.
                                       Most recent year's row gives YTD
                                       totals + per-surface splits in one
                                       payload.
  - `/{tour}/h2h/info/{a}/{b}`       — per-surface H2H counts. We sum
                                       across surfaces for total H2H.
  - `/{tour}/h2h/matches/{a}/{b}`    — reverse-chronological meeting list.
                                       First entry is the most recent
                                       meeting — feeds `last_meeting_*`.

Naming normalization: vendor names sometimes carry diacritics
(e.g. "Cóbolli") that Polymarket strips. We index on a lowercase +
diacritic-stripped form so common labelings cross-match without exact
casing.
"""

from __future__ import annotations

import asyncio
import logging
import unicodedata
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, Self

import httpx

from skimsmarkets.tennis.identity import TennisMatchIdentity
from skimsmarkets.tennis.models import (
    TennisHeadToHead,
    TennisPlayerStats,
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

# `round` and `tournament` get added to `include=` on h2h/matches calls
# so the vendor ships the joined `round.name` (e.g. "Final", "1/2",
# "1/4") and `tournament.courtId` inline rather than forcing a separate
# round-id lookup. Centralised so the include string and the parser
# stay in sync.
_H2H_MATCHES_INCLUDE = "tournament,round"

_RETRY_ATTEMPTS = 3
_RETRY_BASE_S = 1.0


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


def _parse_date(v: Any) -> Any:
    """Vendor ships dates as `2026-04-20T00:00:00.000Z`. The model's
    `_coerce_date` validator accepts the ISO datetime string already; we
    pass the raw value through unchanged.
    """
    return v


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
        """
        if self._client is None:
            raise RuntimeError(
                "MatchStatTennisProvider used outside of `async with` context"
            )
        url = f"{_BASE_URL}{path}"
        for attempt in range(_RETRY_ATTEMPTS):
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
                    try:
                        wait = float(retry_after) if retry_after else _RETRY_BASE_S * (2 ** attempt)
                    except ValueError:
                        wait = _RETRY_BASE_S * (2 ** attempt)
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

    # ----- Per-player fetches -----

    async def _player_profile(
        self, tour: str, pid: int
    ) -> tuple[list[str], int | None]:
        """Returns (form_array_or_empty, best_rank_or_None).

        Single profile call with `include=form` covers two needs: the
        recent W/L array AND career-high ranking — the latter sits at
        `data.bestRank.position`. Free to extract on the same response;
        no extra HTTP. The vendor ships `form` oldest → newest; we
        pass it through unchanged and let the renderer upper-case +
        slice to the most recent N.
        """
        # `include=form,ranking` ships both the recent W/L array and
        # the `bestRank` block on the same response. Without `ranking`
        # the vendor omits `bestRank` even though `curRank` always lands.
        body = await self._get(
            f"/tennis/v2/{tour}/player/profile/{pid}",
            params={"include": "form,ranking"},
        )
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, dict):
            return [], None
        form = data.get("form") or []
        if not isinstance(form, list):
            form = []
        best_rank = None
        best_block = data.get("bestRank")
        if isinstance(best_block, dict):
            best_rank = _coerce_int(best_block.get("position"))
        return [str(x) for x in form if isinstance(x, str)], best_rank

    async def _player_last_match_date(self, tour: str, pid: int) -> Any:
        """Return the date string of the player's most recent match, or None.

        Vendor ships past-matches reverse-chronological; pageSize=1 means
        we pay for one row regardless of the player's match count. The
        date is left as the vendor's ISO string and the model's
        `_coerce_date` validator parses it later.
        """
        body = await self._get(
            f"/tennis/v2/{tour}/player/past-matches/{pid}",
            params={"pageSize": 1},
        )
        rows = body.get("data") if isinstance(body, dict) else None
        if not isinstance(rows, list) or not rows:
            return None
        first = rows[0]
        if not isinstance(first, dict):
            return None
        return first.get("date")

    async def _player_tier_records(
        self, tour: str, pid: int
    ) -> dict[str, tuple[int, int] | None]:
        """Pull current-year W/L vs top-10 + at Slams + at Masters.

        Perf-breakdown ships a year-keyed dict whose value is a 4-axis
        matrix (`court`, `round`, `rank`, `level`). We deliberately
        consume only three cells of one year's slice — the full payload
        is enormous and most cells overlap signal we already have via
        surface-summary or h2h. Year selection: the largest numeric key
        present (vendor sorts unspecified, max() makes us robust to
        order changes). Cells use `aw`/`al` (all wins / all losses) per
        the vendor's convention; the bare `w`/`l` columns track only
        finals.
        """
        body = await self._get(f"/tennis/v2/{tour}/player/perf-breakdown/{pid}")
        data = body.get("data") if isinstance(body, dict) else None
        out: dict[str, tuple[int, int] | None] = {
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

        out["record_vs_top_10"] = _cell("rank", "top10")
        out["record_at_grand_slam"] = _cell("level", "grandSlam")
        out["record_at_masters"] = _cell("level", "masters")
        return out

    async def _player_match_stats(
        self, tour: str, pid: int
    ) -> dict[str, float | None]:
        """Career serve / return / break-point percentages.

        The vendor ships raw counters (numerator + denominator) under
        `serviceStats` and `breakPointsServeStats` / `breakPointsRtnStats`.
        We compute the ratios here so the prompt block carries
        percentages directly — the reasoner shouldn't have to do
        arithmetic on raw counts in-context. Field naming convention:
        `<x>Gm` is the numerator (count of events meeting condition),
        `<x>OfGm` is the denominator (eligible events). Ratios that
        can't be computed (zero denominator, missing fields) come back
        as None and the renderer suppresses those lines.

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

        bp_srv = data.get("breakPointsServeStats") if isinstance(data.get("breakPointsServeStats"), dict) else {}
        out["break_point_save_pct"] = _ratio(
            bp_srv.get("breakPointSavedGm"), bp_srv.get("breakPointFacedGm")
        )

        bp_rtn = data.get("breakPointsRtnStats") if isinstance(data.get("breakPointsRtnStats"), dict) else {}
        out["break_point_convert_pct"] = _ratio(
            bp_rtn.get("breakPointWonGm"), bp_rtn.get("breakPointChanceGm")
        )
        return out

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

        # All five per-player endpoints are independent — fan them out
        # so each player's full block lands in one round-trip's worth
        # of wall time rather than five sequential ones. The outer
        # `fetch` ALSO gathers across players, so two-player wall time
        # is bounded by the slowest single call. Total cost per match:
        # 5 (this player) + 5 (other player) + 2 (h2h info+matches) +
        # 1 (h2h stats) = 13 + 5 = 18 calls. With the upgraded 5/sec
        # ceiling that's ~3.6s of vendor budget per match, fully
        # parallel → ~1s wall clock.
        (
            (form_arr, best_rank),
            (ytd_pair, surfaces),
            match_stats,
            tier_records,
            last_match_raw,
        ) = await asyncio.gather(
            self._player_profile(tour, pid),
            self._player_surface_year_record(tour, pid),
            self._player_match_stats(tour, pid),
            self._player_tier_records(tour, pid),
            self._player_last_match_date(tour, pid),
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
            ytd_win_loss=ytd_pair,
            surface_win_loss=surfaces,
            last_10_form=last_10_form or None,
            last_match_date=last_match_raw,
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
        """Fetch overall H2H counts + the most recent meeting.

        h2h/info returns per-surface rows; we sum across surfaces for
        totals. h2h/matches is reverse-chronological — the first entry
        is the latest meeting, fed into `last_meeting_*` for the prompt.
        Empty H2H (no prior meetings) returns `(0, 0)` rather than None,
        and the renderer suppresses the section when both counts are
        zero (via `has_actionable_signal`).
        """
        # /h2h/stats is the matchup-conditioned-stats endpoint —
        # firstSetWin/Lose-MatchWin %, decidingSet record, tiebreak
        # record across all prior meetings between this exact pair.
        # Fanned out alongside info+matches; one extra call total.
        info_body, matches_body, stats_body = await asyncio.gather(
            self._get(f"/tennis/v2/{tour}/h2h/info/{a_id}/{b_id}"),
            self._get(
                f"/tennis/v2/{tour}/h2h/matches/{a_id}/{b_id}",
                params={"pageSize": 1, "include": _H2H_MATCHES_INCLUDE},
            ),
            self._get(f"/tennis/v2/{tour}/h2h/stats/{a_id}/{b_id}"),
        )

        a_wins = 0
        b_wins = 0
        info_rows = info_body.get("data") if isinstance(info_body, dict) else None
        if isinstance(info_rows, list):
            for row in info_rows:
                if not isinstance(row, dict):
                    continue
                a_wins += _coerce_int(row.get("player1wins")) or 0
                b_wins += _coerce_int(row.get("player2wins")) or 0

        last_meeting = None
        last_winner = None
        last_surface = None
        last_round = None
        last_result = None
        match_rows = matches_body.get("data") if isinstance(matches_body, dict) else None
        if isinstance(match_rows, list) and match_rows:
            first = match_rows[0]
            if isinstance(first, dict):
                last_meeting = _parse_date(first.get("date"))
                # `match_winner` is the winner's player id. h2h/matches
                # always comes back keyed (player1, player2) = (a, b),
                # but we still compare on id rather than position to
                # stay defensive.
                winner_id = _coerce_int(first.get("match_winner"))
                if winner_id == a_id:
                    last_winner = name_a
                elif winner_id == b_id:
                    last_winner = name_b
                tourn = first.get("tournament")
                if isinstance(tourn, dict):
                    cid = _coerce_int(tourn.get("courtId"))
                    if cid is not None:
                        last_surface = _COURT_ID_TO_SURFACE.get(cid)
                # `round.name` ships joined when `include=round`. Vendor
                # uses fraction shorthand for late rounds ("1/2" = SF,
                # "1/4" = QF) — pass through unchanged so anyone
                # familiar with tour notation reads it natively.
                rnd = first.get("round")
                if isinstance(rnd, dict):
                    rname = rnd.get("name")
                    if isinstance(rname, str) and rname.strip():
                        last_round = rname.strip()
                # Score line as the vendor ships it (e.g. "6-4 6-2",
                # "7-6(5) 6-3 4-6 6-2"). Distinguishing a straight-sets
                # win from a five-setter the same player won is itself
                # a form signal.
                result = first.get("result")
                if isinstance(result, str) and result.strip():
                    last_result = result.strip()

        # Matchup-specific clutch records from /h2h/stats. Vendor's
        # `data.player1Stats` corresponds to player_a (the IDs in the
        # URL path are positional). Fields live on the nested per-player
        # blocks. We pull (wins, total) pairs rather than the
        # vendor-supplied percentage so the prompt shows sample size,
        # which the reasoner needs to weight 1-of-1 vs 5-of-7
        # appropriately.
        decider_a = decider_b = None
        tiebreak_a = tiebreak_b = None
        comeback_a = comeback_b = None
        stats_data = stats_body.get("data") if isinstance(stats_body, dict) else None
        if isinstance(stats_data, dict):
            p1 = stats_data.get("player1Stats")
            p2 = stats_data.get("player2Stats")

            def _wins_total(
                block: Any, win_key: str, total_key: str
            ) -> tuple[int, int] | None:
                if not isinstance(block, dict):
                    return None
                wins = _coerce_int(block.get(win_key))
                total = _coerce_int(block.get(total_key))
                if wins is None or total is None or total <= 0:
                    return None
                return (wins, total)

            def _pct(block: Any, key: str) -> float | None:
                if not isinstance(block, dict):
                    return None
                v = block.get(key)
                if v is None:
                    return None
                # Vendor ships percentages as integers 0–100; we store
                # them as ratios in [0, 1] for consistency with the
                # career serve/return percentages elsewhere.
                f = _coerce_int(v)
                if f is None:
                    return None
                return f / 100.0

            decider_a = _wins_total(p1, "decidingSetWin", "decidingSetCount")
            decider_b = _wins_total(p2, "decidingSetWin", "decidingSetCount")
            tiebreak_a = _wins_total(p1, "tiebreakWon", "tiebreakCount")
            tiebreak_b = _wins_total(p2, "tiebreakWon", "tiebreakCount")
            comeback_a = _pct(p1, "firstSetLoseMatchWinPercentage")
            comeback_b = _pct(p2, "firstSetLoseMatchWinPercentage")

        # Suppress only when literally nothing was found — a populated
        # stats block alone is enough signal even without h2h/info.
        if (
            a_wins == 0 and b_wins == 0
            and last_meeting is None
            and decider_a is None and decider_b is None
        ):
            return None

        return TennisHeadToHead(
            a_wins=a_wins,
            b_wins=b_wins,
            last_meeting=last_meeting,
            last_meeting_winner=last_winner,
            last_meeting_surface=last_surface,
            last_meeting_round=last_round,
            last_meeting_result=last_result,
            decider_record_a=decider_a,
            decider_record_b=decider_b,
            tiebreak_record_a=tiebreak_a,
            tiebreak_record_b=tiebreak_b,
            first_set_lost_match_won_pct_a=comeback_a,
            first_set_lost_match_won_pct_b=comeback_b,
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
