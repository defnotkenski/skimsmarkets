"""Kalshi-sourced slate builder — replaces Polymarket gamma `/events`.

Discovers tennis match-winner series via `KalshiClient.list_tennis_
match_series()` (with hardcoded fallback to `cfg.KALSHI_TENNIS_SERIES_
TICKERS`), lists events per series, and adapts each Kalshi event into
a `PolymarketEvent` so the rest of the pipeline (selection, lens
dispatch, director, judge, JSONL persistence, retro grading) keeps
working unchanged.

The adapter is the architectural pivot of the Polymarket → Kalshi
data swap: `PolymarketEvent` stays as the venue-neutral pipeline
event type, and `_kalshi_event_to_polymarket_event` is the only
place that knows the source is Kalshi. Polymarket-specific fields
(`context_description`, `team_record`, gamma piggyback) are left as
None — director and lens code already gate on `is not None` for
each.

NO-side semantics differ from Polymarket. Kalshi exposes BOTH YES
sides as independent books (one `KalshiMarket` per player, each with
its own bid/ask and depth). The adapter constructs ONE
`PolymarketMarket` per `KalshiMarket` reading prices/depth directly
from that side's native book — no `inverted_no_side` call. The
favorite (higher mid) is `is_no_side=False`; the other is
`is_no_side=True` so the director's NO-side branch at
`agents/director.py:64` still fires correctly.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from skimsmarkets import config as cfg
from skimsmarkets.kalshi.client import KalshiClient
from skimsmarkets.kalshi.models import KalshiEvent, KalshiMarket
from skimsmarkets.polymarket.models import PolymarketEvent, PolymarketMarket
from skimsmarkets.tennis.matchstat import _normalize_name

log = logging.getLogger(__name__)


# --- public dataclass shared with the legacy gamma path ---------------------
# Kept as a tiny mirror of `pipeline.SlateOptions` so this module doesn't
# import from pipeline (which would cycle). pipeline.fetch_slate constructs
# the options on its side.
@dataclass(frozen=True)
class KalshiSlateOptions:
    horizon_hours: int
    max_implied_probability: float = cfg.MAX_IMPLIED_PROBABILITY
    min_open_interest_dollars: float = cfg.MIN_OPEN_INTEREST_DOLLARS


# --- main entry point -------------------------------------------------------


async def fetch_kalshi_slate(
    client: KalshiClient,
    opts: KalshiSlateOptions,
) -> list[PolymarketEvent]:
    """Build the tennis slate from Kalshi `/events`, returned as
    `PolymarketEvent` objects so the downstream pipeline is unchanged.

    Pipeline:
      1. Series discovery via `list_tennis_match_series()`. Falls back
         to `cfg.KALSHI_TENNIS_SERIES_TICKERS` on empty/raise so a
         Kalshi `/series` outage doesn't kill the whole slate path.
      2. `list_events(series, status="open")` per series, in parallel.
      3. `_kalshi_event_to_polymarket_event` per event: synthesises an
         `atp-`/`wta-`-prefixed slug + tournament so
         `tennis/identity.py` surface/tier lookups can fire.
      4. Tradability filter: drop markets without yes_sub_title or
         bid/ask. Drop events that lose all their markets to this.
      5. No-tipoff filter: drop events whose every market lacks
         `occurrence_datetime` (the downstream MatchStats overlay
         needs a date to derive the fixtures lookup key).
      6. Blowout filter: drop events whose favorite mid (highest
         `(bid+ask)/2` across markets) ≥ `max_implied_probability`.
      7. Sort by tipoff ascending.

    **Horizon filtering moved downstream.** It now runs in
    `pipeline.apply_horizon_filter` AFTER `pipeline.overlay_matchstats_
    tipoffs` so the cut uses MatchStats's per-match precision when
    available, not Kalshi's session-bucketed `occurrence_datetime`
    (which can be 3-9h off for tournament-evening matches).
    `MAX_SLATE_EVENTS` truncation continues to live downstream in
    `selection.select_top_events`.
    """
    try:
        series_tickers = await client.list_tennis_match_series()
    except Exception as e:  # noqa: BLE001
        log.warning("kalshi /series discovery failed (%s); falling back", e)
        series_tickers = []
    if not series_tickers:
        series_tickers = list(cfg.KALSHI_TENNIS_SERIES_TICKERS)
        log.info(
            "kalshi /series returned no tennis series — using fallback %s",
            series_tickers,
        )

    series_results = await asyncio.gather(
        *(client.list_events(series_ticker=s, status="open") for s in series_tickers),
        return_exceptions=True,
    )
    raw_events: list[KalshiEvent] = []
    for ticker, result in zip(series_tickers, series_results, strict=True):
        if isinstance(result, BaseException):
            log.warning("kalshi list_events series=%s failed: %s", ticker, result)
            continue
        raw_events.extend(result)
    log.info(
        "fetched %d kalshi events across %d series",
        len(raw_events),
        len(series_tickers),
    )

    kept: list[PolymarketEvent] = []
    dropped_blowout = 0
    dropped_no_tipoff = 0
    dropped_no_tradable = 0
    dropped_thin_oi = 0
    for raw in raw_events:
        ev = _kalshi_event_to_polymarket_event(raw)
        if ev is None:
            continue
        # `game_start_time` presence is required even though the
        # actual horizon-cut moves to a downstream pipeline stage
        # (`pipeline.apply_horizon_filter`). The MatchStats overlay
        # stage between this and the horizon filter needs SOME date
        # to derive `(tour, date_iso)` for the fixtures lookup.
        starts = [t for m in ev.markets if (t := m.game_start_time) is not None]
        if not starts:
            dropped_no_tipoff += 1
            continue
        live_markets = [
            m
            for m in ev.markets
            if m.yes_sub_title
            and m.yes_bid_dollars is not None
            and m.yes_ask_dollars is not None
        ]
        if not live_markets:
            dropped_no_tradable += 1
            continue
        favorite_mid = max(
            (m.yes_bid_dollars + m.yes_ask_dollars) / 2.0  # type: ignore[operator]
            for m in live_markets
        )
        if favorite_mid >= opts.max_implied_probability:
            dropped_blowout += 1
            continue
        # OI floor — drop events where ANY side has thin resting
        # interest. Compares MIN OI across markets because the
        # ranker may predict either side and the trader needs ask-
        # side liquidity on whichever side gets the buy. An event
        # with one well-funded side and one cold side fails this
        # check (you can't reliably trade the cold side even if the
        # other half is healthy). Kalshi reports OI in contracts
        # which equals dollars-at-par for binary markets, matching
        # `open_interest_dollars`. None treated as 0 — fresh markets
        # without OI yet shouldn't bypass the floor.
        if opts.min_open_interest_dollars > 0:
            min_oi = min(
                (m.open_interest_dollars or 0.0) for m in live_markets
            )
            if min_oi < opts.min_open_interest_dollars:
                dropped_thin_oi += 1
                continue
        if len(live_markets) != len(ev.markets):
            ev = ev.model_copy(update={"markets": live_markets})
        kept.append(ev)

    log.info(
        "kept %d kalshi events after tradability + blowout (>=%.2f) + "
        "min_oi (>=$%.0f) filters; dropped blowouts=%d, no-tipoff=%d, "
        "no-tradable=%d, thin-oi=%d "
        "(horizon filter runs downstream after matchstats overlay)",
        len(kept),
        opts.max_implied_probability,
        opts.min_open_interest_dollars,
        dropped_blowout,
        dropped_no_tipoff,
        dropped_no_tradable,
        dropped_thin_oi,
    )

    _far_future = datetime.max.replace(tzinfo=UTC)

    def _tipoff(ev: PolymarketEvent) -> datetime:
        starts = [t for m in ev.markets if (t := m.game_start_time) is not None]
        return min(starts) if starts else _far_future

    kept.sort(key=_tipoff)
    return kept


# --- adapter ---------------------------------------------------------------

# Kalshi series-ticker → tour mapping for the slug prefix. Slug must
# start with `atp-` / `wta-` so `tennis/identity.py:_tour_from_slug`
# recognises the tour and the entire tennis sport gate fires correctly.
#
# ITF futures are folded into atp/wta based on gender — same
# convention MatchStats uses (M-events under /atp/fixtures/, W-events
# under /wta/fixtures/). This keeps a single tennis sport gate and
# reuses the existing rankings indexes; ITF players outside the top
# ATP/WTA rankings simply degrade gracefully (lookup_player_rank
# returns None and selection scoring falls back to other signals).
#
# **ORDER MATTERS**: `KXITFW` MUST appear before `KXITF` because
# `"KXITFWMATCH".startswith("KXITF")` is True — the longer prefix
# has to match first or women's events get misclassified as men's.
_TOUR_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("KXATP", "atp"),
    ("KXWTA", "wta"),
    ("KXITFW", "wta"),  # women's ITF futures (W15/W25/W35/W75) — MUST be before KXITF
    ("KXITF", "atp"),   # men's ITF futures (M15/M25/M35)
)


def _kalshi_event_to_polymarket_event(
    raw: KalshiEvent,
) -> PolymarketEvent | None:
    """Construct a `PolymarketEvent` from one Kalshi `KalshiEvent`.

    Returns None when the event lacks the minimum we need (≥2 markets
    with names, recognisable tour). Per-market `is_no_side` is set so
    the higher-mid market is the YES (favorite) and the other is the
    NO (underdog) — matches `tennis/identity.py:_favorite_market_index`
    semantics.
    """
    tour = _tour_from_series_ticker(raw.series_ticker)
    if tour is None:
        return None
    if len(raw.markets) < 2:
        return None

    competition = (
        raw.product_metadata.competition
        if raw.product_metadata is not None
        else None
    )

    # Per-market mids for favorite / underdog assignment. We need both
    # bids+asks so a partial market with only one side quoted gets
    # treated as missing (skipped at filter time).
    mids: list[tuple[int, float | None]] = []
    for i, m in enumerate(raw.markets):
        if m.yes_bid_dollars is None or m.yes_ask_dollars is None:
            mids.append((i, None))
            continue
        mids.append((i, (m.yes_bid_dollars + m.yes_ask_dollars) / 2.0))
    valid_mids = [(i, p) for i, p in mids if p is not None]
    if len(valid_mids) < 2:
        return None
    valid_mids.sort(key=lambda pair: pair[1] or 0.0, reverse=True)
    favorite_idx = valid_mids[0][0]

    # Earliest market tipoff anchors the slug date. Falls back to
    # `now` only if every market lacks `occurrence_datetime` — that
    # event would also fail the horizon filter upstream, so the
    # synthesised slug is best-effort.
    tipoffs = [
        m.occurrence_datetime
        for m in raw.markets
        if m.occurrence_datetime is not None
    ]
    earliest = min(tipoffs) if tipoffs else None
    date_part = earliest.strftime("%Y-%m-%d") if earliest else "unknown"

    # Surnames for the slug. Use the FAVORITE'S surname first, then the
    # other, so the slug reads as `{tour}-{tournament}-{fav}-{dog}-
    # {date}` — same ordering convention `tennis_match_identity` uses
    # downstream (player_a = favorite).
    fav_surname = _surname_from_yes_sub_title(raw.markets[favorite_idx].yes_sub_title)
    dog_surname = _surname_from_yes_sub_title(
        next(
            (
                m.yes_sub_title
                for i, m in enumerate(raw.markets)
                if i != favorite_idx and m.yes_sub_title
            ),
            None,
        )
    )
    if not fav_surname or not dog_surname:
        return None

    tournament_key = _slugify_competition(competition, tour)
    slug_parts = [tour]
    if tournament_key:
        slug_parts.append(tournament_key)
    slug_parts.extend([fav_surname, dog_surname, date_part])
    synth_slug = "-".join(slug_parts)

    series_slug: str | None
    if competition:
        series_slug = competition.lower().replace(" ", "-")
    else:
        series_slug = tour

    # Per-market title format mirrors the H2H pattern
    # `_parse_h2h_question` expects (`"Tournament: A vs B"`) so
    # `tennis/identity.py:tennis_match_identity` can extract a
    # `tournament_hint`.
    if competition and raw.title:
        market_title = f"{competition}: {raw.title}"
    else:
        market_title = raw.title

    markets: list[PolymarketMarket] = []
    for i, km in enumerate(raw.markets):
        pm = _kalshi_market_to_polymarket_market(
            km,
            event_title=market_title,
            is_no_side=(i != favorite_idx),
        )
        if pm is not None:
            markets.append(pm)
    if len(markets) < 2:
        return None

    return PolymarketEvent(
        id=raw.event_ticker,
        slug=synth_slug,
        title=raw.title,
        category="sports",
        series_slug=series_slug,
        active=True,
        closed=False,
        sport_type="tennis",
        markets=markets,
    )


def _kalshi_market_to_polymarket_market(
    km: KalshiMarket,
    *,
    event_title: str | None,
    is_no_side: bool,
) -> PolymarketMarket | None:
    """Construct one `PolymarketMarket` from one `KalshiMarket`.

    The market's `slug` and `id` are the Kalshi ticker — used as the
    join key for `enrich_kalshi_book` and `enrich_kalshi_history`.
    Reads bid/ask + sizes from this side's native YES book (Kalshi
    exposes both sides independently — no inversion needed).
    """
    if not km.yes_sub_title:
        return None
    one_day_change: float | None = None
    if km.last_price_dollars is not None and km.previous_price_dollars is not None:
        one_day_change = km.last_price_dollars - km.previous_price_dollars
    # Kalshi reports volume + open interest in CONTRACTS, not dollars.
    # Convert to USD for the renderer + downstream consumers that
    # expect dollar magnitudes (mirrors Polymarket's `volume_dollars`
    # = sharesTraded × reference_price semantic).
    #   - Volume USD ≈ contracts × current_mid (best-effort proxy for
    #     `Σ price_at_fill × qty` since per-trade prices aren't on
    #     `/events`). When mid is missing the slate filter would
    #     already have dropped this market — defensive None below.
    #   - Open interest in dollars uses the par-value convention
    #     (1 contract = $1 of YES-side liability if it resolves YES),
    #     matching Polymarket's `liquidity_dollars` framing.
    mid: float | None = None
    if km.yes_bid_dollars is not None and km.yes_ask_dollars is not None:
        mid = (km.yes_bid_dollars + km.yes_ask_dollars) / 2.0
    volume_dollars_24h = (
        km.volume_24h_fp * mid
        if km.volume_24h_fp is not None and mid is not None
        else None
    )
    return PolymarketMarket(
        slug=km.ticker,
        id=km.ticker,
        title=event_title,
        yes_sub_title=km.yes_sub_title,
        team_aliases=[km.yes_sub_title],
        sports_market_type="moneyline",
        is_no_side=is_no_side,
        yes_bid_dollars=km.yes_bid_dollars,
        yes_ask_dollars=km.yes_ask_dollars,
        last_trade_price_dollars=km.last_price_dollars,
        # `yes_bid_size_top` is the contract count at top-of-book.
        # `enrich_kalshi_book` overrides this with full-book numbers
        # later, but the listing payload's `yes_bid_size_fp` already
        # gives a good first-pass for events the book enrichment
        # might fail on.
        yes_bid_size_top=km.yes_bid_size_fp,
        yes_ask_size_top=km.yes_ask_size_fp,
        # `volume_dollars` is what the leaderboard renderer reads;
        # `notional_traded_dollars` is the canonical truth-source on
        # PolymarketMarket and falls back to volume_dollars on the
        # gamma path. Populate both with the 24h figure so both
        # consumers see the same number (Kalshi has no separate
        # "lifetime notional" stream).
        volume_dollars=volume_dollars_24h,
        notional_traded_dollars=volume_dollars_24h,
        liquidity_dollars=km.liquidity_dollars,
        # `open_interest_fp` is in contracts; for binary Kalshi
        # markets each contract has $1 par value, so the contract
        # count equals the total $ resting at par (matches the
        # Polymarket gamma `liquidity` convention, NOT the actual
        # cost-basis at current price).
        open_interest_dollars=km.open_interest_fp,
        # Repurpose the gamma_one_day_price_change slot for Kalshi's
        # last - previous diff. Director rendering reads this field by
        # name; the source is documented in the JSONL schema and via
        # CLAUDE.md's role-split note.
        gamma_one_day_price_change=one_day_change,
        market_state=_kalshi_status_to_market_state(km.status),
        game_start_time=km.occurrence_datetime,
    )


# --- helpers ---------------------------------------------------------------


def _tour_from_series_ticker(series_ticker: str | None) -> str | None:
    if not series_ticker:
        return None
    upper = series_ticker.upper()
    for prefix, tour in _TOUR_BY_PREFIX:
        if upper.startswith(prefix):
            return tour
    return None


# Stripping pattern: leading "atp", "wta", "atp tour", "wta tour"
# (case-insensitive) plus any whitespace following. Run after
# lowercase, so the regex is anchored on lowercase tokens.
_TOUR_PREFIX_RE = re.compile(r"^(atp|wta)(\s+tour)?\s+")
# Anything that isn't alnum or dash gets collapsed to a dash, then
# repeated dashes collapse and trailing dashes strip.
_NON_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify_competition(competition: str | None, tour: str) -> str | None:
    """Turn `"ATP Rome"` into `"rome"` so the slug becomes
    `atp-rome-...` and `tennis/identity.py:_slug_surface` can match.

    Strips the leading tour token (and "Tour " variants) before
    slugifying so the final slug doesn't carry `atp-atp-...`.
    Returns None if competition is empty or strips to empty.
    """
    if not competition:
        return None
    text = competition.strip().lower()
    text = _TOUR_PREFIX_RE.sub("", text)
    text = _NON_SLUG_RE.sub("-", text).strip("-")
    return text or None


def _surname_from_yes_sub_title(name: str | None) -> str | None:
    """Last whitespace-separated token, normalized via
    `tennis/matchstat.py:_normalize_name`. Mirrors the surname-
    extraction logic in `kalshi/matcher.py:last_token` so the synth
    slug uses the same player key the matcher and MatchStats lookups
    use.
    """
    if not name:
        return None
    norm = _normalize_name(name)
    if not norm:
        return None
    last = norm.split()[-1] if norm.split() else norm
    last = _NON_SLUG_RE.sub("-", last).strip("-")
    return last or None


def _kalshi_status_to_market_state(status: str | None) -> str | None:
    """Map Kalshi's free-form `status` string to the
    `MARKET_STATE_*` enum the director rendering expects.
    """
    if not status:
        return None
    s = status.strip().lower()
    if s == "active":
        return "MARKET_STATE_OPEN"
    if s in ("paused", "halted"):
        return "MARKET_STATE_HALTED"
    if s in ("closed", "expired", "settled"):
        return "MARKET_STATE_EXPIRED"
    return f"MARKET_STATE_{s.upper()}"
