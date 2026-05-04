from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from anthropic import AsyncAnthropic

from skimsmarkets import config as cfg
from skimsmarkets.agents.director import synthesize_prediction
from skimsmarkets.agents.fetchers import FetcherProvider, build_provider
from skimsmarkets.agents.judge import judge_slate
from skimsmarkets.agents.reasoners import REASONERS
from skimsmarkets.agents.schemas import (
    DefensibilityAssessment,
    LensNotebook,
    MarketPrediction,
    SpecialistReport,
)
from skimsmarkets.clob import (
    fetch_book,
    fetch_price_history,
    invert_sparkline,
    summarize_book,
    summarize_history,
)
from skimsmarkets.polymarket.models import PolymarketEvent
from skimsmarkets.tennis import TennisStatsContext
from skimsmarkets.tennis.identity import tennis_match_identity
from skimsmarkets.tennis.provider import (
    TennisStatsProvider,
    build_tennis_provider,
)
from skimsmarkets.unusual_whales import (
    GammaTokenResolver,
    UnusualWhalesClient,
    fetch_gamma_event,
    list_gamma_events,
)

log = logging.getLogger(__name__)


@dataclass
class ErrorRecord:
    event_id: str
    stage: str  # "fetcher:<lens>" / "reasoner:<lens>" / "director"
    error: str


@dataclass
class RunResult:
    run_id: str
    # Fetcher provider name + model id captured at run start. Persisted
    # to every JSONL row so retrospective A/B grading can group hit-rate
    # by provider / model version with a one-line jq filter. Defaults so
    # callers / tests that build RunResult directly don't need to set
    # them; `run_pipeline` always overwrites both.
    fetcher_provider: str = ""
    fetcher_model: str = ""
    predictions: list[MarketPrediction] = field(default_factory=list)
    errors: list[ErrorRecord] = field(default_factory=list)
    fetched_events: int = 0
    considered_events: int = 0
    # Per-event notebooks (Stage A — fetcher) and reasoner reports
    # (Stage B — Claude) keyed event_id → lens_name → object. Persisted
    # alongside the final MarketPrediction to JSONL so retrospective
    # grading can ask "did the fetcher find the right facts?" and "did
    # Claude reason correctly?" as separate questions.
    notebooks: dict[str, dict[str, LensNotebook]] = field(default_factory=dict)
    reports: dict[str, dict[str, SpecialistReport]] = field(default_factory=dict)
    # Per-event tennis stats vendor payload (when present). Keyed
    # event_id → TennisStatsContext. Populated in `enrich_tennis_stats`
    # for ATP/WTA singles head-to-heads only; non-tennis / no-key runs
    # leave this empty. Persisted as a top-level JSONL field next to
    # `notebooks` / `specialist_reports` so retro grading can ask "did
    # the API have the right facts?" separately from "did the fetcher
    # use them?".
    tennis_stats: dict[str, TennisStatsContext] = field(default_factory=dict)
    # Slate-level judge output keyed event_id → DefensibilityAssessment.
    # Populated by `judge_slate` after all per-event directors finish; left
    # empty when the judge call fails (leaderboard then falls back to the
    # legacy predicted-probability sort). Same persistence posture as
    # `notebooks` / `reports` — best-effort, never aborts a run.
    defensibility_assessments: dict[str, DefensibilityAssessment] = field(
        default_factory=dict
    )


@dataclass
class _LensOutcome:
    """Internal: per-lens result of one event's two-stage chain."""

    lens: str
    notebook: LensNotebook | None = None
    report: SpecialistReport | None = None
    error_stage: str | None = None  # "fetcher" or "reasoner"
    error: BaseException | None = None


@dataclass(frozen=True)
class SlateOptions:
    """Inputs shared by every slate-building entry point — both the
    `skims fetch` CLI path and `run_pipeline`'s own slate stage. Frozen so
    callers can pass the same instance through multiple stages without
    worrying about mutation.

    `leagues`, `slugs`, and `sports` default to empty lists rather than
    None because every callsite already normalizes the "no filter" case
    to an empty iterable; one less branch downstream. Empty `leagues` =
    no league filter (browse all sports). Empty `sports` = use gamma's
    umbrella `tag_slug=sports` (current default).

    `sports` filters at the gamma API layer via `tag_slug=<sport>` — one
    gamma query per sport, fanned out and unioned. Common values:
    `tennis`, `soccer`, `nba`, `mma`, `ufc`, `mlb`, `wnba`, `ice-hockey`.
    Different mechanic from `leagues`, which is a client-side slug-
    prefix filter applied AFTER the listing call.
    """

    leagues: list[str] = field(default_factory=list)
    slugs: list[str] = field(default_factory=list)
    sports: list[str] = field(default_factory=list)
    horizon_hours: int = cfg.DEFAULT_HORIZON_HOURS


async def fetch_gamma_slate(
    http: httpx.AsyncClient,
    leagues: list[str],
    horizon_hours: int,
    *,
    sports: list[str] | None = None,
) -> list[PolymarketEvent]:
    """Fetch the Polymarket sports slate from gamma-api.

    Single source of truth for "what's in today's slate" — both the `skims
    fetch` display path and the full pipeline iterate exactly the events
    returned here. Filters layered top to bottom:

    1. **Bulk listing.** `list_gamma_events` pages through gamma's
       `/events?tag_slug=<tag>&order=endDate&ascending=true` for upcoming
       events soonest-first. Pagination is necessary because esports
       (cs2, lol, dota2) and high-volume markets crowd out actual sports
       leagues in page 1. When `sports` is non-empty, one listing call
       fans out per sport tag (gamma's `tag_slug` query param accepts
       only a single value), and the per-sport payloads are unioned and
       deduped by event slug. Empty `sports` falls back to gamma's
       umbrella `tag_slug=sports`.
    2. **Variant-bundle drop.** `PolymarketEvent.from_gamma` skips
       `-more-markets` / `-halftime-result` / `-exact-score` /
       `-total-corners` / `-player-props` event-level variants and
       non-moneyline market slugs (`-spread-`, `-total-`, `-set-handicap-`,
       `-points-`, `-1h-...`, etc.) inline.
    3. **League prefix filter.** When `leagues` is non-empty, keep only
       events whose slug starts with `<league>-` for any of them. Empty
       list = no filter (browse all sports). Anchored on the dash so
       `arg` doesn't accidentally swallow `argf-`.
    4. **Horizon time window.** Keep events whose earliest market
       `gameStartTime` falls within `[now - 6h, now + horizon_hours]`.
       The 6h backstop catches long-tail endings (overtime, weather
       delays) that haven't settled yet. **Critical:** filter on
       per-market `gameStartTime`, NOT event `endDate` — `endDate` is
       frozen at market creation and lags rescheduled fixtures by days.
    5. **Tradability filter.** Drop markets without `yes_sub_title` or
       bid/ask. `from_gamma` already pre-filters bid/ask presence, so
       this is a belt-and-suspenders check that mostly catches
       label-less futures placeholders.

    The CLOB book + price-history enrichment stages run on the unified
    slate post-`fetch_slate`, not here — keeps `fetch_gamma_slate` a pure
    fetch+filter without HTTP fan-out beyond the listing call.
    """
    now = datetime.now(tz=UTC)
    horizon_start = now - timedelta(hours=6)
    horizon_end = now + timedelta(hours=horizon_hours)

    sports = sports or []
    if sports:
        # Gamma's `tag_slug` query param is single-valued — fan out one
        # listing per sport tag and union the payloads. Dedupe by event
        # slug because tags overlap (e.g. `ufc` ⊂ `mma`).
        page_lists = await asyncio.gather(
            *(list_gamma_events(http, tag_slug=s) for s in sports)
        )
        seen: set[str] = set()
        payloads: list[dict] = []
        for plist in page_lists:
            for p in plist:
                slug = p.get("slug")
                if not isinstance(slug, str) or slug in seen:
                    continue
                seen.add(slug)
                payloads.append(p)
        log.info(
            "fetched %d gamma payloads across %d sport tag(s) [%s] "
            "(leagues=%s, horizon=%sh)",
            len(payloads),
            len(sports),
            ",".join(sports),
            leagues or "all",
            horizon_hours,
        )
    else:
        payloads = await list_gamma_events(http)
        log.info(
            "fetched %d gamma payloads (leagues=%s, horizon=%sh)",
            len(payloads),
            leagues or "all",
            horizon_hours,
        )

    # League prefixes are anchored on dash so `arg` doesn't swallow `argf-`.
    league_prefixes = [f"{p}-" for p in leagues]

    kept: list[PolymarketEvent] = []
    dropped_blowout = 0
    for payload in payloads:
        slug = payload.get("slug")
        if not isinstance(slug, str) or not slug:
            continue
        if league_prefixes and not any(slug.startswith(p) for p in league_prefixes):
            continue
        ev = PolymarketEvent.from_gamma(payload)
        if ev is None:
            # `from_gamma` already drops -more-markets variants, settled
            # markets, etc. Silently skip those.
            continue
        starts = [t for m in ev.markets if (t := m.game_start_time) is not None]
        if starts:
            tipoff = min(starts)
            if not (horizon_start <= tipoff <= horizon_end):
                continue
        # Belt-and-suspenders tradability filter — `from_gamma` already
        # drops bid/ask=None markets, but a missing `yes_sub_title` would
        # leave the LLM unable to identify which side is which.
        live_markets = [
            m
            for m in ev.markets
            if m.yes_sub_title
            and m.yes_bid_dollars is not None
            and m.yes_ask_dollars is not None
        ]
        if not live_markets:
            continue
        # Blowout filter — drop events whose favorite is priced at or above
        # `MAX_IMPLIED_PROBABILITY` on the YES mid. Mid is the cleanest
        # consensus implied prob; `max` across markets identifies the
        # favorite uniformly across binary head-to-heads (max of YES + NO
        # clone) and 3-way soccer (max of home/draw/away). No ranking
        # signal at 99% — pure LLM-spend waste.
        favorite_mid = max(
            (m.yes_bid_dollars + m.yes_ask_dollars) / 2.0  # type: ignore[operator]
            for m in live_markets
        )
        if favorite_mid >= cfg.MAX_IMPLIED_PROBABILITY:
            dropped_blowout += 1
            continue
        if len(live_markets) != len(ev.markets):
            ev = ev.model_copy(update={"markets": live_markets})
        kept.append(ev)

    log.info(
        "kept %d gamma events after league + horizon + tradability + "
        "blowout (>=%.2f) filters; dropped %d as blowouts",
        len(kept),
        cfg.MAX_IMPLIED_PROBABILITY,
        dropped_blowout,
    )

    # Sort by earliest market tipoff ascending, then cap to
    # `MAX_SLATE_EVENTS`. Sort is unconditional because gamma's listing is
    # ordered by `endDate` (settlement window), not tipoff — the two
    # diverge on tours like ATP where settlement lags match end by days,
    # so a naive head-slice would pick an arbitrary subset rather than
    # "the soonest games." Events without any market `game_start_time`
    # sort last so they don't displace tradable events at the head.
    _far_future = datetime.max.replace(tzinfo=UTC)

    def _tipoff(ev: PolymarketEvent) -> datetime:
        starts = [t for m in ev.markets if (t := m.game_start_time) is not None]
        return min(starts) if starts else _far_future

    kept.sort(key=_tipoff)
    if len(kept) > cfg.MAX_SLATE_EVENTS:
        truncated = len(kept) - cfg.MAX_SLATE_EVENTS
        kept = kept[: cfg.MAX_SLATE_EVENTS]
        log.info(
            "capped slate to %d soonest-tipoff events (truncated %d)",
            cfg.MAX_SLATE_EVENTS,
            truncated,
        )
    return kept


# noinspection PyUnusedLocal
async def fetch_gamma_events(
    http: httpx.AsyncClient,
    slugs: list[str],
    horizon_hours: int,  # noqa: ARG001 — kept for signature symmetry with fetch_gamma_slate
    sem: asyncio.Semaphore,
) -> list[PolymarketEvent]:
    """Fetch specific Polymarket events by slug from gamma-api.

    Each slug fetched in parallel under `sem`; failures degrade per-slug —
    bogus slugs log a warning and drop out.

    No horizon filter is applied here, unlike `fetch_gamma_slate`. Slugs
    reach this function only via explicit `--slug` CLI args, so the user
    has already opted in to that specific event — second-guessing with a
    horizon check produces surprising drops when gamma's `endDate` is a
    settlement window (e.g. some ATP markets) rather than a tipoff.
    `horizon_hours` is kept in the signature for symmetry with the
    default-browse path.
    """
    if not slugs:
        return []

    async def _one(slug: str) -> PolymarketEvent | None:
        async with sem:
            payload = await fetch_gamma_event(http, slug)
            if payload is None:
                return None
            event = PolymarketEvent.from_gamma(payload)
            if event is None:
                log.warning(
                    "gamma slug=%s: no tradable moneyline markets after filter",
                    slug,
                )
            return event

    raw_events = await asyncio.gather(*(_one(s) for s in slugs))
    kept = [ev for ev in raw_events if ev is not None]

    log.info("fetched %d/%d events from gamma-api by slug", len(kept), len(slugs))
    return kept


async def fetch_slate(
    opts: SlateOptions,
    *,
    http: httpx.AsyncClient,
    gamma_sem: asyncio.Semaphore,
) -> list[PolymarketEvent]:
    """Build the unified Polymarket slate from gamma. Single source of truth
    used by both `run_pipeline` and the `skims fetch` CLI path so they can
    never drift.

    Caller owns the lifetime of `http`. `run_pipeline` passes `uw.http`
    (sharing the UW client's httpx for connection reuse); the `skims fetch`
    path passes a standalone `httpx.AsyncClient` since it has no UW
    context to piggyback on.

    Composition rules — chosen so each flag matches its instinctive read:
    - bare (no flags): default browse, all sports within horizon.
    - `--league` only: default browse filtered by those league prefixes.
    - `--sport` only: gamma listing scoped to those sport tags
      (server-side `tag_slug=<sport>`).
    - `--slug` only: those events specifically. The default browse is
      SKIPPED — `skims fetch --slug X` means "show me X", not "show me X
      plus today's whole slate".
    - `--league` and/or `--sport` + `--slug`: union (filtered default
      browse plus the explicit slugs added on top, deduped by event id).

    `--slug` always bypasses the horizon filter so the user can pull a
    specific event regardless of when it starts. CLOB book + price-history
    enrichment runs in the caller after `fetch_slate` returns, so the
    heavy HTTP fan-out happens once on the deduped union rather than
    per-fetcher.
    """
    # Skip the default browse when the user gave only `--slug` and no
    # `--league` / `--sport` — they're asking for those events
    # specifically, not "those events plus today's full slate". When any
    # filter flag is present alongside `--slug`, the default browse
    # runs (scoped by those filters) and slugs add on top.
    if opts.slugs and not opts.leagues and not opts.sports:
        events: list[PolymarketEvent] = []
    else:
        events = await fetch_gamma_slate(
            http, opts.leagues, opts.horizon_hours, sports=opts.sports
        )

    if opts.slugs:
        extra = await fetch_gamma_events(http, opts.slugs, opts.horizon_hours, gamma_sem)
        seen: set[str] = {ev.id for ev in events}
        for ev in extra:
            if ev.id in seen:
                continue
            seen.add(ev.id)
            events.append(ev)
    return events


async def resolve_unusual_whales(
    uw: UnusualWhalesClient,
    events: list[PolymarketEvent],
    sem: asyncio.Semaphore,
    *,
    resolver: GammaTokenResolver,
) -> None:
    """Fetch Unusual Whales flow context per event and attach to `event.uw_context`.

    For each event: resolve `event.slug` → YES-side ERC-1155 `asset_id` via
    Polymarket's gamma-api, then GET UW's `/predictions/market/{asset_id}` for
    the compact flow snapshot. Failures at any step leave `uw_context=None`
    and don't abort the run — this is an enrichment stage, not a dependency.

    Events with no valid slug or no gamma-api match are silently skipped.
    """
    if not uw.enabled:
        return

    slugged = [(ev.slug, ev) for ev in events if ev.slug]
    if not slugged:
        return

    # Dedupe by slug — for our data model event.slug is unique per event, but
    # guard against future multi-event-per-slug shapes regardless.
    by_slug: dict[str, list[PolymarketEvent]] = {}
    for slug, ev in slugged:
        by_slug.setdefault(slug, []).append(ev)

    # `resolver` is passed in (was instantiated at run_pipeline scope) so the
    # CLOB price-history stage can share its cache. Both stages key off the
    # same gamma `/markets?slug=` response.

    async def _one(s: str, evs: list[PolymarketEvent]) -> None:
        async with sem:
            # One gamma fetch covers two needs: the token IDs UW is keyed
            # by, and the supplementary fields (1d move, competitive,
            # spread, real CLOB liquidity) we piggyback onto each market.
            # Both come off the same `/markets?slug=` response so this is
            # a single HTTP call regardless of how much we extract.
            snap = await resolver.resolve_snapshot(s)
            if snap is None:
                return
            if snap.clob_token_ids is not None:
                yes_asset_id, _no_asset_id = snap.clob_token_ids
                ctx = await uw.get_market_detail(yes_asset_id)
                # Skip contexts that carry no flow signal — UW's index
                # extends past its smart-money / tag pipelines, so e.g.
                # offshore tennis markets resolve to a 200 with empty
                # trades, null tags, volume=0, no MCI. Attaching that
                # gives the director a UW block that's all `?`s and adds
                # no information past the main bid/ask. See
                # `UnusualWhalesContext.has_actionable_signal`.
                if ctx is not None and ctx.has_actionable_signal():
                    for e in evs:
                        e.uw_context = ctx
            # Merge gamma fields onto every market on every event sharing
            # this slug. NO-side clones get the same values — `spread`,
            # `1d`, `competitive`, `liquidityClob` are market-level, not
            # side-directional, so no inversion is needed.
            for e in evs:
                for i, m in enumerate(e.markets):
                    e.markets[i] = m.model_copy(
                        update={
                            "gamma_spread": snap.spread,
                            "gamma_one_day_price_change": snap.one_day_price_change,
                            "gamma_one_month_price_change": snap.one_month_price_change,
                            "gamma_competitive": snap.competitive,
                            "gamma_liquidity_dollars": snap.liquidity_clob,
                            "gamma_volume_dollars": snap.volume_clob,
                            "gamma_accepting_orders": snap.accepting_orders,
                        }
                    )

    await asyncio.gather(*(_one(s, evs) for s, evs in by_slug.items()))
    attached = sum(1 for ev in events if ev.uw_context is not None)
    log.info("attached unusual-whales context to %d/%d events", attached, len(events))


async def enrich_clob_book(
    events: list[PolymarketEvent],
    resolver: GammaTokenResolver,
    http: httpx.AsyncClient,
    sem: asyncio.Semaphore,
) -> None:
    """Attach CLOB order-book size + depth + book-$ fields to each market.

    Same iteration shape as `enrich_price_history` (per unique market slug,
    NO clones swap sides without flipping values), and same posture: an
    enrichment stage that degrades silently per slug. Replaces the depth
    fields that the legacy `markets.book(...)` SDK call used to populate.

    The CLOB `/book` endpoint returns the **full global book** (bids +
    asks, multi-level), so the depth/size/book-$ numbers we land here are
    significantly larger than the US SDK's slice — that's expected and
    intentional. The unauthenticated endpoint is the same provider as
    `/prices-history`; concurrency is shared via `CLOB_FETCH_SEM`.

    NO-side clones receive the same shape with bid/ask swapped (the YES
    bid book IS the implied NO ask book and vice versa) — no value flip,
    just side label swap. Mirrors the in-place inversion at
    `PolymarketMarket.inverted_no_side` lines 502–507.
    """
    by_market_slug: dict[str, list[tuple[PolymarketEvent, int]]] = {}
    for ev in events:
        for i, m in enumerate(ev.markets):
            if m.slug:
                by_market_slug.setdefault(m.slug, []).append((ev, i))
    if not by_market_slug:
        return

    async def _one(market_slug: str, refs: list[tuple[PolymarketEvent, int]]) -> None:
        async with sem:
            snap = await resolver.resolve_snapshot(market_slug)
            if snap is None or snap.clob_token_ids is None:
                return
            yes_token_id, _no_token_id = snap.clob_token_ids
            book = await fetch_book(http, yes_token_id)
            summary = summarize_book(book)
            if summary is None:
                return
            for evt, idx in refs:
                mkt = evt.markets[idx]
                if mkt.is_no_side:
                    # NO clone: bid/ask sides swap (YES bid book = implied
                    # NO ask book) but values themselves don't flip.
                    evt.markets[idx] = mkt.model_copy(
                        update={
                            "yes_bid_size_top": summary.ask_top_size,
                            "yes_ask_size_top": summary.bid_top_size,
                            "yes_bid_book_dollars": summary.ask_book_dollars,
                            "yes_ask_book_dollars": summary.bid_book_dollars,
                            "yes_bid_depth": summary.ask_depth,
                            "yes_ask_depth": summary.bid_depth,
                        }
                    )
                else:
                    evt.markets[idx] = mkt.model_copy(
                        update={
                            "yes_bid_size_top": summary.bid_top_size,
                            "yes_ask_size_top": summary.ask_top_size,
                            "yes_bid_book_dollars": summary.bid_book_dollars,
                            "yes_ask_book_dollars": summary.ask_book_dollars,
                            "yes_bid_depth": summary.bid_depth,
                            "yes_ask_depth": summary.ask_depth,
                        }
                    )

    await asyncio.gather(*(_one(s, refs) for s, refs in by_market_slug.items()))
    enriched = sum(
        1
        for ev in events
        for m in ev.markets
        if m.yes_bid_book_dollars is not None or m.yes_ask_book_dollars is not None
    )
    total = sum(len(ev.markets) for ev in events)
    log.info("attached clob book to %d/%d markets", enriched, total)


async def enrich_price_history(
    events: list[PolymarketEvent],
    resolver: GammaTokenResolver,
    http: httpx.AsyncClient,
    sem: asyncio.Semaphore,
) -> None:
    """Attach a CLOB price-history sparkline + recency scalars to each market.

    Iteration is per **market slug** (deduplicated across events), not per
    event slug. Soccer-style 3-way events split each outcome into its own
    market slug (e.g. `epl-lee-bur-2026-05-01-{lee,bur,draw}`), and gamma's
    `/markets?slug=` only resolves on the per-outcome slug. Tennis/UFC
    binary head-to-heads share one slug across YES + inverted-NO clones;
    per-market iteration picks up both clones, and the NO record carries
    the inverted summary.

    Per unique market slug: get the YES `clobTokenId` from the gamma
    resolver (shared with the UW + book stages), fetch ~24h of mid prices
    from `clob.polymarket.com/prices-history`, summarize, and attach.

    Failures are silent per-slug (logged at WARNING by the core fetcher);
    affected markets keep `clob_*` fields as None and renderers skip them.
    Same posture as `resolve_unusual_whales` — this is an enrichment stage,
    not a dependency.

    Concurrency is capped by the caller-supplied `sem`. The token IDs
    already cached by the UW + book stages are free re-hits; cold slugs
    fire one gamma `/markets?slug=` call followed by one CLOB
    `/prices-history` call.
    """
    by_market_slug: dict[str, list[tuple[PolymarketEvent, int]]] = {}
    for ev in events:
        for i, m in enumerate(ev.markets):
            if m.slug:
                by_market_slug.setdefault(m.slug, []).append((ev, i))
    if not by_market_slug:
        return

    async def _one(market_slug: str, refs: list[tuple[PolymarketEvent, int]]) -> None:
        async with sem:
            snap = await resolver.resolve_snapshot(market_slug)
            if snap is None or snap.clob_token_ids is None:
                return
            yes_token_id, _no_token_id = snap.clob_token_ids
            history = await fetch_price_history(http, yes_token_id)
            if not history:
                return
            summary = summarize_history(history)
            if summary is None:
                return
            for evt, idx in refs:
                mkt = evt.markets[idx]
                if mkt.is_no_side:
                    evt.markets[idx] = mkt.model_copy(
                        update={
                            "clob_price_change_30m": (
                                -summary.change_30m
                                if summary.change_30m is not None
                                else None
                            ),
                            "clob_price_change_1h": (
                                -summary.change_1h
                                if summary.change_1h is not None
                                else None
                            ),
                            "clob_price_change_4h": (
                                -summary.change_4h
                                if summary.change_4h is not None
                                else None
                            ),
                            "clob_price_change_24h": (
                                -summary.change_24h
                                if summary.change_24h is not None
                                else None
                            ),
                            "clob_price_path_sparkline": invert_sparkline(
                                summary.sparkline
                            ),
                            "clob_price_history": [
                                (t, 1.0 - p) for t, p in summary.raw_points
                            ],
                        }
                    )
                else:
                    evt.markets[idx] = mkt.model_copy(
                        update={
                            "clob_price_change_30m": summary.change_30m,
                            "clob_price_change_1h": summary.change_1h,
                            "clob_price_change_4h": summary.change_4h,
                            "clob_price_change_24h": summary.change_24h,
                            "clob_price_path_sparkline": summary.sparkline,
                            "clob_price_history": summary.raw_points,
                        }
                    )

    await asyncio.gather(
        *(_one(s, refs) for s, refs in by_market_slug.items())
    )
    enriched = sum(
        1
        for ev in events
        for m in ev.markets
        if m.clob_price_path_sparkline is not None
    )
    total = sum(len(ev.markets) for ev in events)
    log.info("attached clob price history to %d/%d markets", enriched, total)


async def enrich_tennis_stats(
    provider: TennisStatsProvider,
    events: list[PolymarketEvent],
    sem: asyncio.Semaphore,
    errors: list[ErrorRecord],
) -> None:
    """Attach `event.tennis_stats` for ATP/WTA singles head-to-heads.

    Iterates per event (not per market slug) — tennis stats are
    match-level data, not side-directional, so a NO clone shares the
    parent event's context naturally. Same fail-silent posture as the
    other enrichment stages: vendor errors record one ErrorRecord with
    `stage="tennis_stats"`, leave `tennis_stats=None`, and let the rest
    of the pipeline continue.

    Runs LAST among enrichers because (a) the sport gate consumes
    `event.sport_type` populated upstream by `from_gamma`, and (b) it's
    the only enricher that can be skipped per-event by sport — every
    non-tennis event short-circuits at `tennis_match_identity` without
    touching the vendor.

    The `provider` is always non-None — the factory returns the stub
    when no key is configured rather than `None`, so this stage doesn't
    need an `if enabled` branch. The stub returns `None` for every
    event, which leaves the pipeline behaving identically to a run
    where this enrichment didn't exist.
    """
    if not events:
        return

    async def _one(event: PolymarketEvent) -> None:
        identity = tennis_match_identity(event)
        if identity is None:
            return
        async with sem:
            try:
                ctx = await provider.fetch(identity)
            except Exception as e:  # noqa: BLE001
                errors.append(
                    ErrorRecord(
                        event_id=event.id,
                        stage="tennis_stats",
                        error=f"{type(e).__name__}: {e}",
                    )
                )
                log.warning(
                    "tennis_stats fetch failed for %s (%s vs %s): %s",
                    event.id,
                    identity.player_a,
                    identity.player_b,
                    type(e).__name__,
                )
                return
        # `has_actionable_signal` matches UW's posture — drop empty
        # contexts (every numeric field None, no H2H) so the renderer
        # doesn't waste prompt tokens on a header with no body.
        if ctx is not None and ctx.has_actionable_signal():
            event.tennis_stats = ctx

    await asyncio.gather(*(_one(ev) for ev in events))
    attached = sum(1 for ev in events if ev.tennis_stats is not None)
    log.info(
        "attached tennis stats to %d/%d events (provider=%s)",
        attached,
        len(events),
        provider.name,
    )


async def _run_lenses(
    provider: FetcherProvider,
    anthropic: AsyncAnthropic,
    event: PolymarketEvent,
    fetcher_sem: asyncio.Semaphore,
    reasoner_sem: asyncio.Semaphore,
    errors: list[ErrorRecord],
) -> tuple[dict[str, LensNotebook], dict[str, SpecialistReport]] | None:
    """Run all four lenses for one event. Each lens is a provider fetcher
    (Stage A) → Claude reasoner (Stage B) chain; lenses run in parallel,
    fetcher→reasoner is sequential within a lens.

    `fetcher_sem` is released between stages so a slow fetcher search loop
    doesn't tie up a fetcher slot through the (typically faster) Claude
    reasoner call. Per-event failure posture matches the legacy specialists
    pipeline: any failure at either stage of any lens drops the event so
    the director never receives a partial set of reports.
    """

    async def _one(lens: str) -> _LensOutcome:
        try:
            async with fetcher_sem:
                notebook = await provider.fetch(event, lens)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001
            return _LensOutcome(lens=lens, error_stage="fetcher", error=e)
        try:
            async with reasoner_sem:
                report = await REASONERS[lens](anthropic, event, notebook)
        except Exception as e:  # noqa: BLE001
            return _LensOutcome(
                lens=lens, notebook=notebook, error_stage="reasoner", error=e
            )
        return _LensOutcome(lens=lens, notebook=notebook, report=report)

    outcomes = await asyncio.gather(*(_one(n) for n in REASONERS))
    notebooks: dict[str, LensNotebook] = {}
    reports: dict[str, SpecialistReport] = {}
    failed = False
    for o in outcomes:
        if o.error is not None:
            errors.append(
                ErrorRecord(
                    event_id=event.id,
                    stage=f"{o.error_stage}:{o.lens}",
                    error=f"{type(o.error).__name__}: {o.error}",
                )
            )
            failed = True
        else:
            assert o.notebook is not None and o.report is not None
            notebooks[o.lens] = o.notebook
            reports[o.lens] = o.report
    if failed:
        return None
    return notebooks, reports


# Logs live at <repo-root>/logs/runs/<run_id>.jsonl. Resolved at module-load
# so behaviour doesn't drift with cwd. `parents[2]` walks
# src/skimsmarkets/pipeline.py → src/skimsmarkets → src → repo-root.
_LOG_ROOT = Path(__file__).resolve().parents[2] / "logs" / "runs"


def _persist_run(result: RunResult) -> None:
    """Write each prediction (with its entry decision) to a per-run JSONL.

    One file per run named `<run_id>.jsonl`. Best-effort: any I/O failure is
    logged and swallowed — persistence must never abort the run, matching the
    enrichment-stage posture.

    Why JSONL not parquet: lines are easy to tail, easy to grep, easy to feed
    into a future grading script that joins against gamma settlement after
    kickoff. Volume is tiny (one line per ranked event, ~30/day).
    """
    try:
        _LOG_ROOT.mkdir(parents=True, exist_ok=True)
        path = _LOG_ROOT / f"{result.run_id}.jsonl"
        logged_at = datetime.now(UTC).isoformat()
        with path.open("w") as f:
            for p in result.predictions:
                # Notebooks + reports come from the two-stage agent chain
                # (Grok fetcher → Claude reasoner). Persisting both lets a
                # grading script ask "was the evidence right?" and "was the
                # reasoning right?" as separate questions. mode="json" so
                # any datetime/Decimal fields serialize cleanly.
                notebooks_for_event = {
                    lens: nb.model_dump(mode="json")
                    for lens, nb in result.notebooks.get(p.event_id, {}).items()
                }
                reports_for_event = {
                    lens: r.model_dump(mode="json")
                    for lens, r in result.reports.get(p.event_id, {}).items()
                }
                # Judge output: persisted alongside the prediction so
                # retrospective grading can correlate the judge's score
                # against actual hit-rate as a separate question from the
                # director's predicted probability. Null/empty when the
                # judge call failed or didn't cover this event.
                da = result.defensibility_assessments.get(p.event_id)
                # Tennis stats vendor payload — top-level (not nested in
                # notebooks) so retrospective grading can ask "did the
                # API have the right facts?" separately from "did the
                # fetcher use them?". Null on non-tennis events and on
                # tennis events the stub / vendor failed to populate.
                ts = result.tennis_stats.get(p.event_id)
                payload = {
                    "run_id": result.run_id,
                    "logged_at_utc": logged_at,
                    # Run-level fetcher metadata — top-level (not nested in
                    # `notebooks`) so retrospective A/B grading can group
                    # rows by provider via `jq '.fetcher_provider'`.
                    "fetcher_provider": result.fetcher_provider,
                    "fetcher_model": result.fetcher_model,
                    "event_id": p.event_id,
                    "event_title": p.event_title,
                    "market_slug": p.market_slug,
                    "predicted_winner": p.predicted_winner,
                    "predicted_yes_probability": p.predicted_yes_probability,
                    "polymarket_implied_probability": p.polymarket_implied_probability,
                    "confidence": p.confidence,
                    "headline": p.headline,
                    # Director synthesis fields — `reasoning` is the 3-6
                    # sentence rationale, `specialist_weights` shows how
                    # the four lenses were weighted, `disagreements_flagged`
                    # surfaces material directional disagreements between
                    # specialists, and `uw_flow_note` captures the director's
                    # read on Unusual Whales flow when present (null
                    # otherwise). All four feed retrospective grading of
                    # synthesis quality and UW alignment over time.
                    "reasoning": p.reasoning,
                    "specialist_weights": p.specialist_weights,
                    "disagreements_flagged": p.disagreements_flagged,
                    "uw_flow_note": p.uw_flow_note,
                    "defensibility_score": (
                        da.defensibility_score if da is not None else None
                    ),
                    "defensibility_rationale": (
                        da.defensibility_rationale if da is not None else None
                    ),
                    "defensibility_flags": (
                        da.defensibility_flags if da is not None else []
                    ),
                    "tennis_stats": (
                        ts.model_dump(mode="json") if ts is not None else None
                    ),
                    "notebooks": notebooks_for_event,
                    "specialist_reports": reports_for_event,
                }
                f.write(json.dumps(payload, separators=(",", ":")) + "\n")
        log.info("persisted %d predictions to %s", len(result.predictions), path)
    except Exception as e:  # noqa: BLE001
        log.warning("run-log persistence failed: %s", e)


async def process_event(
    provider: FetcherProvider,
    anthropic: AsyncAnthropic,
    event: PolymarketEvent,
    fetcher_sem: asyncio.Semaphore,
    reasoner_sem: asyncio.Semaphore,
    director_sem: asyncio.Semaphore,
    errors: list[ErrorRecord],
) -> tuple[
    MarketPrediction, dict[str, LensNotebook], dict[str, SpecialistReport]
] | None:
    """Run the full agent chain for one event.

    Returns the prediction alongside the per-lens notebooks and reasoner
    reports so the caller can persist them to the run JSONL. Returns None
    when the event was dropped (any lens stage failure or director failure).
    """
    log.info("processing event %s (%s)", event.id, event.title)
    pairs = await _run_lenses(
        provider, anthropic, event, fetcher_sem, reasoner_sem, errors
    )
    if pairs is None:
        return None
    notebooks, reports = pairs

    async with director_sem:
        try:
            prediction = await synthesize_prediction(anthropic, event, reports)
        except Exception as e:  # noqa: BLE001
            errors.append(
                ErrorRecord(
                    event_id=event.id,
                    stage="director",
                    error=f"{type(e).__name__}: {e}",
                )
            )
            return None
    return prediction, notebooks, reports


async def run_pipeline(
    *,
    leagues: list[str] | None = None,
    horizon_hours: int = cfg.DEFAULT_HORIZON_HOURS,
    slugs: list[str] | None = None,
    sports: list[str] | None = None,
    fetcher_provider: str | None = None,
    tennis_stats_disabled: bool = False,
) -> RunResult:
    """End-to-end: fetch the Polymarket sports slate inside the horizon,
    enrich with CLOB book + price history, then run 4 specialists + director
    per event. Returns a leaderboard-ready `RunResult` sorted downstream by
    predicted probability.

    Slate composition:
    - default browse: tag-listed sports events filtered by `leagues` prefix(es)
      and the horizon time window. Empty `leagues` = no client-side league
      filter.
    - `sports`: gamma `tag_slug=<sport>` server-side filter (e.g. `tennis`,
      `nba`). Repeatable: each tag is queried separately and unioned.
      Empty = umbrella `tag_slug=sports` (current default).
    - `slugs`: explicit list of event slugs to include (bypasses the horizon
      filter so a specific event always lands).
    - `fetcher_provider`: 'grok' or 'gemini'. None = use FETCHER_PROVIDER env
      var, defaulting to 'grok'. The provider runs the per-lens Stage A
      (fetcher) for every event in the slate; reasoner / director / judge
      are always Claude regardless.
    """
    config = cfg.Config.from_env(
        fetcher_provider=fetcher_provider,
        tennis_stats_disabled=tennis_stats_disabled,
    )
    run_id = uuid.uuid4().hex[:8]
    result = RunResult(run_id=run_id)

    fetcher_sem = asyncio.Semaphore(cfg.FETCHER_SEM)
    reasoner_sem = asyncio.Semaphore(cfg.REASONER_SEM)
    director_sem = asyncio.Semaphore(cfg.DIRECTOR_SEM)
    uw_sem = asyncio.Semaphore(cfg.UW_FETCH_SEM)
    gamma_sem = asyncio.Semaphore(cfg.GAMMA_FETCH_SEM)
    clob_sem = asyncio.Semaphore(cfg.CLOB_FETCH_SEM)
    tennis_sem = asyncio.Semaphore(cfg.TENNIS_STATS_FETCH_SEM)

    # `public_http` is a shared httpx client for any public, unauthed
    # Polymarket-host call: gamma `/events` (slate + token IDs +
    # supplementary fields) and CLOB `/book` + `/prices-history`. Shared
    # so the connection pool is reused across enrichment stages.
    async with (
        UnusualWhalesClient(config.unusual_whales_api_key) as uw,
        httpx.AsyncClient(timeout=20.0) as public_http,
        build_tennis_provider(config) as tennis_provider,
    ):
        gamma_resolver = GammaTokenResolver(public_http)
        # `fetch_slate` is the single source of truth (default browse +
        # explicit slugs, deduped) — same call shape as the `skims fetch`
        # CLI path so the two can never drift. UW's http is reused for
        # connection pooling on the gamma calls.
        slate_opts = SlateOptions(
            leagues=leagues or [],
            slugs=slugs or [],
            sports=sports or [],
            horizon_hours=horizon_hours,
        )
        events = await fetch_slate(slate_opts, http=uw.http, gamma_sem=gamma_sem)
        result.fetched_events = len(events)

        result.considered_events = len(events)
        log.info("considering %d events", len(events))

        if not events:
            return result

        # Enrichment stages all share the same `gamma_resolver` so token-ID
        # lookups are paid for at most once per slug across UW + CLOB book +
        # CLOB price-history. UW runs first because it can drop events with
        # no actionable signal; book/history then enrich whatever remains.
        await resolve_unusual_whales(uw, events, uw_sem, resolver=gamma_resolver)
        # CLOB book — top-of-book size, depth, full-book $ totals. Adds
        # one HTTP per unique market slug (deduplicated). Replaces the
        # legacy US `markets.book(...)` BBO refresh; the CLOB endpoint
        # exposes the full global book so book-$ totals here are 5–20×
        # what the legacy US slice used to show.
        await enrich_clob_book(events, gamma_resolver, public_http, clob_sem)
        # Optional CLOB price-history enrichment — opt-in via the
        # `CLOB_HISTORY_ENABLED` constant in `config.py`. Adds 1 HTTP per
        # unique slug. When off, this is a no-op (zero CLOB calls).
        if cfg.CLOB_HISTORY_ENABLED:
            await enrich_price_history(events, gamma_resolver, public_http, clob_sem)
        else:
            log.info("clob price history disabled (cfg.CLOB_HISTORY_ENABLED=False)")
        # Tennis stats run LAST among enrichers — gated per-event by sport,
        # so deferring its work means non-tennis events have already paid
        # all the upstream gamma/CLOB enrichment costs we want anyway.
        # The provider is always non-None (factory returns the stub when no
        # key is configured), so this is safe without an `if enabled` branch.
        await enrich_tennis_stats(tennis_provider, events, tennis_sem, result.errors)
        # Snapshot the per-event tennis context onto the RunResult so
        # `_persist_run` can write it to the JSONL row even though the
        # event itself isn't carried into persistence (only the resulting
        # `MarketPrediction` is). Keyed by event id to match how
        # `notebooks` / `reports` line up against `predictions`.
        for ev in events:
            if ev.tennis_stats is not None:
                result.tennis_stats[ev.id] = ev.tennis_stats

        provider = build_provider(config.fetcher_provider, config)
        result.fetcher_provider = provider.name
        result.fetcher_model = provider.model
        log.info(
            "fetcher provider=%s model=%s", provider.name, provider.model
        )
        anthropic = AsyncAnthropic(api_key=config.anthropic_api_key)
        try:
            outcomes = await asyncio.gather(
                *(
                    process_event(
                        provider,
                        anthropic,
                        e,
                        fetcher_sem,
                        reasoner_sem,
                        director_sem,
                        result.errors,
                    )
                    for e in events
                )
            )
        finally:
            await provider.aclose()

        for outcome in outcomes:
            if outcome is None:
                continue
            prediction, notebooks, reports = outcome
            result.predictions.append(prediction)
            result.notebooks[prediction.event_id] = notebooks
            result.reports[prediction.event_id] = reports

        # Slate-level judge — one Anthropic call after all per-event
        # directors finish. Reads each MarketPrediction's reasoning + flags
        # + UW note and emits a DefensibilityAssessment per event; the
        # leaderboard then sorts by `defensibility_score` desc with
        # `predicted_yes_probability` as a tiebreak. Failure here is
        # silent-degrade: log a warning, record one slate-level
        # ErrorRecord, and let the leaderboard fall back to
        # predicted-probability sort. Skipped when the slate is empty (the
        # existing `if result.predictions:` guard below would catch that,
        # but skipping the call avoids spending a token on a known no-op).
        if result.predictions:
            try:
                judgment = await judge_slate(anthropic, result.predictions)
                for a in judgment.assessments:
                    result.defensibility_assessments[a.event_id] = a
            except Exception as e:  # noqa: BLE001
                result.errors.append(
                    ErrorRecord(
                        event_id="*",
                        stage="judge",
                        error=f"{type(e).__name__}: {e}",
                    )
                )
                log.warning(
                    "judge failed; falling back to "
                    "predicted_probability sort: %s",
                    e,
                )

    if result.predictions:
        _persist_run(result)
    return result
