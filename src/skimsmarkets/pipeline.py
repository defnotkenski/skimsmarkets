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
from xai_sdk import AsyncClient as XAIAsyncClient

from skimsmarkets import config as cfg
from skimsmarkets.agents.director import synthesize_prediction
from skimsmarkets.agents.schemas import MarketPrediction, SpecialistReport
from skimsmarkets.agents.specialists import SPECIALISTS
from skimsmarkets.clob import (
    fetch_book,
    fetch_price_history,
    invert_sparkline,
    summarize_book,
    summarize_history,
)
from skimsmarkets.polymarket.models import PolymarketEvent
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
    stage: str  # "specialist:<name>" or "director"
    error: str


@dataclass
class RunResult:
    run_id: str
    predictions: list[MarketPrediction] = field(default_factory=list)
    errors: list[ErrorRecord] = field(default_factory=list)
    fetched_events: int = 0
    considered_events: int = 0


@dataclass(frozen=True)
class SlateOptions:
    """Inputs shared by every slate-building entry point — both the
    `skims fetch` CLI path and `run_pipeline`'s own slate stage. Frozen so
    callers can pass the same instance through multiple stages without
    worrying about mutation.

    `leagues` and `slugs` default to empty lists rather than None because
    every callsite already normalizes the "no filter" case to an empty
    iterable; one less branch downstream. Empty `leagues` = no league
    filter (browse all sports).
    """

    leagues: list[str] = field(default_factory=list)
    slugs: list[str] = field(default_factory=list)
    horizon_hours: int = cfg.DEFAULT_HORIZON_HOURS


async def fetch_gamma_slate(
    http: httpx.AsyncClient,
    leagues: list[str],
    horizon_hours: int,
) -> list[PolymarketEvent]:
    """Fetch the Polymarket sports slate from gamma-api.

    Single source of truth for "what's in today's slate" — both the `skims
    fetch` display path and the full pipeline iterate exactly the events
    returned here. Filters layered top to bottom:

    1. **Bulk listing.** `list_gamma_events` pages through gamma's
       `/events?tag_slug=sports&order=endDate&ascending=true` for upcoming
       events soonest-first. Pagination is necessary because esports
       (cs2, lol, dota2) and high-volume markets crowd out actual sports
       leagues in page 1.
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
        if len(live_markets) != len(ev.markets):
            ev = ev.model_copy(update={"markets": live_markets})
        kept.append(ev)

    log.info(
        "kept %d gamma events after league + horizon + tradability filters",
        len(kept),
    )
    return kept


async def fetch_gamma_events(
    http: httpx.AsyncClient,
    slugs: list[str],
    horizon_hours: int,
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
    - `--slug` only: those events specifically. The default browse is
      SKIPPED — `skims fetch --slug X` means "show me X", not "show me X
      plus today's whole slate".
    - `--league` + `--slug`: union (default browse filtered by leagues,
      plus the explicit slugs added on top, deduped by event id).

    `--slug` always bypasses the horizon filter so the user can pull a
    specific event regardless of when it starts. CLOB book + price-history
    enrichment runs in the caller after `fetch_slate` returns, so the
    heavy HTTP fan-out happens once on the deduped union rather than
    per-fetcher.
    """
    # Skip the default browse when the user gave only `--slug` and no
    # `--league` — they're asking for those events specifically, not "those
    # events plus today's full slate". When both flags are present, the
    # default browse runs (filtered by leagues) and slugs add on top.
    if opts.slugs and not opts.leagues:
        events: list[PolymarketEvent] = []
    else:
        events = await fetch_gamma_slate(http, opts.leagues, opts.horizon_hours)

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
            for ev, i in refs:
                m = ev.markets[i]
                if m.is_no_side:
                    # NO clone: bid/ask sides swap (YES bid book = implied
                    # NO ask book) but values themselves don't flip.
                    ev.markets[i] = m.model_copy(
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
                    ev.markets[i] = m.model_copy(
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
            for ev, i in refs:
                m = ev.markets[i]
                if m.is_no_side:
                    ev.markets[i] = m.model_copy(
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
                    ev.markets[i] = m.model_copy(
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


async def _run_specialists(
    xai: XAIAsyncClient,
    event: PolymarketEvent,
    sem: asyncio.Semaphore,
    errors: list[ErrorRecord],
) -> dict[str, SpecialistReport] | None:
    """Run all specialists for one event. Return None if any specialist failed
    (the event then skips director)."""

    async def _one(specialist: str) -> tuple[str, "SpecialistReport | Exception"]:
        async with sem:
            try:
                return specialist, await SPECIALISTS[specialist](xai, event)
            except Exception as e:  # noqa: BLE001
                return specialist, e

    results = await asyncio.gather(*(_one(n) for n in SPECIALISTS))
    reports: dict[str, SpecialistReport] = {}
    failed = False
    for name, result in results:
        if isinstance(result, Exception):
            errors.append(
                ErrorRecord(
                    event_id=event.id,
                    stage=f"specialist:{name}",
                    error=f"{type(result).__name__}: {result}",
                )
            )
            failed = True
        else:
            reports[name] = result
    if failed:
        return None
    return reports


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
                payload = {
                    "run_id": result.run_id,
                    "logged_at_utc": logged_at,
                    "event_id": p.event_id,
                    "event_title": p.event_title,
                    "market_slug": p.market_slug,
                    "predicted_winner": p.predicted_winner,
                    "predicted_yes_probability": p.predicted_yes_probability,
                    "polymarket_implied_probability": p.polymarket_implied_probability,
                    "confidence": p.confidence,
                    "headline": p.headline,
                }
                f.write(json.dumps(payload, separators=(",", ":")) + "\n")
        log.info("persisted %d predictions to %s", len(result.predictions), path)
    except Exception as e:  # noqa: BLE001
        log.warning("run-log persistence failed: %s", e)


async def process_event(
    xai: XAIAsyncClient,
    anthropic: AsyncAnthropic,
    event: PolymarketEvent,
    specialist_sem: asyncio.Semaphore,
    director_sem: asyncio.Semaphore,
    errors: list[ErrorRecord],
) -> MarketPrediction | None:
    log.info("processing event %s (%s)", event.id, event.title)
    reports = await _run_specialists(xai, event, specialist_sem, errors)
    if reports is None:
        return None

    async with director_sem:
        try:
            return await synthesize_prediction(anthropic, event, reports)
        except Exception as e:  # noqa: BLE001
            errors.append(
                ErrorRecord(
                    event_id=event.id,
                    stage="director",
                    error=f"{type(e).__name__}: {e}",
                )
            )
            return None


async def run_pipeline(
    *,
    leagues: list[str] | None = None,
    dry_run: bool = False,
    horizon_hours: int = cfg.DEFAULT_HORIZON_HOURS,
    slugs: list[str] | None = None,
) -> RunResult:
    """End-to-end: fetch the Polymarket sports slate inside the horizon,
    enrich with CLOB book + price history, then run 4 specialists + director
    per event. Returns a leaderboard-ready `RunResult` sorted downstream by
    predicted probability.

    Slate composition:
    - default browse: tag-listed sports events filtered by `leagues` prefix(es)
      and the horizon time window. Empty `leagues` = browse all sports.
    - `slugs`: explicit list of event slugs to include (bypasses the horizon
      filter so a specific event always lands).
    """
    config = cfg.Config.from_env()
    run_id = uuid.uuid4().hex[:8]
    result = RunResult(run_id=run_id)

    specialist_sem = asyncio.Semaphore(cfg.SPECIALIST_SEM)
    director_sem = asyncio.Semaphore(cfg.DIRECTOR_SEM)
    uw_sem = asyncio.Semaphore(cfg.UW_FETCH_SEM)
    gamma_sem = asyncio.Semaphore(cfg.GAMMA_FETCH_SEM)
    clob_sem = asyncio.Semaphore(cfg.CLOB_FETCH_SEM)

    # `public_http` is a shared httpx client for any public, unauthed
    # Polymarket-host call: gamma `/events` (slate + token IDs +
    # supplementary fields) and CLOB `/book` + `/prices-history`. Shared
    # so the connection pool is reused across enrichment stages.
    async with (
        UnusualWhalesClient(config.unusual_whales_api_key) as uw,
        httpx.AsyncClient(timeout=20.0) as public_http,
    ):
        gamma_resolver = GammaTokenResolver(public_http)
        # `fetch_slate` is the single source of truth (default browse +
        # explicit slugs, deduped) — same call shape as the `skims fetch`
        # CLI path so the two can never drift. UW's http is reused for
        # connection pooling on the gamma calls.
        slate_opts = SlateOptions(
            leagues=leagues or [],
            slugs=slugs or [],
            horizon_hours=horizon_hours,
        )
        events = await fetch_slate(slate_opts, http=uw.http, gamma_sem=gamma_sem)
        result.fetched_events = len(events)

        if dry_run:
            events = events[:1]
        result.considered_events = len(events)
        log.info("considering %d events (dry_run=%s)", len(events), dry_run)

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

        xai = XAIAsyncClient(api_key=config.xai_api_key)
        anthropic = AsyncAnthropic(api_key=config.anthropic_api_key)
        try:
            predictions = await asyncio.gather(
                *(
                    process_event(
                        xai,
                        anthropic,
                        e,
                        specialist_sem,
                        director_sem,
                        result.errors,
                    )
                    for e in events
                )
            )
        finally:
            await xai.close()

        for p in predictions:
            if p is not None:
                result.predictions.append(p)

    if result.predictions:
        _persist_run(result)
    return result
