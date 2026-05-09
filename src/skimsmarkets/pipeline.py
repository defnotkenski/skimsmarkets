from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from anthropic import AsyncAnthropic
from pydantic import BaseModel

from skimsmarkets import config as cfg
from skimsmarkets.agents.director import synthesize_prediction
from skimsmarkets.agents.fetchers import FetcherProvider, build_provider
from skimsmarkets.agents.judge import judge_slate
from skimsmarkets.agents.reasoners import run_reasoner
from skimsmarkets.agents.schemas import (
    DefensibilityAssessment,
    LensNotebook,
    MarketPrediction,
)
from skimsmarkets.agents.sports import resolve_lens_set
from skimsmarkets.agents.sports.base import LensSet, LensSpec
from skimsmarkets.clob import (
    fetch_book,
    fetch_price_history,
    invert_sparkline,
    summarize_book,
    summarize_history,
)
from skimsmarkets.polymarket.models import PolymarketEvent
from skimsmarkets.selection import select_top_events
from skimsmarkets.tennis import (
    TennisGbtContext,
    TennisSimulationContext,
    TennisStatsContext,
)
from skimsmarkets.tennis.gbt import predict_for_event as gbt_predict_for_event
from skimsmarkets.tennis.identity import tennis_match_identity
from skimsmarkets.tennis.provider import (
    TennisStatsProvider,
    build_tennis_provider,
)
from skimsmarkets.tennis.simulation import simulate_for_event
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
    stage: str  # "fetcher:<lens>" / "reasoner:<lens>" / "director" / "lens_dispatch" / "tennis_stats" / "judge"
    error: str
    # `sport_type` is captured at error creation so JSONL retro-analysis can
    # group drops by sport (e.g. `jq '.stage=="lens_dispatch" | .sport_type'`).
    # None for slate-level errors (`event_id="*"`, e.g. judge failures) or for
    # events where sport_type wasn't resolved.
    sport_type: str | None = None


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
    # Reports are typed as `BaseModel` cross-pipeline because per-sport
    # lens sets emit per-sport report schemas (no closed union). The
    # per-sport director path is the only place that knows the concrete
    # types — pipeline plumbing just stores and serializes via
    # `model_dump`.
    reports: dict[str, dict[str, BaseModel]] = field(default_factory=dict)
    # Per-event tennis stats vendor payload (when present). Keyed
    # event_id → TennisStatsContext. Populated in `enrich_tennis_stats`
    # for ATP/WTA singles head-to-heads only; non-tennis / no-key runs
    # leave this empty. Persisted as a top-level JSONL field next to
    # `notebooks` / `specialist_reports` so retro grading can ask "did
    # the API have the right facts?" separately from "did the fetcher
    # use them?".
    tennis_stats: dict[str, TennisStatsContext] = field(default_factory=dict)
    # Per-event Monte Carlo simulation result, keyed event_id →
    # TennisSimulationContext. Populated in `enrich_tennis_simulation`
    # after `enrich_tennis_stats` ships the inputs. Director-only — the
    # same persistence posture as `tennis_stats` so retro grading can
    # ask "did the sim track the market or the director better in
    # hindsight?" as a separate question from the director's read.
    tennis_simulation: dict[str, TennisSimulationContext] = field(
        default_factory=dict
    )
    # Per-event GBT prediction, keyed event_id → TennisGbtContext.
    # Populated in `enrich_tennis_gbt` after `enrich_tennis_simulation`.
    # Director-only — same persistence posture as `tennis_simulation`
    # so retro grading can ask "did the GBT prior track outcomes
    # better than the sim or the director?" as a separate question.
    # Empty when no GBT artefact / parquet have been built (no spike
    # training has occurred yet) — silent degrade, no error rows.
    tennis_gbt: dict[str, TennisGbtContext] = field(default_factory=dict)
    # Slate-level judge output keyed event_id → DefensibilityAssessment.
    # Populated by `judge_slate` after all per-event directors finish; left
    # empty when the judge call fails (leaderboard then falls back to the
    # legacy predicted-probability sort). Same persistence posture as
    # `notebooks` / `reports` — best-effort, never aborts a run.
    defensibility_assessments: dict[str, DefensibilityAssessment] = field(
        default_factory=dict
    )
    # Per-stage wall-clock timings (seconds), populated by `run_pipeline`
    # right before persist. Persisted to JSONL as a `record_type="meta"`
    # row so post-hoc bottleneck attribution doesn't depend on captured
    # stderr. Empty on direct-construction paths (tests) that don't go
    # through `run_pipeline`.
    stage_timings: dict[str, float] = field(default_factory=dict)
    total_seconds: float = 0.0
    # Per-event lens-chain timings: event_id → stage_name → seconds.
    # Stage names are `fetcher:<lens>`, `reasoner:<lens>`, and `director`.
    # Populated incrementally inside `process_event` so a partial dict is
    # left behind when an event drops mid-chain (the dropped event's row
    # has whichever stages completed before the error). Used to attribute
    # the `process_events` wall time across lenses; the high-level
    # `stage_timings["process_events"]` is still the gather wall clock,
    # which is dominated by the slowest event.
    lens_timings: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class _LensOutcome:
    """Internal: per-lens result of one event's two-stage chain."""

    lens: str
    notebook: LensNotebook | None = None
    report: BaseModel | None = None
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
    # Favorite-blowout threshold on the YES mid. Defaults to the config
    # constant; CLI surfaces `--max-implied-prob` for ad-hoc overrides
    # without editing config.py.
    max_implied_probability: float = cfg.MAX_IMPLIED_PROBABILITY


async def fetch_gamma_slate(
    http: httpx.AsyncClient,
    leagues: list[str],
    horizon_hours: int,
    *,
    sports: list[str] | None = None,
    max_implied_probability: float = cfg.MAX_IMPLIED_PROBABILITY,
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

    The `MAX_SLATE_EVENTS` cap is NOT applied here. It moved to the
    pre-LLM `selection.select_top_events` stage in `run_pipeline`, which
    ranks candidates by *fundamental imbalance* (player-rank ratio for
    tennis, win-pct delta for team sports) instead of by tipoff. This
    function returns the full filtered slate sorted by tipoff so the
    `skims fetch` CLI path (which has no LLM cost to manage) sees an
    uncapped slate, and so the selection stage has the full population
    to score over. Callers that want the soonest-N can slice the result
    directly.
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
        if favorite_mid >= max_implied_probability:
            dropped_blowout += 1
            continue
        if len(live_markets) != len(ev.markets):
            ev = ev.model_copy(update={"markets": live_markets})
        kept.append(ev)

    log.info(
        "kept %d gamma events after league + horizon + tradability + "
        "blowout (>=%.2f) filters; dropped %d as blowouts",
        len(kept),
        max_implied_probability,
        dropped_blowout,
    )

    # Sort by earliest market tipoff ascending. Sort is unconditional
    # because gamma's listing is ordered by `endDate` (settlement
    # window), not tipoff — the two diverge on tours like ATP where
    # settlement lags match end by days. The cap that used to live here
    # moved to `selection.select_top_events` (called from
    # `run_pipeline`); this function now returns the full filtered slate
    # so the selection stage can score the entire population by
    # fundamental imbalance rather than just the soonest-N. Events
    # without any market `game_start_time` sort last.
    _far_future = datetime.max.replace(tzinfo=UTC)

    def _tipoff(ev: PolymarketEvent) -> datetime:
        starts = [t for m in ev.markets if (t := m.game_start_time) is not None]
        return min(starts) if starts else _far_future

    kept.sort(key=_tipoff)
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
            http,
            opts.leagues,
            opts.horizon_hours,
            sports=opts.sports,
            max_implied_probability=opts.max_implied_probability,
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
                        sport_type=event.sport_type,
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


def enrich_tennis_simulation(
    events: list[PolymarketEvent],
    errors: list[ErrorRecord],
) -> None:
    """Attach `event.tennis_simulation` for tennis events whose
    `tennis_stats` carries the career serve/return primitives the sim
    needs.

    Pure-CPU work: the sim runs against data already on the event
    (no HTTP, no semaphore). Sub-second per event for the default
    10k-trial count, so there's no need to fan out concurrently —
    a simple sync loop keeps the pipeline ordering predictable.
    Failure at the per-event scope records an
    `ErrorRecord(stage="tennis_simulation")` and leaves
    `tennis_simulation=None` on that event; the run continues with
    other events unaffected. Same fail-silent posture as the
    enrichment stages above.

    Director-only by design — see CLAUDE.md and
    `TennisSimulationContext` docstring. Lenses don't see this
    attachment.
    """
    if not events:
        return
    attached = 0
    for event in events:
        if event.tennis_stats is None:
            # Not a tennis event with vendor data, or vendor returned
            # empty — sim has nothing to compute against.
            continue
        try:
            ctx = simulate_for_event(event.tennis_stats, slug=event.slug)
        except Exception as e:  # noqa: BLE001
            errors.append(
                ErrorRecord(
                    event_id=event.id,
                    stage="tennis_simulation",
                    error=f"{type(e).__name__}: {e}",
                    sport_type=event.sport_type,
                )
            )
            log.warning(
                "tennis_simulation failed for %s: %s",
                event.id,
                type(e).__name__,
            )
            continue
        if ctx is not None:
            event.tennis_simulation = ctx
            attached += 1
    log.info(
        "attached tennis simulation to %d/%d events (career-baseline iid)",
        attached,
        len(events),
    )


def enrich_tennis_gbt(
    events: list[PolymarketEvent],
    errors: list[ErrorRecord],
) -> None:
    """Attach `event.tennis_gbt` for tennis events whose `tennis_stats`
    carries the player MatchStat ids the GBT predictor needs.

    Pure-CPU (no HTTP) — same posture as `enrich_tennis_simulation`.
    The GBT predictor is responsible for its own gating: missing
    artefact / parquet, cold-start (< MIN_PRIORS_PER_SIDE), or
    unresolvable player ids all produce None silently. Failure at the
    per-event scope records an `ErrorRecord(stage="tennis_gbt")` and
    leaves `tennis_gbt=None` on that event; the run continues with
    other events unaffected. Same fail-silent posture as the
    simulation enrichment above.

    Director-only by design — see CLAUDE.md and `TennisGbtContext`
    docstring. Lenses don't see this attachment.
    """
    if not events:
        return
    attached = 0
    for event in events:
        if event.tennis_stats is None:
            continue
        try:
            ctx = gbt_predict_for_event(event)
        except Exception as e:  # noqa: BLE001
            errors.append(
                ErrorRecord(
                    event_id=event.id,
                    stage="tennis_gbt",
                    error=f"{type(e).__name__}: {e}",
                    sport_type=event.sport_type,
                )
            )
            log.warning(
                "tennis_gbt failed for %s: %s",
                event.id,
                type(e).__name__,
            )
            continue
        if ctx is not None:
            event.tennis_gbt = ctx
            attached += 1
    log.info(
        "attached tennis GBT prior to %d/%d events",
        attached,
        len(events),
    )


async def _run_lenses(
    provider: FetcherProvider,
    anthropic: AsyncAnthropic,
    event: PolymarketEvent,
    lens_set: LensSet,
    fetcher_sem: asyncio.Semaphore,
    reasoner_sem: asyncio.Semaphore,
    errors: list[ErrorRecord],
    per_event_timings: dict[str, float],
) -> tuple[dict[str, LensNotebook], dict[str, BaseModel]] | None:
    """Run every lens declared by `lens_set` for one event. Each lens is a
    provider fetcher (Stage A) → Claude reasoner (Stage B) chain; lenses
    run in parallel, fetcher→reasoner is sequential within a lens.

    `fetcher_sem` is released between stages so a slow fetcher search loop
    doesn't tie up a fetcher slot through the (typically faster) Claude
    reasoner call. Per-event failure posture is unchanged from the legacy
    pipeline: any failure at either stage of any lens drops the event so
    the director never receives a partial set of reports.

    Per-sport-lens-set refactor: iterates `lens_set.lenses` (a tuple of
    `LensSpec`) instead of the legacy `REASONERS` dict; the reasoner
    helper is the generic `run_reasoner(anthropic, event, notebook, spec)`
    rather than per-lens dispatch.
    """

    async def _one(spec: LensSpec) -> _LensOutcome:
        lens = spec.name
        try:
            async with fetcher_sem:
                with _time_stage(per_event_timings, f"fetcher:{lens}"):
                    notebook = await provider.fetch(
                        event, lens, lens_set=lens_set
                    )
        except Exception as e:  # noqa: BLE001
            return _LensOutcome(lens=lens, error_stage="fetcher", error=e)
        try:
            async with reasoner_sem:
                with _time_stage(per_event_timings, f"reasoner:{lens}"):
                    report = await run_reasoner(anthropic, event, notebook, spec)
        except Exception as e:  # noqa: BLE001
            return _LensOutcome(
                lens=lens, notebook=notebook, error_stage="reasoner", error=e
            )
        return _LensOutcome(lens=lens, notebook=notebook, report=report)

    outcomes = await asyncio.gather(*(_one(spec) for spec in lens_set.lenses))
    notebooks: dict[str, LensNotebook] = {}
    reports: dict[str, BaseModel] = {}
    failed = False
    for o in outcomes:
        if o.error is not None:
            errors.append(
                ErrorRecord(
                    event_id=event.id,
                    stage=f"{o.error_stage}:{o.lens}",
                    error=f"{type(o.error).__name__}: {o.error}",
                    sport_type=event.sport_type,
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
    """Write predictions AND per-event drops to a per-run JSONL.

    One file per run named `<run_id>.jsonl`. Best-effort: any I/O failure is
    logged and swallowed — persistence must never abort the run, matching the
    enrichment-stage posture.

    Three row shapes share the file, distinguished by a top-level
    `record_type` field:
      - `record_type="prediction"` — one row per ranked event with the
        director's synthesis, judge's defensibility score, full lens
        notebooks + reasoner reports.
      - `record_type="error"` — one row per dropped event with `event_id`,
        `stage` (e.g. `fetcher:tennis_form_and_surface`,
        `reasoner:tennis_matchup_and_clutch`, `director`, `tennis_stats`,
        `judge`), and the captured error string.
      - `record_type="meta"` — one row per run with the per-stage
        wall-clock timings, slate counts, and total seconds. Lets
        bottleneck attribution be a `jq` query rather than depending on
        captured stderr.
    All three share the run-level metadata (`run_id`, `logged_at_utc`,
    `fetcher_provider`, `fetcher_model`) so a grading script can `jq`
    over the slate without joining against a sidecar.

    Why JSONL not parquet: lines are easy to tail, easy to grep, easy to feed
    into a future grading script that joins against gamma settlement after
    kickoff. Volume is tiny (≈one line per ranked event + a handful of error
    rows on a bad slate).
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
                # Career-baseline Monte Carlo sim — top-level too so
                # retro grading can ask "did the long-run baseline track
                # outcomes better than the director?" without joining
                # against a sidecar.
                tsim = result.tennis_simulation.get(p.event_id)
                # GBT third prior — same persistence posture as the sim.
                # Retro grading can ask GBT-vs-sim-vs-director-vs-market
                # as four separate questions without joining.
                tgbt = result.tennis_gbt.get(p.event_id)
                # `lens_names` is derived from the keys of the prediction's
                # specialist_reports (which match the LensSet's declared
                # lens names by construction). Stable order via sorting
                # is fine for jq filtering even though the LensSet itself
                # has an ordered tuple.
                lens_names_for_row = sorted(reports_for_event.keys())
                payload = {
                    # Discriminator — see function docstring for shapes.
                    # Listed first so `jq '.record_type'` is one cheap
                    # field-read per row when grouping.
                    "record_type": "prediction",
                    "run_id": result.run_id,
                    "logged_at_utc": logged_at,
                    # Run-level fetcher metadata — top-level (not nested in
                    # `notebooks`) so retrospective A/B grading can group
                    # rows by provider via `jq '.fetcher_provider'`.
                    "fetcher_provider": result.fetcher_provider,
                    "fetcher_model": result.fetcher_model,
                    "event_id": p.event_id,
                    "event_title": p.event_title,
                    # Sport / lens-set metadata at the top level so jq
                    # filters can group by sport without reaching into
                    # `notebooks` keys: `jq 'select(.sport_type=="tennis")'`,
                    # `jq 'select(.lens_set_name=="tennis")'`, or
                    # `jq '.lens_names[]' | sort | uniq -c` for distribution
                    # across the slate.
                    "sport_type": p.sport_type,
                    "lens_set_name": p.lens_set_name,
                    "lens_names": lens_names_for_row,
                    "market_slug": p.market_slug,
                    "predicted_winner": p.predicted_winner,
                    "predicted_yes_probability": p.predicted_yes_probability,
                    "polymarket_implied_probability": p.polymarket_implied_probability,
                    "confidence": p.confidence,
                    "headline": p.headline,
                    # Director synthesis fields — `reasoning` is the 3-6
                    # sentence rationale, `specialist_weights` shows how
                    # the three lenses were weighted, `disagreements_flagged`
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
                    "tennis_simulation": (
                        tsim.model_dump(mode="json") if tsim is not None else None
                    ),
                    "tennis_gbt": (
                        tgbt.model_dump(mode="json") if tgbt is not None else None
                    ),
                    "notebooks": notebooks_for_event,
                    "specialist_reports": reports_for_event,
                }
                f.write(json.dumps(payload, separators=(",", ":")) + "\n")
            # Error rows — one per dropped event. Useful for measuring
            # provider-specific drop rate (`jq 'select(.record_type=="error"
            # and .fetcher_provider=="gemini") | .stage'`), the stage
            # distribution (which lens fails most), and tracking
            # error-message classes (Gemini STOP-truncations vs MAX_TOKENS
            # vs schema parse failures vs reasoner timeouts).
            for err in result.errors:
                error_payload = {
                    "record_type": "error",
                    "run_id": result.run_id,
                    "logged_at_utc": logged_at,
                    "fetcher_provider": result.fetcher_provider,
                    "fetcher_model": result.fetcher_model,
                    "event_id": err.event_id,
                    # Top-level so `jq 'select(.stage=="lens_dispatch") | .sport_type'`
                    # works for analyzing dropped-event distributions across
                    # sports without reaching into nested fields.
                    "sport_type": err.sport_type,
                    "stage": err.stage,
                    "error": err.error,
                }
                f.write(json.dumps(error_payload, separators=(",", ":")) + "\n")
            # Run-level meta — single row per file. `stage_timings` is the
            # bottleneck-attribution data structure; `total_seconds` is
            # the wall-clock from pipeline entry to just-before-this-write
            # (the persist row itself isn't included in `stage_timings`
            # because we're inside the persist call). Empty `stage_timings`
            # on direct-construction paths (tests / non-orchestrator
            # callers) is harmless — the row still records counts.
            meta_payload = {
                "record_type": "meta",
                "run_id": result.run_id,
                "logged_at_utc": logged_at,
                "fetcher_provider": result.fetcher_provider,
                "fetcher_model": result.fetcher_model,
                "fetched_events": result.fetched_events,
                "considered_events": result.considered_events,
                "n_predictions": len(result.predictions),
                "n_errors": len(result.errors),
                "total_seconds": result.total_seconds,
                "stage_timings": result.stage_timings,
                # Per-event lens-chain breakdown — `event_id` →
                # `fetcher:<lens>` / `reasoner:<lens>` / `director` →
                # seconds. Use `jq` to attribute `process_events` wall
                # time across lenses, e.g.:
                #   jq 'select(.record_type=="meta") | .lens_timings'
                "lens_timings": result.lens_timings,
            }
            f.write(json.dumps(meta_payload, separators=(",", ":")) + "\n")
        log.info(
            "persisted %d predictions and %d errors to %s",
            len(result.predictions),
            len(result.errors),
            path,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("run-log persistence failed: %s", e)


async def process_event(
    provider: FetcherProvider,
    anthropic: AsyncAnthropic,
    event: PolymarketEvent,
    lens_set: LensSet,
    fetcher_sem: asyncio.Semaphore,
    reasoner_sem: asyncio.Semaphore,
    director_sem: asyncio.Semaphore,
    errors: list[ErrorRecord],
    lens_timings_out: dict[str, dict[str, float]],
) -> tuple[
    MarketPrediction, dict[str, LensNotebook], dict[str, BaseModel]
] | None:
    """Run the full agent chain for one event under its sport's `lens_set`.

    Returns the prediction alongside the per-lens notebooks and reasoner
    reports so the caller can persist them to the run JSONL. Returns None
    when the event was dropped (any lens stage failure or director failure).

    `lens_timings_out` is a per-run accumulator: keyed by `event_id`, each
    value is a dict of `fetcher:<lens>` / `reasoner:<lens>` / `director`
    → seconds. Registered eagerly here so a dropped event still leaves
    behind whatever stages completed before the error — useful for
    attributing "which lens failed slowly" vs "which lens failed fast".
    Concurrent writes from sibling `process_event` tasks are safe because
    asyncio is single-threaded and each task writes only under its own
    `event.id` key.
    """
    log.info(
        "processing event %s sport=%s lens_set=%s (%s)",
        event.id, event.sport_type, lens_set.sport, event.title,
    )
    per_event: dict[str, float] = {}
    lens_timings_out[event.id] = per_event
    pairs = await _run_lenses(
        provider, anthropic, event, lens_set,
        fetcher_sem, reasoner_sem, errors,
        per_event,
    )
    if pairs is None:
        return None
    notebooks, reports = pairs

    async with director_sem:
        try:
            with _time_stage(per_event, "director"):
                prediction = await synthesize_prediction(
                    anthropic, event, reports, lens_set
                )
        except Exception as e:  # noqa: BLE001
            errors.append(
                ErrorRecord(
                    event_id=event.id,
                    stage="director",
                    error=f"{type(e).__name__}: {e}",
                    sport_type=event.sport_type,
                )
            )
            return None
    return prediction, notebooks, reports


@contextmanager
def _time_stage(timings: dict[str, float], name: str):
    """Record wall-clock time for a pipeline stage into `timings`.

    Sync context manager — works inside `async` code because awaiting
    inside a `with` block is fine; `__enter__`/`__exit__` themselves
    don't need to be async. Same instance can wrap synchronous calls
    (`enrich_tennis_simulation`) and async calls (`fetch_slate`,
    `enrich_*`, the per-event `asyncio.gather`).

    `timings` accumulates with `+=` so the same name can be wrapped
    twice (e.g. enrichment stages that branch on a config flag) without
    clobbering — though no current caller relies on that. Diagnostic
    only — never mutates pipeline behaviour, never raises.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - t0)


def _format_stage_timings(
    timings: dict[str, float], total_seconds: float
) -> str:
    """Format the per-stage breakdown as a single info-level line.

    Sorted descending by elapsed seconds so the bottleneck is at the
    front; rows with `<1ms` cost are dropped (lens_dispatch on small
    slates) to keep the line readable. Each entry is `name=Xs (P%)`
    where P is the share of total wall time, so a quick eyeball tells
    you both the absolute cost AND whether it's a meaningful fraction
    of the run.
    """
    items = [
        (name, dt) for name, dt in timings.items() if dt >= 0.001
    ]
    items.sort(key=lambda x: -x[1])
    parts = []
    for name, dt in items:
        pct = (100.0 * dt / total_seconds) if total_seconds > 0 else 0.0
        parts.append(f"{name}={dt:.2f}s ({pct:.0f}%)")
    return f"total={total_seconds:.2f}s | " + " ".join(parts)


def _aggregate_lens_timings(
    lens_timings: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Sum per-event lens-stage timings into across-the-slate totals.

    Returns a flat `stage → total_seconds` dict (e.g.
    `fetcher:tennis_form_and_surface=420.5`) where each value is the
    SUM across every event's lens chain. Note: this is CPU-seconds (or
    rather wait-seconds), NOT wall-clock — the per-event chains run in
    parallel via `asyncio.gather`, so the sum is what you'd see if the
    work were serialised. Use this to ask "which lens stage burns the
    most cumulative work across the slate?"; pair with the per-event
    detail in `lens_timings` to ask "is one EVENT dragging the gather?"
    """
    totals: dict[str, float] = {}
    for per_event in lens_timings.values():
        for stage_name, dt in per_event.items():
            totals[stage_name] = totals.get(stage_name, 0.0) + dt
    return totals


def _format_lens_aggregate(totals: dict[str, float]) -> str:
    """Single line: `lens-stage totals (across N events)`, sorted desc."""
    items = [(n, dt) for n, dt in totals.items() if dt >= 0.001]
    items.sort(key=lambda x: -x[1])
    return " ".join(f"{n}={dt:.1f}s" for n, dt in items)


async def run_pipeline(
    *,
    leagues: list[str] | None = None,
    horizon_hours: int = cfg.DEFAULT_HORIZON_HOURS,
    max_implied_probability: float = cfg.MAX_IMPLIED_PROBABILITY,
    slugs: list[str] | None = None,
    sports: list[str] | None = None,
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

    The fetcher provider runs the per-lens Stage A for every event in the
    slate. Reasoner / director / judge are always Claude regardless. The
    provider choice is hand-edited in `config.py` (`FETCHER_PROVIDER`
    constant, default `gemini`) — no env var, no per-invocation override.
    """
    config = cfg.Config.from_env(
        tennis_stats_disabled=tennis_stats_disabled,
    )
    run_id = uuid.uuid4().hex[:8]
    result = RunResult(run_id=run_id)

    # Per-stage wall-clock breakdown — diagnostic only, dumped to log
    # at the end of the run with the bottleneck at the front. Cost is
    # ~12 `time.perf_counter()` calls per run, effectively free.
    stage_timings: dict[str, float] = {}
    pipeline_t0 = time.perf_counter()

    def _snapshot_timings() -> None:
        """Mirror the in-flight `stage_timings` dict + total elapsed onto
        `result` so `_persist_run` can write them into the meta row.

        Called immediately before each `_persist_run` site so the meta
        row carries the most up-to-date breakdown short of the persist
        call itself. The persist time is not retroactively folded back
        in (it lands in `stage_timings` after `_persist_run` returns,
        which the on-disk meta row already won't have observed).
        """
        result.stage_timings = dict(stage_timings)
        result.total_seconds = time.perf_counter() - pipeline_t0

    def _emit_timings() -> None:
        """Dump the per-stage breakdown. Inner closure so the early-exit
        paths (empty slate post-select, all-dropped post-lens_dispatch)
        can call it before returning without duplicating the formatting.

        Two lines: the high-level pipeline stages (fetch_slate, select,
        enrich_*, process_events, judge, persist), then a second line
        aggregating per-lens-stage CPU-seconds summed across events.
        Skip the second line when no per-event chains ran (no events
        survived to `process_events`, or all dropped at lens_dispatch).
        """
        log.info(
            "pipeline timings: %s",
            _format_stage_timings(
                stage_timings, time.perf_counter() - pipeline_t0
            ),
        )
        if result.lens_timings:
            totals = _aggregate_lens_timings(result.lens_timings)
            log.info(
                "lens-stage totals (across %d events, summed): %s",
                len(result.lens_timings),
                _format_lens_aggregate(totals),
            )

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
            max_implied_probability=max_implied_probability,
        )
        with _time_stage(stage_timings, "fetch_slate"):
            events = await fetch_slate(
                slate_opts, http=uw.http, gamma_sem=gamma_sem
            )
        result.fetched_events = len(events)

        # Pre-LLM selection — score by fundamental imbalance (player-rank
        # ratio for tennis, win-pct delta for team-record sports) and
        # cap to MAX_SLATE_EVENTS. Replaces the legacy "soonest-tipoff"
        # cap that lived in `fetch_gamma_slate`. Tipoff is preserved as
        # the tiebreaker so events without a stat-based signal (futures,
        # niche sports) still order soonest-first among themselves.
        # Tennis-event scoring requires the matchstat rankings index to
        # be warm; `select_top_events` warms it lazily on first use, so
        # non-tennis-only slates pay no warmup cost.
        with _time_stage(stage_timings, "select"):
            events = await select_top_events(
                events,
                max_events=cfg.MAX_SLATE_EVENTS,
                tennis_provider=tennis_provider,
            )

        result.considered_events = len(events)
        log.info("considering %d events", len(events))

        if not events:
            _emit_timings()
            return result

        # Lens-set dispatch — strict-declaration: events whose `sport_type`
        # has no registered LensSet drop with `ErrorRecord(stage=
        # "lens_dispatch")` BEFORE the enrichment fan-out, so we don't
        # pay UW / CLOB book / CLOB price-history / tennis-stats HTTP for
        # events we'll never process. Keep `events` as the dispatchable
        # subset for the rest of the run.
        dispatchable: list[PolymarketEvent] = []
        lens_sets_by_event: dict[str, LensSet] = {}
        for ev in events:
            ls = resolve_lens_set(ev)
            if ls is None:
                result.errors.append(
                    ErrorRecord(
                        event_id=ev.id,
                        stage="lens_dispatch",
                        error=(
                            f"no lens set registered for sport_type="
                            f"{ev.sport_type!r}"
                        ),
                        sport_type=ev.sport_type,
                    )
                )
                continue
            dispatchable.append(ev)
            lens_sets_by_event[ev.id] = ls
        dropped_dispatch = len(events) - len(dispatchable)
        if dropped_dispatch:
            log.info(
                "lens_dispatch dropped %d/%d events with no registered lens set",
                dropped_dispatch, len(events),
            )
        events = dispatchable

        if not events:
            # Whole slate dropped at lens_dispatch. Persist whatever
            # error rows accumulated before bailing.
            if result.errors:
                _snapshot_timings()
                with _time_stage(stage_timings, "persist"):
                    _persist_run(result)
            _emit_timings()
            return result

        # Enrichment stages all share the same `gamma_resolver` so token-ID
        # lookups are paid for at most once per slug across UW + CLOB book +
        # CLOB price-history. UW runs first because it can drop events with
        # no actionable signal; book/history then enrich whatever remains.
        with _time_stage(stage_timings, "enrich_uw"):
            await resolve_unusual_whales(
                uw, events, uw_sem, resolver=gamma_resolver
            )
        # CLOB book — top-of-book size, depth, full-book $ totals. Adds
        # one HTTP per unique market slug (deduplicated). Replaces the
        # legacy US `markets.book(...)` BBO refresh; the CLOB endpoint
        # exposes the full global book so book-$ totals here are 5–20×
        # what the legacy US slice used to show.
        with _time_stage(stage_timings, "enrich_clob_book"):
            await enrich_clob_book(events, gamma_resolver, public_http, clob_sem)
        # Optional CLOB price-history enrichment — opt-in via the
        # `CLOB_HISTORY_ENABLED` constant in `config.py`. Adds 1 HTTP per
        # unique slug. When off, this is a no-op (zero CLOB calls).
        if cfg.CLOB_HISTORY_ENABLED:
            with _time_stage(stage_timings, "enrich_clob_history"):
                await enrich_price_history(
                    events, gamma_resolver, public_http, clob_sem
                )
        else:
            log.info("clob price history disabled (cfg.CLOB_HISTORY_ENABLED=False)")
        # Tennis stats run LAST among enrichers — gated per-event by sport,
        # so deferring its work means non-tennis events have already paid
        # all the upstream gamma/CLOB enrichment costs we want anyway.
        # The provider is always non-None (factory returns the stub when no
        # key is configured), so this is safe without an `if enabled` branch.
        with _time_stage(stage_timings, "enrich_tennis_stats"):
            await enrich_tennis_stats(
                tennis_provider, events, tennis_sem, result.errors
            )
        # Career-baseline Monte Carlo sim — pure CPU on the inputs
        # `enrich_tennis_stats` just attached. Director-only feed, so
        # this MUST run before the lens chain so the director's
        # per-event context block can render the simulation block
        # alongside the UW block. Lens fetchers don't see it.
        with _time_stage(stage_timings, "enrich_tennis_sim"):
            enrich_tennis_simulation(events, result.errors)
        # GBT prior — third deterministic prior alongside market + sim.
        # Pure-CPU; reads the historical parquet built by
        # `skims gbt backfill` and the catboost artefact built by
        # `skims gbt train`. Silent degrade when either is missing
        # (fresh checkout, no spike training yet).
        with _time_stage(stage_timings, "enrich_tennis_gbt"):
            enrich_tennis_gbt(events, result.errors)
        # Snapshot the per-event tennis context onto the RunResult so
        # `_persist_run` can write it to the JSONL row even though the
        # event itself isn't carried into persistence (only the resulting
        # `MarketPrediction` is). Keyed by event id to match how
        # `notebooks` / `reports` line up against `predictions`.
        for ev in events:
            if ev.tennis_stats is not None:
                result.tennis_stats[ev.id] = ev.tennis_stats
            if ev.tennis_simulation is not None:
                result.tennis_simulation[ev.id] = ev.tennis_simulation
            if ev.tennis_gbt is not None:
                result.tennis_gbt[ev.id] = ev.tennis_gbt

        provider = build_provider(config.fetcher_provider, config)
        result.fetcher_provider = provider.name
        result.fetcher_model = provider.model
        log.info(
            "fetcher provider=%s model=%s", provider.name, provider.model
        )
        anthropic = AsyncAnthropic(api_key=config.anthropic_api_key)
        try:
            with _time_stage(stage_timings, "process_events"):
                outcomes = await asyncio.gather(
                    *(
                        process_event(
                            provider,
                            anthropic,
                            e,
                            lens_sets_by_event[e.id],
                            fetcher_sem,
                            reasoner_sem,
                            director_sem,
                            result.errors,
                            result.lens_timings,
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
                with _time_stage(stage_timings, "judge"):
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

    # Persist when there's anything to write — predictions OR drops. An
    # all-failed run (every event hit a fetcher/reasoner/director error)
    # still produces useful telemetry: the error rows tell us WHY the slate
    # collapsed, which the terminal Errors table loses to scrollback.
    if result.predictions or result.errors:
        _snapshot_timings()
        with _time_stage(stage_timings, "persist"):
            _persist_run(result)

    _emit_timings()
    return result
