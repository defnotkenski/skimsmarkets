from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import Counter
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
from skimsmarkets.agents.pricing import cost_usd
from skimsmarkets.agents.schemas import (
    DefensibilityAssessment,
    LensNotebook,
    MarketPrediction,
    TokenUsage,
)
from skimsmarkets.agents.sports._director_shared import PROMPT_VERSION
from skimsmarkets.agents.sports import resolve_lens_set
from skimsmarkets.agents.sports.base import LensSet, LensSpec
from skimsmarkets.agents.sports.tennis.deterministic_notebook import (
    is_tennis_event_rich_coverage,
)
from skimsmarkets.calibration import apply_temperature, load_temperature
from skimsmarkets.classify import classify_ev, classify_risk
from skimsmarkets.ev import compute_ev_per_dollar
from skimsmarkets.progress import ProgressReporter
from skimsmarkets.polymarket.enrichment import (
    enrich_clob_book,
    enrich_price_history,
)
from skimsmarkets.polymarket.models import PolymarketEvent
from skimsmarkets.polymarket.slate import (
    fetch_gamma_events,
    fetch_gamma_slate,
)
from skimsmarkets.selection import select_top_events
from skimsmarkets.tennis import (
    TennisGbtContext,
    TennisSimulationContext,
    TennisStatsContext,
)
from skimsmarkets.tennis.gbt import predict_for_event as gbt_predict_for_event
from skimsmarkets.tennis.identity import tennis_match_identity
from skimsmarkets.tennis.matchstat import _surname_candidates
from skimsmarkets.tennis.provider import (
    TennisStatsProvider,
    build_tennis_provider,
)
from skimsmarkets.tennis.simulation import simulate_for_event
from skimsmarkets.unusual_whales import (
    GammaTokenResolver,
    UnusualWhalesClient,
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
    # UW insider-tier counts captured at decision time, keyed event_id →
    # {"total": N, "notable": N, "smart": N}. Populated by
    # `_collect_uw_insider_counts` right after `resolve_uw_context` so the
    # counts exactly match what the director rendered. Events without a
    # UW context (offshore, no coverage, fetch failure) are absent from
    # the dict; `_get_uw_counts` returns Nones for missing keys. Drives
    # `scripts/uw_enrichment_retro.py` hit-rate-by-tier analysis.
    uw_insider_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    # Deterministic risk classification keyed event_id → (risk_bucket,
    # risk_score). Populated by `_persist_run` as it builds each JSONL row
    # — that's where the team_a-anchored gap to the market is computed, the
    # third classifier input. `reporting.py` reads this to group the
    # leaderboard by bucket without recomputing the gap. `risk_score` is
    # None (bucket `Unrated`) when the judge produced no defensibility score.
    risk_classifications: dict[str, tuple[str, float | None]] = field(
        default_factory=dict
    )
    # Parallel EV classification keyed event_id → (ev_bucket, ev_score).
    # Computed in `_persist_run` alongside the risk classifier so the
    # dual-mode reporting / sorting (`--sort-by ev`) and the executor's
    # `--mode ev` filter both have a per-event label without recomputing.
    # `ev_score` is None (bucket `Unrated`) when EV is uncomputable (model
    # or market probability missing, or market price at a degenerate edge).
    ev_classifications: dict[str, tuple[str, float | None]] = field(
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
    # Per-event token usage: event_id → list of TokenUsage records, one per
    # LLM call (fetcher per lens, reasoner per lens, director). Populated
    # alongside `lens_timings`. Persisted on the prediction row so retro
    # grading can correlate token cost vs predictive value per lens
    # (especially relevant given the form/matchup fetchers are confirmed
    # to do little web search beyond LLM-rewriting the structured block).
    token_usage: dict[str, list[TokenUsage]] = field(default_factory=dict)
    # Slate-level token usage: judge call. Persisted on the meta row.
    slate_token_usage: list[TokenUsage] = field(default_factory=list)


@dataclass
class _LensOutcome:
    """Internal: per-lens result of one event's two-stage chain."""

    lens: str
    notebook: LensNotebook | None = None
    report: BaseModel | None = None
    error_stage: str | None = None  # "fetcher", "reasoner", or "algo"
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
    # constant; CLI surfaces `--max-prob` for ad-hoc overrides without
    # editing config.py.
    max_implied_probability: float = cfg.MAX_IMPLIED_PROBABILITY
    # Minimum open interest (dollars at par) for at least one side of
    # an event. Defaults to the config constant; CLI surfaces `--min-oi`
    # for ad-hoc overrides. Set to 0 to disable.
    min_open_interest_dollars: float = cfg.MIN_OPEN_INTEREST_DOLLARS


async def fetch_slate(
    opts: SlateOptions,
    *,
    http: httpx.AsyncClient,
    gamma_sem: asyncio.Semaphore,
) -> list[PolymarketEvent]:
    """Build the unified Polymarket slate from gamma. Single source of
    truth used by both `run_pipeline` and the `skims fetch` CLI path so
    they can never drift.

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
            min_open_interest_dollars=opts.min_open_interest_dollars,
        )

    if opts.slugs:
        extra = await fetch_gamma_events(http, opts.slugs, gamma_sem)
        seen: set[str] = {ev.id for ev in events}
        for ev in extra:
            if ev.id in seen:
                continue
            seen.add(ev.id)
            events.append(ev)
    return events


async def overlay_matchstats_tipoffs(
    events: list[PolymarketEvent],
    provider: TennisStatsProvider,
) -> tuple[set[str], int]:
    """Overlay MatchStats per-match scheduled tipoffs onto each market's
    `game_start_time`.

    Returns `(matched_event_ids, total_fixtures_fetched)`:
      - `matched_event_ids` — the subset of input event IDs whose
        surname-pair matched a MatchStats fixture row. Used by the
        downstream `filter_unmatched_matchstats_events` step to drop
        events without comprehensive data coverage before the
        lens-chain spend kicks in.
      - `total_fixtures_fetched` — sum of fixture rows across every
        (tour, date) combo queried. Zero means the provider returned
        nothing for any combo (stub provider with no API key, or
        vendor outage) — the coverage filter treats zero as "can't
        judge, let everything through" rather than dropping the whole
        tennis slate.

    Why: gamma's `gameStartTime` is best-effort vendor data and can drift
    when tour officials reshuffle the session schedule late. MatchStats's
    `/tennis/v2/{tour}/fixtures/{date}` `date` field is sourced from
    tour-official schedules and is authoritative for tennis. The overlay
    runs before `apply_horizon_filter` so the time window cut operates
    on the most accurate available tipoff.

    Implementation:
      1. Build the set of unique `(tour, date_iso)` combinations
         needed across the slate (tour from slug prefix, date from
         the event's earliest market `game_start_time`).
      2. Fan out one `/fixtures/{date}` call per combination in
         parallel — bounded above by `tours × unique_dates`, which
         is 1-4 in practice for a 24-72h horizon.
      3. Index returned fixtures by surname-pair (frozenset) and
         look up each event's pair in the matching index.
      4. When a hit lands, overlay the MatchStats tipoff onto every
         market in the event. Silent fallback to the gamma value
         when the surname pair isn't in the index (minor matches,
         doubles, walkovers, last-minute reschedules).

    Pre-condition: `events` are `PolymarketEvent` instances whose
    `slug` carries the tennis canonical shape — `{tour}-{last_a}-{last_b}
    -{yyyy-mm-dd}` (or `{tour}-{tournament}-{last_a}-{last_b}-{yyyy-mm-dd}`
    with extra tokens between tour and surnames). `_event_surname_pair_candidates`
    reads the last two pre-date tokens, so either shape works.
    Post-condition: each market's `game_start_time` reflects the most
    accurate available source, NEVER less accurate than before.

    Stub provider returns empty dict from `fetch_fixtures_for_date` so
    this is a no-op when no MatchStats key is configured — pipeline
    still runs with the raw gamma tipoff.
    """
    needed: set[tuple[str, str]] = set()
    for ev in events:
        tours = _matchstat_tours_for_slug(ev.slug or "")
        if not tours:
            continue
        starts = [m.game_start_time for m in ev.markets if m.game_start_time]
        if not starts:
            continue
        date_iso = min(starts).strftime("%Y-%m-%d")
        for tour in tours:
            needed.add((tour, date_iso))
    if not needed:
        log.info("matchstats overlay: no eligible (tour, date) pairs")
        return set(), 0

    needed_list = sorted(needed)
    indexes = await asyncio.gather(
        *(provider.fetch_fixtures_for_date(tour=t, date_iso=d) for t, d in needed_list)
    )
    by_combo = dict(zip(needed_list, indexes, strict=True))
    total_fixtures = sum(len(idx) for idx in indexes)
    log.info(
        "matchstats overlay: fetched %d fixtures across %d (tour, date) combos",
        total_fixtures, len(needed_list),
    )

    matched_event_ids: set[str] = set()
    refreshed_tipoff = 0
    matched_events = 0
    # Per-reason miss tally + per-event debug lines. Enabled via `-v`,
    # this surfaces *why* each unmatched event missed (no_tipoff /
    # no_pair / not_in_index) so we can target the dominant cause
    # instead of guessing.
    miss_counts: Counter[str] = Counter()
    for ev in events:
        tours = _matchstat_tours_for_slug(ev.slug or "")
        if not tours:
            miss_counts["non_tennis"] += 1
            continue
        starts = [m.game_start_time for m in ev.markets if m.game_start_time]
        if not starts:
            miss_counts["no_tipoff"] += 1
            log.debug("matchstats miss: slug=%s reason=no_tipoff", ev.slug)
            continue
        date_iso = min(starts).strftime("%Y-%m-%d")
        pair_keys = _event_surname_pair_candidates(ev)
        if not pair_keys:
            miss_counts["no_pair"] += 1
            log.debug(
                "matchstats miss: slug=%s reason=no_pair detail=%s",
                ev.slug, _classify_pair_failure(ev),
            )
            continue
        # First-hit wins across the candidate tour indexes. ATP/WTA
        # slugs have exactly one candidate; ITF slugs have two (atp
        # then wta) since MatchStats serves ITF M-tier under atp and
        # ITF W-tier under wta — surname pairs are unique across
        # genders so there's no risk of a wrong-gender false match.
        # For each tour we also try every candidate pair key so
        # Hispanic / Iberian double-surname names match regardless of
        # which side abbreviated (paternal-only vs full paternal+maternal).
        fixture = None
        for tour in tours:
            index = by_combo.get((tour, date_iso))
            if not index:
                continue
            for pair_key in pair_keys:
                f = index.get(pair_key)
                if f is not None:
                    fixture = f
                    break
            if fixture is not None:
                break
        if fixture is None:
            miss_counts["not_in_index"] += 1
            log.debug(
                "matchstats miss: slug=%s reason=not_in_index "
                "tours=%s date=%s pairs_tried=%s",
                ev.slug, tours, date_iso,
                [sorted(p) for p in pair_keys],
            )
            continue
        matched_events += 1
        matched_event_ids.add(ev.id)
        # Tipoff overlay only fires when MatchStats has a confirmed
        # `date`. Early-round Challenger / ITF matches often ship with
        # `date=None` in the fixtures payload — the bracket exists but
        # tour officials haven't confirmed the slot yet. Fall through
        # to gamma's `gameStartTime` in that case.
        if fixture.date is not None:
            for i, m in enumerate(ev.markets):
                ev.markets[i] = m.model_copy(update={"game_start_time": fixture.date})
            refreshed_tipoff += 1
        # Player IDs are seeded into the rankings index as a side
        # effect of `fetch_fixtures_for_date` itself — no action
        # needed here. Subsequent `_resolve(tour, name)` lookups in
        # `enrich_tennis_stats` will find ITF players via these
        # seeded entries instead of returning None.
    log.info(
        "matchstats overlay: matched %d/%d events, refreshed tipoff on %d",
        matched_events, len(events), refreshed_tipoff,
    )
    if miss_counts:
        breakdown = ", ".join(
            f"{k}={v}" for k, v in sorted(miss_counts.items())
        )
        log.info("matchstats overlay: miss breakdown — %s", breakdown)
    return matched_event_ids, total_fixtures


def filter_unmatched_matchstats_events(
    events: list[PolymarketEvent],
    *,
    matched_event_ids: set[str],
    total_fixtures_fetched: int,
) -> list[PolymarketEvent]:
    """Drop tennis events whose surname-pair didn't match a MatchStats
    fixture row.

    Rationale: the tennis lens chain depends on MatchStats coverage
    for the player-stats context block (form, career percentages),
    sim inputs (career baseline serve/return), GBT priors (player ID
    lookups), and h2h. An unmatched event would still produce a
    prediction — the pipeline degrades gracefully per-event — but on
    materially thinner evidence, and the resulting confidence /
    defensibility scores would mislead the leaderboard. The gate
    keeps the ranker honest by dropping events MatchStats *could
    have* covered but didn't (surname transliteration variants,
    last-minute fixture additions, players outside the indexed feed).

    Scope: tennis only — non-tennis slugs (`mlb-`, `nba-`, etc.)
    never enter the MatchStats overlay and pass through untouched.

    Safety: when `total_fixtures_fetched == 0` the gate is skipped
    entirely. That covers the stub-provider case (no API key
    configured → empty fixtures) and vendor outages (MatchStats
    returns nothing across every combo). Treating "no data" as
    "can't judge, let everything through" prevents the gate from
    silently emptying the slate in cases we can't distinguish from
    genuine coverage failure.
    """
    if total_fixtures_fetched == 0:
        log.info(
            "matchstats coverage filter: no fixtures fetched, "
            "skipping gate (treating empty feed as can't-judge)"
        )
        return events

    kept: list[PolymarketEvent] = []
    dropped = 0
    for ev in events:
        tours = _matchstat_tours_for_slug(ev.slug or "")
        if not tours:
            # Non-tennis events bypass the gate.
            kept.append(ev)
            continue
        if ev.id in matched_event_ids:
            kept.append(ev)
        else:
            dropped += 1
    log.info(
        "matchstats coverage filter: kept %d events "
        "(dropped %d unmatched tennis events)",
        len(kept), dropped,
    )
    return kept


def apply_horizon_filter(
    events: list[PolymarketEvent],
    *,
    horizon_hours: int,
) -> list[PolymarketEvent]:
    """Drop events whose earliest market `game_start_time` falls outside
    `[now - cfg.HORIZON_BACKSTOP_HOURS, now + horizon_hours]`.

    Runs AFTER `overlay_matchstats_tipoffs` so the cut operates on the
    most accurate available tipoff (MatchStats per-match precision when
    present, raw gamma `gameStartTime` otherwise). Events without any
    market `game_start_time` after the overlay are dropped — the slate
    filter's no-tradable / no-tipoff drops still happen upstream; this
    stage only enforces the time window.

    The slate-level horizon cut in `polymarket/slate.py:fetch_gamma_slate`
    uses the same constant — this is the second pass, which can drop
    events the MatchStats overlay revealed to be outside the window.
    """
    now = datetime.now(tz=UTC)
    backstop = now - timedelta(hours=cfg.HORIZON_BACKSTOP_HOURS)
    horizon_end = now + timedelta(hours=horizon_hours)
    kept: list[PolymarketEvent] = []
    dropped = 0
    for ev in events:
        starts = [m.game_start_time for m in ev.markets if m.game_start_time]
        if not starts:
            dropped += 1
            continue
        tipoff = min(starts)
        if not (backstop <= tipoff <= horizon_end):
            dropped += 1
            continue
        kept.append(ev)
    log.info(
        "horizon filter: kept %d events, dropped %d outside [%dh back, +%dh]",
        len(kept), dropped, cfg.HORIZON_BACKSTOP_HOURS, horizon_hours,
    )
    return kept


def _matchstat_tours_for_slug(slug: str) -> list[str]:
    """MatchStats tour values to query for this event's slug.

    Polymarket uses `atp-` / `wta-` / `itf-` slug prefixes. MatchStats
    serves ATP main + men's Challengers + ITF M-tier futures under
    `tour=atp`, and WTA main + ITF W-tier futures under `tour=wta`.
    `tour=itf` is rejected by the API with a 400 ("Tour type is not
    valid"), so ITF events must be looked up against BOTH atp and wta
    fixture indexes — gender isn't encoded in the Polymarket slug.
    Surname pairs are unique across genders on any given date, so the
    first-hit lookup in `overlay_matchstats_tipoffs` is safe (no risk
    of pulling a women's fixture for a men's match or vice versa).

    Empty list for slugs we can't route (non-tennis, doubles with a
    different slug shape) — caller skips those events.
    """
    if slug.startswith("atp-"):
        return ["atp"]
    if slug.startswith("wta-"):
        return ["wta"]
    if slug.startswith("itf-"):
        return ["atp", "wta"]
    return []


def _event_surname_pair_candidates(
    ev: PolymarketEvent,
) -> list[frozenset[str]]:
    """All plausible `frozenset({surname_a, surname_b})` lookup keys for
    this event's YES/NO market labels.

    Reads the FULL player name from `yes_sub_title` on each side rather
    than parsing the slug — Polymarket truncates surnames to ~7 chars
    in the slug (`atp-virtane-perrica-2026-05-13` for Virtanen vs
    Perricard), which would miss the MatchStats index keyed on full
    surnames. `_surname_candidates` returns the last token plus, for
    3+ token names, the penultimate — covering Hispanic / Iberian
    double-surname abbreviation (Polymarket "Camila Osorio" vs
    MatchStats "Maria Camila Osorio Serrano"). The cross-product
    deduped pairs are what the overlay tries against the fixture index.

    Returns an empty list when either side lacks a `yes_sub_title`,
    either side's candidate list is empty, or every cross-product
    pair collides on the same surname.
    """
    yes_market = next(
        (m for m in ev.markets if not m.is_no_side and m.yes_sub_title),
        None,
    )
    no_market = next(
        (m for m in ev.markets if m.is_no_side and m.yes_sub_title),
        None,
    )
    if yes_market is None or no_market is None:
        return []
    candidates_a = _surname_candidates(yes_market.yes_sub_title)
    candidates_b = _surname_candidates(no_market.yes_sub_title)
    if not candidates_a or not candidates_b:
        return []
    seen: set[frozenset[str]] = set()
    out: list[frozenset[str]] = []
    for a in candidates_a:
        for b in candidates_b:
            if a == b:
                continue
            key = frozenset({a, b})
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def _classify_pair_failure(ev: PolymarketEvent) -> str:
    """Diagnostic for why `_event_surname_pair_candidates` returned an
    empty list — used only by the overlay's per-miss debug log.
    Mirrors the checks inside the candidates function so we can name
    the specific sub-cause instead of an opaque empty list.
    """
    yes_market = next(
        (m for m in ev.markets if not m.is_no_side and m.yes_sub_title),
        None,
    )
    no_market = next(
        (m for m in ev.markets if m.is_no_side and m.yes_sub_title),
        None,
    )
    if yes_market is None and no_market is None:
        return "no_subtitle_both"
    if yes_market is None:
        return "no_subtitle_yes"
    if no_market is None:
        return "no_subtitle_no"
    candidates_a = _surname_candidates(yes_market.yes_sub_title)
    candidates_b = _surname_candidates(no_market.yes_sub_title)
    if not candidates_a or not candidates_b:
        return f"surname_empty(yes={candidates_a!r}, no={candidates_b!r})"
    return (
        f"surname_collide(yes={candidates_a!r}, no={candidates_b!r})"
    )


async def resolve_uw_context(
    uw: UnusualWhalesClient,
    events: list[PolymarketEvent],
    sem: asyncio.Semaphore,
    *,
    resolver: GammaTokenResolver,
) -> None:
    """Attach UW wallet-flow context to each event by its native gamma slug.

    UW indexes wallet flow on Polymarket markets, keyed by the YES-side
    ERC-1155 `asset_id`. Since the slate is already gamma-sourced, the
    event slug is the lookup key directly — no cross-venue surname
    matching needed.

    For each event:
      1. Resolve the FIRST market's slug to `(yes_asset_id, no_asset_id)`
         via the gamma resolver (cache-shared with the CLOB book + history
         enrichers, so this is a free re-hit per slug).
      2. GET UW's `/predictions/market/{yes_asset_id}` for the compact
         flow snapshot.
      3. Attach if it carries an actionable signal — skip empty contexts
         (offshore markets, low-volume events) so the director's UW block
         isn't rendered as all `?`s.

    Silent degrade at every step — events with no matching gamma snapshot,
    no actionable UW signal, or any HTTP failure leave `uw_context=None`.
    UW disabled at the client level (no API key) is the same code path
    and short-circuits before any HTTP fires.

    Per-market gamma piggyback (`gamma_spread`, `gamma_one_day_price_
    change`, etc.) is populated INSIDE `PolymarketEvent.from_gamma`
    directly from the listing payload — no separate piggyback merge here.
    """
    if not uw.enabled:
        return

    by_slug: dict[str, list[PolymarketEvent]] = {}
    for ev in events:
        # Use the first market slug — gamma's `/markets?slug=` resolves on
        # per-market slugs, and head-to-head events expose their YES side
        # as the first market under `from_gamma` (the inverted NO clone
        # shares the same slug).
        first_market_slug = next((m.slug for m in ev.markets if m.slug), None)
        if first_market_slug:
            by_slug.setdefault(first_market_slug, []).append(ev)
    if not by_slug:
        return

    async def _one(slug: str, evs: list[PolymarketEvent]) -> None:
        async with sem:
            snap = await resolver.resolve_snapshot(slug)
            if snap is None or snap.clob_token_ids is None:
                return
            yes_asset_id, _no_asset_id = snap.clob_token_ids
            ctx = await uw.get_market_detail(yes_asset_id)
            if ctx is None or not ctx.has_actionable_signal():
                return
            # Profile-enrich `is_notable()` insiders (z >= 2). The per-asset
            # detail_agg gives us size-vs-baseline conviction (z) but not
            # edge quality (is_smart, win_rate); fetch each notable wallet's
            # trader profile to pair the two signals. Cache on the client
            # short-circuits duplicate wallets across events in the slate.
            # Profile fetch failures leave `insider.profile=None` — the
            # renderer treats that as "no edge data" and renders only the
            # size signal, which is current production behaviour.
            notable = [
                ins for ins in ctx.insiders
                if ins.is_notable() and ins.user_address
            ]
            if notable:
                profiles = await asyncio.gather(
                    *(uw.get_trader_profile(ins.user_address) for ins in notable),
                    return_exceptions=True,
                )
                for ins, prof in zip(notable, profiles):
                    if not isinstance(prof, BaseException) and prof is not None:
                        ins.profile = prof
            for e in evs:
                e.uw_context = ctx

    await asyncio.gather(*(_one(s, evs) for s, evs in by_slug.items()))
    attached = sum(1 for ev in events if ev.uw_context is not None)
    log.info(
        "attached unusual-whales context to %d/%d events",
        attached, len(events),
    )


def _collect_uw_insider_counts(
    events: list[PolymarketEvent],
) -> dict[str, dict[str, int]]:
    """Build {event_id: {total, notable, smart}} from attached UW contexts.

    Called immediately after `resolve_uw_context` so the counts reflect
    exactly what the director will render. Events without UW context are
    omitted from the dict (not zero-filled) so retro analysis can
    distinguish "no UW coverage" from "UW coverage with empty insider
    list" — both render the same to the director but mean different
    things upstream.
    """
    out: dict[str, dict[str, int]] = {}
    for ev in events:
        ctx = ev.uw_context
        if ctx is None:
            continue
        total = len(ctx.insiders)
        notable = sum(1 for i in ctx.insiders if i.is_notable())
        smart = sum(
            1 for i in ctx.insiders
            if i.profile is not None and i.profile.is_smart is True
        )
        out[ev.id] = {"total": total, "notable": notable, "smart": smart}
    return out


def _get_uw_counts(result: RunResult, event_id: str) -> dict[str, int | None]:
    """Lookup helper for `_persist_run` — returns a dict ready to spread
    into the prediction-row payload. Missing events get all-None values
    so the JSONL row carries explicit nulls (distinguishable from "no
    UW field at all" in older rows that predate the snapshot)."""
    counts = result.uw_insider_counts.get(event_id)
    if counts is None:
        return {
            "uw_insiders_total": None,
            "uw_insiders_notable": None,
            "uw_insiders_smart": None,
        }
    return {
        "uw_insiders_total": counts.get("total"),
        "uw_insiders_notable": counts.get("notable"),
        "uw_insiders_smart": counts.get("smart"),
    }


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
    per_event_tokens: list[TokenUsage],
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
        if spec.compute is not None:
            # Algorithmic path — no fetcher, no reasoner, no LLM. Synthesize
            # a placeholder LensNotebook so persistence + retro keep their
            # uniform per-lens shape.
            try:
                with _time_stage(per_event_timings, f"algo:{lens}"):
                    report = spec.compute(event)
            except Exception as e:  # noqa: BLE001
                return _LensOutcome(lens=lens, error_stage="algo", error=e)
            if report is None:
                return _LensOutcome(
                    lens=lens,
                    error_stage="algo",
                    error=RuntimeError(f"algorithmic lens {lens!r} returned None"),
                )
            notebook = LensNotebook(
                lens=lens,
                team_a_name=getattr(report, "team_a_name", "(unknown)"),
                team_b_name=getattr(report, "team_b_name", "(unknown)"),
                research_notes=(
                    f"Algorithmic lens {lens!r} — deterministic compute, no fetcher/reasoner."
                ),
                citations=[],
                computed_numbers=[],
                coverage="thin",
            )
            return _LensOutcome(lens=lens, notebook=notebook, report=report)

        # Fetcher-bypass path — when the lens has a deterministic_notebook
        # builder AND the flag is on AND the builder accepts this event
        # (rich coverage), skip the fetcher LLM call and go straight to
        # the reasoner (which has web_search + code_execution tools to
        # fill any residual gap, see agents/reasoners.py).
        bypass_notebook: LensNotebook | None = None
        if (
            cfg.FETCHER_BYPASS_ON_RICH_DATA
            and spec.deterministic_notebook is not None
        ):
            try:
                with _time_stage(per_event_timings, f"bypass_check:{lens}"):
                    bypass_notebook = spec.deterministic_notebook(event)
            except Exception:  # noqa: BLE001
                # Coverage check itself shouldn't fail per-event; if it
                # does, log and fall through to the fetcher path rather
                # than dropping the event.
                log.exception(
                    "deterministic_notebook builder crashed for lens=%s event=%s — "
                    "falling through to fetcher path",
                    lens, event.id,
                )
                bypass_notebook = None

        if bypass_notebook is not None:
            notebook = bypass_notebook
        else:
            try:
                async with fetcher_sem:
                    with _time_stage(per_event_timings, f"fetcher:{lens}"):
                        notebook = await provider.fetch(
                            event, lens, lens_set=lens_set,
                            token_sink=per_event_tokens,
                        )
            except Exception as e:  # noqa: BLE001
                return _LensOutcome(lens=lens, error_stage="fetcher", error=e)
        try:
            async with reasoner_sem:
                with _time_stage(per_event_timings, f"reasoner:{lens}"):
                    report = await run_reasoner(
                        anthropic, event, notebook, spec,
                        token_sink=per_event_tokens,
                    )
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


# ---------------------------------------------------------------------------
# Per-prediction derived metrics — pure functions over already-persisted
# fields. Promoted to top-level columns so retro queries don't have to
# walk into nested specialist_reports / notebooks / sim / GBT to ask
# "did the override pay off" / "did the lens stack track outcomes".
# ---------------------------------------------------------------------------

# Tennis lens-set stack composition. Used by `_compute_tennis_stack` to
# reconstruct the literal `baseline + sum(shifts)` the director was asked
# to apply, so retro can compute `stack_vs_final_delta`. List of
# (lens_name, field_name) pairs in the same order the synthesis tail
# spells out — keeping it data-driven means a future shift addition just
# adds a tuple here.
_TENNIS_STACK_SHIFTS: tuple[tuple[str, str], ...] = (
    ("tennis_form_and_surface", "form_signed_shift"),
    ("tennis_form_and_surface", "surface_signed_shift"),
    ("tennis_matchup_and_clutch", "h2h_signed_shift"),
    ("tennis_matchup_and_clutch", "clutch_signed_shift"),
    ("tennis_conditions_and_context", "physical_signed_shift"),
    ("tennis_conditions_and_context", "stakes_signed_shift"),
)


def _compute_tennis_stack(
    reports: dict[str, BaseModel],
    tennis_gbt: TennisGbtContext | None = None,
) -> tuple[float, float] | None:
    """Reconstruct the literal stack math for a tennis event.

    Returns (baseline, stack_team_a) where stack_team_a is the unclipped
    sum of baseline + all six signed shifts; the caller clips to [0,1].
    Returns None when neither GBT prior nor form lens baseline is
    available (only happens on a partial-failure event the director
    shouldn't have synthesised anyway — defensive).

    Baseline selection mirrors the director prompt
    (`DIRECTOR_SYSTEM_TENNIS_TAIL`, "GBT prior — THIS IS YOUR BASELINE
    ANCHOR" section, effective 2026-05-15): the GBT prior
    `p_team_a_wins` is the baseline when present; the form lens
    `team_a_win_probability` is the cold-start fallback. Earlier
    versions of this helper always used the form lens baseline, which
    silently desynchronised `stack_team_a_probability` from what the
    director was actually instructed to compute and made
    `stack_vs_final_delta` conflate a baseline-mismatch with real
    director overrides.
    """
    if tennis_gbt is not None and tennis_gbt.p_team_a_wins is not None:
        baseline = float(tennis_gbt.p_team_a_wins)
    else:
        form = reports.get("tennis_form_and_surface")
        if form is None:
            return None
        baseline_form = getattr(form, "team_a_win_probability", None)
        if baseline_form is None:
            return None
        baseline = float(baseline_form)
    total = baseline
    for lens_name, field_name in _TENNIS_STACK_SHIFTS:
        report = reports.get(lens_name)
        if report is None:
            continue
        v = getattr(report, field_name, None)
        if v is not None:
            total += float(v)
    return baseline, total


def _team_a_probability(
    p: MarketPrediction, reports: dict[str, BaseModel]
) -> float | None:
    """Re-orient `predicted_yes_probability` to team_a's frame.

    `predicted_yes_probability` is the probability of whoever the
    director picked. To compare against the stack (which is team_a-
    anchored) and the deterministic priors (also team_a-anchored), we
    flip when the picked winner is team_b. None when no report carries
    team_a_name (shouldn't happen under tennis but defensive).
    """
    team_a_name: str | None = None
    for r in reports.values():
        name = getattr(r, "team_a_name", None)
        if name:
            team_a_name = name
            break
    if not team_a_name:
        return None
    if p.predicted_winner.strip().lower() == team_a_name.strip().lower():
        return p.predicted_yes_probability
    return 1.0 - p.predicted_yes_probability


def _calibration_bucket(prob: float) -> str:
    """Bin `predicted_yes_probability` into 5pp buckets for calibration plots.

    Predictions naturally cluster in 0.50-1.00 (since predicted_winner is
    always picked above 50% by the contrarian-call discipline), but the
    bucket label is full-range so an underdog flip gets binned correctly.
    Uses inclusive-low / exclusive-high boundaries except the top bucket
    which is inclusive-high to keep p=1.0 in a real bucket.
    """
    if prob >= 0.95:
        return "0.95-1.00"
    lo = int(prob * 20) * 5
    return f"0.{lo:02d}-0.{lo + 5:02d}"


def _coverage_by_lens(
    notebooks: dict[str, LensNotebook],
) -> dict[str, str]:
    """Promote per-lens `notebook.coverage` to a flat top-level dict.

    Lets retro queries filter low-coverage events with a single column
    lookup (`jq '.lens_coverage["tennis_form_and_surface"]=="thin"'`)
    instead of walking into the notebook payload.
    """
    return {lens: nb.coverage for lens, nb in notebooks.items()}


def _aggregate_tokens(
    usage: list[TokenUsage],
) -> dict[str, float | int | None]:
    """Sum token usage across an event's LLM calls.

    Returns a dict with token totals + a dollar `cost_usd_total` and
    `n_calls`. Treats None tokens as 0 in the sum but reports them
    separately as `n_unknown_calls` so retro can flag SDK regressions
    that drop usage metadata.

    Cost is computed via `agents.pricing.cost_usd` for each call and
    summed. Calls whose model isn't in `MODEL_RATES` (non-Anthropic
    providers, or an unregistered Anthropic model) contribute 0 to the
    total and are counted in `n_unpriced_calls` so retro can see how
    much of the spend is missing from the cost estimate.
    """
    in_total = 0
    out_total = 0
    cache_write_total = 0
    cache_read_total = 0
    cost_total = 0.0
    n_unknown = 0
    n_unpriced = 0
    for u in usage:
        if u.input_tokens is None and u.output_tokens is None:
            n_unknown += 1
            continue
        in_total += u.input_tokens or 0
        out_total += u.output_tokens or 0
        cache_write_total += u.cache_creation_input_tokens or 0
        cache_read_total += u.cache_read_input_tokens or 0
        c = cost_usd(u)
        if c is None:
            n_unpriced += 1
        else:
            cost_total += c
    return {
        "input_total": in_total,
        "output_total": out_total,
        "cache_creation_input_total": cache_write_total,
        "cache_read_input_total": cache_read_total,
        "cost_usd_total": round(cost_total, 6),
        "n_calls": len(usage),
        "n_unknown_calls": n_unknown,
        "n_unpriced_calls": n_unpriced,
    }


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

    **Incomplete-run gate**: a run with predictions but no judge
    assessments is treated as incomplete and discarded. The judge is
    slate-level — if its call failed, every prediction in the slate
    would land with `defensibility_score: null`, which pollutes retro
    calibrate's "no judge" bucket without carrying any signal. Skipping
    persistence avoids that. The operator already saw the upstream error
    (judge stage logs the BadRequestError at runtime); the JSONL audit
    trail isn't worth the calibrate-side noise.
    """
    if result.predictions and not result.defensibility_assessments:
        log.warning(
            "skipping run-log persistence for %s: %d predictions but no "
            "judge assessments — treating as incomplete run.",
            result.run_id,
            len(result.predictions),
        )
        return
    try:
        _LOG_ROOT.mkdir(parents=True, exist_ok=True)
        path = _LOG_ROOT / f"{result.run_id}.jsonl"
        logged_at = datetime.now(UTC).isoformat()
        # Calibration temperature, loaded once per run. v1 is tennis-only —
        # every prediction row is tennis (the lens registry drops other
        # sports before they reach here), so a single lookup covers the
        # slate. When a second sport gets a fitted calibration this becomes
        # a per-row `load_temperature(p.sport_type)`.
        run_temperature = load_temperature("tennis")
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

                # Stack reconstruction (tennis-only for now; other sports
                # land here with stack=None until they declare their own
                # stacking math). When stack is present we also compute
                # the team_a-anchored final probability so all the prior
                # gaps are apples-to-apples.
                reports_typed = result.reports.get(p.event_id, {})
                stack_pair = (
                    _compute_tennis_stack(reports_typed, tgbt)
                    if p.lens_set_name == "tennis" else None
                )
                team_a_p_final = _team_a_probability(p, reports_typed)
                stack_baseline = stack_pair[0] if stack_pair else None
                stack_team_a = stack_pair[1] if stack_pair else None
                # Clip to the same [0,1] the director was asked to clip
                # to, so `stack_vs_final_delta` measures override
                # magnitude rather than clip-magnitude.
                stack_team_a_clipped = (
                    max(0.0, min(1.0, stack_team_a))
                    if stack_team_a is not None else None
                )
                stack_vs_final_delta = (
                    team_a_p_final - stack_team_a_clipped
                    if (team_a_p_final is not None
                        and stack_team_a_clipped is not None)
                    else None
                )
                # Director-discipline flag: True iff the director deviated
                # from the literal stack math (|delta| > 0.01, set above
                # the ~0.005 float-rounding floor the director introduces
                # when reporting probs to 2 decimal places) AND did NOT
                # log a `retracted_shifts` entry explaining which shift
                # was set aside. The director prompt explicitly requires
                # logging any deviation — `DIRECTOR_SYSTEM_TENNIS_TAIL`
                # reads "never just shrink the final number without an
                # entry." This flag catches silent violations of that
                # rule; retro can cut on it to measure whether unlogged
                # overrides correlate with outcomes. None on events
                # without a computable stack (non-tennis, or tennis
                # cold-start without baseline).
                override_without_retract = (
                    stack_vs_final_delta is not None
                    and abs(stack_vs_final_delta) > 0.01
                    and not p.retracted_shifts
                )
                # team_a-anchored gaps to deterministic priors. All three
                # are signed (positive = director above prior) so retro
                # queries can ask "are we systematically above the
                # market?" with a one-line aggregate. The market gap
                # needs flipping when the predicted winner is team_b
                # (polymarket_implied_probability is in the picked-
                # winner frame; sim/GBT are already team_a-anchored).
                ta_name = next(
                    (getattr(r, "team_a_name", None)
                     for r in reports_typed.values()
                     if getattr(r, "team_a_name", None)),
                    None,
                )
                predicted_winner_is_team_a = bool(
                    ta_name
                    and ta_name.strip().lower()
                    == p.predicted_winner.strip().lower()
                )
                # Director rule-compliance flags. Each mirrors a specific
                # rule in `DIRECTOR_SYSTEM_TENNIS_TAIL` and fires when
                # the director shipped a prediction violating it.
                # Deterministic — does NOT depend on the director
                # following the rule — so retro can measure compliance
                # rates over time and decide whether prompt-based
                # interventions are sticking. All four flags are
                # tennis-only and None on non-tennis or partial-failure
                # events; older runs (pre-2026-05-16) parse as None.
                #
                # (a) Injury-flag cap: any non-empty injury_concerns
                # entry should cap confidence at "low" (rule added
                # 2026-05-16). Fires when injury_concerns is present
                # AND confidence != "low".
                injury_concerns_present = False
                if p.lens_set_name == "tennis":
                    for r in reports_typed.values():
                        ic = getattr(r, "injury_concerns", None) or []
                        if ic:
                            injury_concerns_present = True
                            break
                confidence_should_be_low_injury = (
                    injury_concerns_present and p.confidence != "low"
                ) if p.lens_set_name == "tennis" else None
                # (b) Multi-shift stack cap: |shift_total| ≥ 0.10 AND
                # ≥2 shifts in the override direction (matching the
                # pick) each ≥ 0.04 should cap confidence at "low"
                # (rule added 2026-05-16). Fires when override structure
                # is met AND confidence != "low".
                shift_total_signed = 0.0
                shifts_in_override_dir = 0
                if p.lens_set_name == "tennis":
                    pick_sign = 1.0 if predicted_winner_is_team_a else -1.0
                    for lens_name, field_name in _TENNIS_STACK_SHIFTS:
                        r = reports_typed.get(lens_name)
                        if r is None:
                            continue
                        v = getattr(r, field_name, None)
                        if v is None:
                            continue
                        shift_total_signed += float(v)
                        if (float(v) * pick_sign) >= 0.04:
                            shifts_in_override_dir += 1
                confidence_should_be_low_stacked = (
                    abs(shift_total_signed) >= 0.10
                    and shifts_in_override_dir >= 2
                    and p.confidence != "low"
                ) if p.lens_set_name == "tennis" else None
                # (c) GBT-vs-sim split discipline: when GBT < 0.50 on
                # the pick AND sim ≥ 0.50 on the pick, the director
                # must cite the GBT `top_features` entry it believes
                # is mis-anchored (rule added 2026-05-16). Fires when
                # the split holds AND no top_features name appears in
                # `reasoning` (substring match, lowercase).
                gbt_sim_split_unjustified: bool | None = None
                if (p.lens_set_name == "tennis"
                        and tgbt is not None
                        and tsim is not None
                        and tgbt.p_team_a_wins is not None
                        and tsim.p_team_a_wins is not None):
                    gbt_pick = (
                        tgbt.p_team_a_wins if predicted_winner_is_team_a
                        else 1.0 - tgbt.p_team_a_wins
                    )
                    sim_pick = (
                        tsim.p_team_a_wins if predicted_winner_is_team_a
                        else 1.0 - tsim.p_team_a_wins
                    )
                    if gbt_pick < 0.50 and sim_pick >= 0.50:
                        reasoning_lc = (p.reasoning or "").lower()
                        feature_names = [
                            tf.name.lower() for tf in (tgbt.top_features or [])
                        ]
                        gbt_sim_split_unjustified = not any(
                            fn in reasoning_lc for fn in feature_names
                        )
                    else:
                        gbt_sim_split_unjustified = False
                if (team_a_p_final is not None
                        and p.polymarket_implied_probability is not None):
                    if predicted_winner_is_team_a:
                        market_team_a = p.polymarket_implied_probability
                    elif ta_name:
                        market_team_a = 1.0 - p.polymarket_implied_probability
                    else:
                        market_team_a = None
                    gap_to_market = (
                        team_a_p_final - market_team_a
                        if market_team_a is not None else None
                    )
                else:
                    gap_to_market = None
                # Pre-compute the gap-to-GBT first so the classifier can
                # consume it alongside gap-to-market. Sim/GBT prior values
                # are extracted here once; gap-to-sim is computed below
                # for the JSONL payload (the classifier doesn't use it).
                sim_p = (
                    tsim.p_team_a_wins if tsim is not None else None
                )
                gap_to_sim = (
                    team_a_p_final - sim_p
                    if (team_a_p_final is not None and sim_p is not None)
                    else None
                )
                gbt_p = (
                    tgbt.p_team_a_wins if tgbt is not None else None
                )
                gap_to_gbt = (
                    team_a_p_final - gbt_p
                    if (team_a_p_final is not None and gbt_p is not None)
                    else None
                )
                # Deterministic risk classifier — combines magnitude
                # (predicted probability), defensibility (judge), and TWO
                # convergence terms (gap to market the LLMs never saw,
                # gap to GBT prior the director uses as baseline anchor)
                # into one of four full-spectrum buckets. Stored on
                # `result` so `reporting.py` groups the leaderboard
                # without recomputing the gaps; also written to the JSONL
                # row below.
                risk_bucket, risk_score = classify_risk(
                    p.predicted_yes_probability,
                    da.defensibility_score if da is not None else None,
                    gap_to_market,
                    predicted_winner_is_team_a=predicted_winner_is_team_a,
                    temperature=run_temperature,
                    gap_to_gbt_signed=gap_to_gbt,
                )
                result.risk_classifications[p.event_id] = (
                    risk_bucket,
                    risk_score,
                )
                # Parallel EV classification — both bucketings run on every
                # event regardless of which the operator filters by at trade
                # time. Uses the rank-time Polymarket implied probability;
                # `skims execute --mode ev` re-grades the row against the
                # LIVE Kalshi ask via `compute_ev_per_dollar` (the persisted
                # bucket is the rank-time snapshot for sorting / reporting).
                ev_bucket, ev_score = classify_ev(
                    p.predicted_yes_probability,
                    p.polymarket_implied_probability,
                )
                result.ev_classifications[p.event_id] = (
                    ev_bucket,
                    ev_score,
                )

                # Per-event token usage rollup
                event_token_usage = result.token_usage.get(p.event_id, [])
                token_summary = _aggregate_tokens(event_token_usage)

                payload = {
                    # Discriminator — see function docstring for shapes.
                    # Listed first so `jq '.record_type'` is one cheap
                    # field-read per row when grouping.
                    "record_type": "prediction",
                    "run_id": result.run_id,
                    "logged_at_utc": logged_at,
                    # Bumped manually whenever any prompt or schema changes
                    # in a way that would alter director behaviour. Lets A/B
                    # analysis split before vs after a change without
                    # joining against git history.
                    "prompt_version": PROMPT_VERSION,
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
                    # True iff the director picked the same side as the
                    # market (predicted_yes_probability and
                    # polymarket_implied_probability share the same
                    # picked-winner frame) but with strictly lower
                    # probability than the market priced. Structurally
                    # weak: agreeing with consensus at lower conviction
                    # carries no informational edge. None when the
                    # market implied is missing.
                    "negative_edge": (
                        p.predicted_yes_probability
                        < p.polymarket_implied_probability
                        if p.polymarket_implied_probability is not None
                        else None
                    ),
                    # Pre-binned 5pp bucket of `predicted_yes_probability`
                    # so calibration-plot scripts can group rows with one
                    # column lookup instead of re-binning every time.
                    "predicted_probability_bucket": _calibration_bucket(
                        p.predicted_yes_probability
                    ),
                    "confidence": p.confidence,
                    "headline": p.headline,
                    # Derived synthesis metrics — promoted to top-level
                    # so retro queries don't have to walk specialist_reports
                    # to ask "did the override pay off" / "did the stack
                    # math track outcomes". `stack_team_a_probability` is
                    # the literal `baseline + sum(shifts)` clipped to
                    # [0,1]; `stack_vs_final_delta` is signed in the
                    # team_a frame (positive = director went above stack).
                    # The gap_to_* fields are also team_a-anchored
                    # (positive = director above prior). All five are
                    # null on non-tennis events until other sports declare
                    # a stacking math, and on events missing the relevant
                    # prior (sim/GBT cold-start gates).
                    "stack_baseline_team_a": stack_baseline,
                    "stack_team_a_probability": stack_team_a_clipped,
                    "team_a_p_final": team_a_p_final,
                    "stack_vs_final_delta": stack_vs_final_delta,
                    # Deterministic director-discipline flags. See
                    # `_persist_run` definitions above for each rule.
                    # All four parse as None on older runs and on
                    # non-tennis events.
                    "override_without_retract": override_without_retract,
                    "confidence_should_be_low_injury": confidence_should_be_low_injury,
                    "confidence_should_be_low_stacked": confidence_should_be_low_stacked,
                    "gbt_sim_split_unjustified": gbt_sim_split_unjustified,
                    "gap_to_market_signed": gap_to_market,
                    "gap_to_sim_signed": gap_to_sim,
                    "gap_to_gbt_signed": gap_to_gbt,
                    # Per-lens coverage tags promoted to a flat dict so
                    # `jq '.lens_coverage["tennis_form_and_surface"]=="thin"'`
                    # is one column lookup instead of a notebook walk.
                    "lens_coverage": _coverage_by_lens(
                        result.notebooks.get(p.event_id, {})
                    ),
                    # Per-event token-usage rollup + the per-call breakdown.
                    # Summary on top for quick aggregation; full list rides
                    # underneath for per-stage analysis (which lens cost
                    # what).
                    "token_usage_summary": token_summary,
                    "token_usage_calls": [
                        {**u.model_dump(mode="json"), "cost_usd": cost_usd(u)}
                        for u in event_token_usage
                    ],
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
                    # UW insider-tier counts at decision time, populated by
                    # `_collect_uw_insider_counts` immediately after the UW
                    # enrichment stage so the counts exactly match what the
                    # director rendered from. All three None when no UW
                    # context was attached (offshore market, no UW coverage,
                    # fetch failure) — see `RunResult.uw_insider_counts`.
                    **_get_uw_counts(result, p.event_id),
                    # EV per $1 staked on predicted-winner side. Analytical
                    # only — informs whether the picked side has asymmetric
                    # edge given Polymarket pricing (independent of whether
                    # Kalshi pricing matches at execution). None when either
                    # the model or market probability is missing.
                    "ev_per_dollar": compute_ev_per_dollar(
                        p.predicted_yes_probability,
                        p.polymarket_implied_probability,
                    ),
                    # Per-event audit log — populated by the director when
                    # it set aside one of the lens-emitted shifts because
                    # the lens's notebook didn't support the magnitude.
                    # Empty when the director accepted the literal stack
                    # math. Drives retro grading of "which shift gets
                    # retracted most" — a direct calibration signal for
                    # the offending reasoner.
                    "retracted_shifts": [
                        rs.model_dump(mode="json") for rs in p.retracted_shifts
                    ],
                    "defensibility_score": (
                        da.defensibility_score if da is not None else None
                    ),
                    "defensibility_rationale": (
                        da.defensibility_rationale if da is not None else None
                    ),
                    "defensibility_flags": (
                        da.defensibility_flags if da is not None else []
                    ),
                    # Deterministic risk classifier output — `risk_bucket` is
                    # the full-spectrum grade (Lock / Lean / Coin-flip /
                    # Avoid, or Unrated when the judge produced no score);
                    # `risk_score` is the continuous [0,1] composite it was
                    # cut from. Combines predicted probability, defensibility,
                    # and convergence to the (LLM-blind) market price — see
                    # `classify.py`.
                    "risk_bucket": risk_bucket,
                    "risk_score": risk_score,
                    # Parallel EV classification (see `classify_ev` in
                    # classify.py). Buckets: Prime / Edge / Thin / Negative
                    # / Unrated. Computed off the rank-time Polymarket
                    # implied probability; `skims execute --mode ev` uses
                    # this for slate-side filtering, and the trader's
                    # `--min-ev-threshold` re-grades against the live
                    # Kalshi ask for trade-time gating.
                    "ev_bucket": ev_bucket,
                    "ev_score": ev_score,
                    # Calibration audit — `calibration_temperature` is the
                    # scalar applied to the magnitude term this run (1.0 when
                    # no artefact is committed); `calibrated_winner_probability`
                    # is what the classifier's magnitude term actually saw.
                    # Raw `predicted_yes_probability` stays the source of truth
                    # for re-fitting; these two just record the live decision.
                    "calibration_temperature": run_temperature,
                    "calibrated_winner_probability": apply_temperature(
                        p.predicted_yes_probability, run_temperature
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
                    "prompt_version": PROMPT_VERSION,
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
            # Slate-level token aggregate: sum across every event's
            # per-call breakdown plus the judge's call. Promoted to the
            # meta row so a `jq '.token_usage_summary'` against the meta
            # record gives a one-line "what did this run cost in tokens"
            # without parsing prediction rows.
            all_calls = [
                u for ev_list in result.token_usage.values() for u in ev_list
            ] + result.slate_token_usage
            slate_summary = _aggregate_tokens(all_calls)
            meta_payload = {
                "record_type": "meta",
                "run_id": result.run_id,
                "logged_at_utc": logged_at,
                # See prediction-row docstring above; same field, same
                # meaning. Stamped on the meta record so retro grading
                # can A/B by version with one filter against meta rows.
                "prompt_version": PROMPT_VERSION,
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
                "token_usage_summary": slate_summary,
                "judge_token_usage_calls": [
                    {**u.model_dump(mode="json"), "cost_usd": cost_usd(u)}
                    for u in result.slate_token_usage
                ],
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
    token_usage_out: dict[str, list[TokenUsage]],
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
    per_event_tokens: list[TokenUsage] = []
    token_usage_out[event.id] = per_event_tokens
    pairs = await _run_lenses(
        provider, anthropic, event, lens_set,
        fetcher_sem, reasoner_sem, errors,
        per_event, per_event_tokens,
    )
    if pairs is None:
        return None
    notebooks, reports = pairs

    async with director_sem:
        try:
            with _time_stage(per_event, "director"):
                prediction = await synthesize_prediction(
                    anthropic, event, reports, lens_set,
                    token_sink=per_event_tokens,
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
    min_open_interest_dollars: float = cfg.MIN_OPEN_INTEREST_DOLLARS,
    slugs: list[str] | None = None,
    sports: list[str] | None = None,
    tennis_stats_disabled: bool = False,
    require_rich_stats: bool = False,
    progress: ProgressReporter | None = None,
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

    # `public_http` is a shared httpx client for the Polymarket vendors:
    #   - Gamma `/events` (slate listing), `/markets?slug=` (token-id
    #     resolution for UW + CLOB enrichment), `/events?slug=` (per-slug
    #     lookups).
    #   - CLOB `/book?token_id=...` + `/prices-history?market=...`
    #     (per-market enrichment).
    # Shared so the connection pool is reused across stages. Kalshi is
    # the execution venue (`skims execute`), not a data source — the
    # ranker doesn't open a Kalshi connection.
    async with (
        UnusualWhalesClient(config.unusual_whales_api_key) as uw,
        httpx.AsyncClient(timeout=20.0) as public_http,
        build_tennis_provider(config) as tennis_provider,
    ):
        gamma_resolver = GammaTokenResolver(public_http)
        # `fetch_slate` is the single source of truth used by both
        # `run_pipeline` and the `skims fetch` CLI path. Internally
        # it lists gamma events, applies league + horizon + tradability
        # filters, and returns `PolymarketEvent` objects so the
        # downstream pipeline (selection / lens dispatch / director /
        # judge / JSONL persistence / retro grading) is unchanged.
        slate_opts = SlateOptions(
            leagues=leagues or [],
            slugs=slugs or [],
            sports=sports or [],
            horizon_hours=horizon_hours,
            max_implied_probability=max_implied_probability,
            min_open_interest_dollars=min_open_interest_dollars,
        )
        if progress is not None:
            progress.start("slate")
        with _time_stage(stage_timings, "fetch_slate"):
            events = await fetch_slate(
                slate_opts, http=public_http, gamma_sem=gamma_sem
            )
        result.fetched_events = len(events)
        if progress is not None:
            progress.complete("slate")

        # MatchStats tipoff overlay — refines gamma's `gameStartTime`
        # with per-match precision from `/tennis/v2/{tour}/fixtures/
        # {date}` where available. Runs BEFORE the horizon filter so
        # the cut uses the more accurate source. Stub provider no-ops
        # (returns empty fixtures dict), so this is free when no
        # MatchStats key is configured — horizon filter then operates
        # on the raw gamma tipoff.
        with _time_stage(stage_timings, "overlay_matchstats_tipoffs"):
            matched_event_ids, total_fixtures = (
                await overlay_matchstats_tipoffs(events, tennis_provider)
            )

        # MatchStats coverage gate — drop tennis events whose
        # surname-pair didn't find a fixture row. Runs BEFORE the
        # horizon filter so the smaller event list flows into the
        # downstream selector warmup + lens chains. Safe under stub
        # provider / vendor outage: when no fixtures were fetched
        # at all, the gate skips itself and nothing is dropped.
        with _time_stage(stage_timings, "filter_unmatched_matchstats"):
            events = filter_unmatched_matchstats_events(
                events,
                matched_event_ids=matched_event_ids,
                total_fixtures_fetched=total_fixtures,
            )

        # Horizon filter — runs AFTER the overlay so the cut uses the
        # most accurate available tipoff per market.
        with _time_stage(stage_timings, "apply_horizon_filter"):
            events = apply_horizon_filter(events, horizon_hours=horizon_hours)

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

        # Enrichment stages. UW context, CLOB book, and CLOB price
        # history all ride the shared `gamma_resolver` cache for
        # slug → token_id lookups — first stage to touch a slug pays
        # the gamma round-trip, subsequent stages hit the cache.
        if progress is not None:
            progress.start("enrich")
        with _time_stage(stage_timings, "enrich_uw"):
            await resolve_uw_context(
                uw,
                events,
                uw_sem,
                resolver=gamma_resolver,
            )
            # Snapshot insider-tier counts right after enrichment so they
            # exactly match what the director will render — any later
            # mutation of `uw_context` would diverge the persisted counts
            # from the LLM-visible reality.
            result.uw_insider_counts = _collect_uw_insider_counts(events)
        # CLOB orderbook — top-of-book size, depth, full-book $ totals.
        # One HTTP per unique market slug; NO-side clones swap bid/ask
        # sides off the same YES-token book.
        with _time_stage(stage_timings, "enrich_clob_book"):
            await enrich_clob_book(events, gamma_resolver, public_http, clob_sem)
        # Optional CLOB price-history enrichment — opt-in via the
        # `CLOB_HISTORY_ENABLED` constant in `config.py`. When off,
        # this is a no-op (zero `/prices-history` calls).
        if cfg.CLOB_HISTORY_ENABLED:
            with _time_stage(stage_timings, "enrich_clob_history"):
                await enrich_price_history(
                    events, gamma_resolver, public_http, clob_sem,
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
        # Opt-in quality filter: when `--require-rich-stats` is set,
        # drop tennis events whose structured tennis_stats block lacks
        # rich coverage for BOTH lenses (form_and_surface AND
        # matchup_and_clutch). Non-tennis events pass through unchanged.
        # Runs immediately after `enrich_tennis_stats` so the dropped
        # events don't pay the pure-CPU sim + GBT costs downstream. The
        # selector ran earlier capped to MAX_SLATE_EVENTS — this filter
        # can shrink the slate further; document in `--require-rich-stats`
        # help text that the user should bump MAX_SLATE_EVENTS if they
        # need a fuller post-filter slate.
        if require_rich_stats:
            with _time_stage(stage_timings, "filter_rich_stats"):
                pre_n = len(events)
                events = [e for e in events if is_tennis_event_rich_coverage(e)]
                dropped = pre_n - len(events)
                log.info(
                    "rich-stats filter: dropped %d/%d, %d remain",
                    dropped, pre_n, len(events),
                )
                # Keep `considered_events` in sync — downstream reporting
                # reads this to show "slate size after pre-LLM gates."
                result.considered_events = len(events)
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
        if progress is not None:
            progress.complete("enrich")
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
        if progress is not None:
            progress.start("predict", total=len(events))
        try:
            with _time_stage(stage_timings, "process_events"):
                # Schedule each event as a task so we can attach a
                # done-callback that advances the progress bar the
                # moment that event's `process_event` resolves. With a
                # plain `asyncio.gather(*coros)` we'd only see the bar
                # jump when the slowest event finished — useless for
                # the most variable phase of the pipeline.
                tasks = [
                    asyncio.create_task(
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
                            result.token_usage,
                        )
                    )
                    for e in events
                ]
                if progress is not None:
                    for t in tasks:
                        t.add_done_callback(
                            lambda _t: progress.advance("predict")
                        )
                outcomes = await asyncio.gather(*tasks)
        finally:
            await provider.aclose()
        if progress is not None:
            progress.complete("predict")

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
            if progress is not None:
                progress.start("judge")
            try:
                with _time_stage(stage_timings, "judge"):
                    judgment = await judge_slate(
                        anthropic, result.predictions,
                        token_sink=result.slate_token_usage,
                    )
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
            finally:
                if progress is not None:
                    progress.complete("judge")

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
