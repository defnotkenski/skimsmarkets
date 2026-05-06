"""Step 1 — gamma resolution sidecar.

For each prediction row in a run log, query gamma `/events?slug=` to
determine whether the moneyline market settled and on which side.
Writes `logs/runs/<run_id>.resolutions.jsonl` (one row per event).

Idempotent: a rerun reads the existing sidecar and only fetches events
whose slug isn't already present (or is present with `settled=False`,
giving still-pending markets a chance to resolve on the next pass).

Concurrency: gamma is unauthed and tolerant; we cap at a small semaphore
to avoid hammering the public endpoint. The pipeline already uses a
similar cap (`GAMMA_FETCH_SEM`) for the slate fetch path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from skimsmarkets import config as cfg
from skimsmarkets.retro.jsonl import (
    iter_predictions,
    resolutions_sidecar_path,
    run_path_for_id,
)
from skimsmarkets.retro.models import PredictionRow, ResolvedOutcome
from skimsmarkets.unusual_whales.gamma import fetch_gamma_event

log = logging.getLogger(__name__)

# Tight cap — gamma is rate-limit-friendly but we don't need to hammer
# it. Run-level retros usually touch 5-30 events; a sweep across all
# logs may touch hundreds, but each event resolves to one HTTP call.
_RESOLVE_CONCURRENCY = cfg.GAMMA_FETCH_SEM


def _norm_name(name: str) -> str:
    """Lowercase + diacritic-stripped + single-spaced.

    Mirrors `tennis/matchstat._normalize_name` so the same comparison
    semantics apply across the retro layer. Used to match the
    director's `predicted_winner` (verbatim from gamma's `team_a_name`
    plumbing) against the resolved gamma outcome name.
    """
    nfkd = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(stripped.lower().split())


def _parse_outcome_pair(raw: Any) -> list[Any] | None:
    """Gamma ships `outcomes` and `outcomePrices` as JSON-encoded strings.

    Returns the parsed list, or None when the field is absent / malformed.
    """
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
    else:
        parsed = raw
    if isinstance(parsed, list):
        return parsed
    return None


def _settled_pair(prices: list[Any]) -> bool:
    """A market is settled when one outcome resolves to ~1 and the other
    to ~0. Cancellations / walkovers resolve 50-50 (`["0.5","0.5"]`)
    which we treat as unsettled — no winner to credit.
    """
    if len(prices) < 2:
        return False
    floats: list[float] = []
    for p in prices:
        try:
            floats.append(float(p))
        except (TypeError, ValueError):
            return False
    has_high = any(abs(p - 1.0) < 1e-6 for p in floats)
    has_low = any(abs(p - 0.0) < 1e-6 for p in floats)
    return has_high and has_low


def _winning_index(prices: list[Any]) -> int | None:
    """Index of the outcome whose price resolved to ~1."""
    for i, p in enumerate(prices):
        try:
            if abs(float(p) - 1.0) < 1e-6:
                return i
        except (TypeError, ValueError):
            continue
    return None


def _find_moneyline_market(
    event_payload: dict[str, Any], slug: str
) -> dict[str, Any] | None:
    """Find the moneyline market in a gamma event response.

    Convention (verified by probing): the moneyline market shares the
    event's slug exactly. Sub-markets (set winners, totals, handicaps,
    etc.) all carry suffixed slugs (`<event_slug>-first-set-winner-...`,
    `<event_slug>-match-total-21pt5`, etc.) and never collide with the
    bare event slug. Returns None when no moneyline is present (rare —
    happens on bundle events that ship only sub-markets, which the
    pipeline already filters out at fetch time).
    """
    for raw in event_payload.get("markets") or []:
        if not isinstance(raw, dict):
            continue
        if raw.get("slug") == slug:
            return raw
    return None


def _resolve_one_payload(
    row: PredictionRow, event_payload: dict[str, Any] | None
) -> ResolvedOutcome:
    """Build a `ResolvedOutcome` from a fetched gamma event payload.

    Pure function — does no I/O. Splits the resolution-decision logic
    out of the async fetch so it's directly testable.
    """
    now = datetime.now(UTC)
    if event_payload is None:
        return ResolvedOutcome(
            event_id=row.event_id,
            slug=row.market_slug,
            closed=False,
            settled=False,
            skip_reason="gamma /events?slug=… returned no record",
            resolved_at_utc=now,
        )

    moneyline = _find_moneyline_market(event_payload, row.market_slug)
    if moneyline is None:
        return ResolvedOutcome(
            event_id=row.event_id,
            slug=row.market_slug,
            closed=bool(event_payload.get("closed")),
            settled=False,
            skip_reason="no moneyline market matching the event slug",
            resolved_at_utc=now,
        )

    closed = bool(moneyline.get("closed"))
    uma_status = moneyline.get("umaResolutionStatus")
    prices = _parse_outcome_pair(moneyline.get("outcomePrices")) or []
    outcomes = _parse_outcome_pair(moneyline.get("outcomes")) or []

    # Settled requires three signals to align: market is closed, UMA
    # has marked it resolved, AND prices are an honest 0/1 pair (not
    # 50-50 cancellation, not still-trading near-the-money).
    settled = (
        closed
        and uma_status == "resolved"
        and _settled_pair(prices)
    )
    if not settled:
        return ResolvedOutcome(
            event_id=row.event_id,
            slug=row.market_slug,
            closed=closed,
            settled=False,
            skip_reason=(
                f"unsettled: closed={closed} uma={uma_status!r} "
                f"prices={prices}"
            ),
            resolved_at_utc=now,
        )

    win_idx = _winning_index(prices)
    if win_idx is None or win_idx >= len(outcomes):
        return ResolvedOutcome(
            event_id=row.event_id,
            slug=row.market_slug,
            closed=closed,
            settled=False,
            skip_reason="settled but couldn't pair winner index to outcomes list",
            resolved_at_utc=now,
        )
    winning_team = outcomes[win_idx]
    if not isinstance(winning_team, str):
        return ResolvedOutcome(
            event_id=row.event_id,
            slug=row.market_slug,
            closed=closed,
            settled=False,
            skip_reason=f"winning outcome wasn't a string: {winning_team!r}",
            resolved_at_utc=now,
        )

    # Side semantics: gamma's outcome at index 0 is "yes" by convention
    # (the first listed team in `marketSides`), index 1 is "no". The
    # `predicted_winner` string is normalised before comparison to
    # tolerate diacritic / casing drift between gamma and the director.
    winning_side = "yes" if win_idx == 0 else "no"
    predicted_correct: bool | None
    pred_norm = _norm_name(row.predicted_winner)
    win_norm = _norm_name(winning_team)
    other_team = outcomes[1 - win_idx] if len(outcomes) > 1 else None
    other_norm = (
        _norm_name(other_team) if isinstance(other_team, str) else None
    )
    if pred_norm == win_norm:
        predicted_correct = True
    elif other_norm is not None and pred_norm == other_norm:
        predicted_correct = False
    else:
        # Predicted name doesn't match either outcome — surface the
        # ambiguity rather than guess. Common cause: the director
        # picked a third-side label on a non-binary event that slipped
        # past the moneyline filter (very rare).
        return ResolvedOutcome(
            event_id=row.event_id,
            slug=row.market_slug,
            closed=closed,
            settled=True,
            winning_side=winning_side,
            winning_team_name=winning_team,
            predicted_correct=None,
            skip_reason=(
                f"predicted_winner={row.predicted_winner!r} did not match "
                f"either outcome {outcomes!r}"
            ),
            resolved_at_utc=now,
        )

    return ResolvedOutcome(
        event_id=row.event_id,
        slug=row.market_slug,
        closed=closed,
        settled=True,
        winning_side=winning_side,
        winning_team_name=winning_team,
        predicted_correct=predicted_correct,
        resolved_at_utc=now,
    )


def _read_existing_sidecar(path: Path) -> dict[str, ResolvedOutcome]:
    """Load an existing sidecar keyed by slug. Empty when absent."""
    out: dict[str, ResolvedOutcome] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                outcome = ResolvedOutcome.model_validate(payload)
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "retro resolve: skipping malformed sidecar line in %s: %s",
                    path.name, e,
                )
                continue
            out[outcome.slug] = outcome
    return out


async def _fetch_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    row: PredictionRow,
) -> ResolvedOutcome:
    async with sem:
        payload = await fetch_gamma_event(client, row.market_slug)
    return _resolve_one_payload(row, payload)


async def resolve_run(run_id: str) -> Path:
    """Resolve every prediction in `<run_id>.jsonl` and write the sidecar.

    Idempotent: events whose slug already has a `settled=True` outcome
    in the sidecar are not re-fetched. Unsettled events ARE re-fetched
    on rerun (their state may have changed). Returns the sidecar path.
    """
    run_path = run_path_for_id(run_id)
    if not run_path.exists():
        raise FileNotFoundError(f"run log not found: {run_path}")
    sidecar_path = resolutions_sidecar_path(run_path)
    existing = _read_existing_sidecar(sidecar_path)

    rows = list(iter_predictions(run_path))
    # Dedupe by slug — a single binary head-to-head ships YES + NO
    # clones with the same slug; we resolve once per slug.
    by_slug: dict[str, PredictionRow] = {}
    for row in rows:
        by_slug.setdefault(row.market_slug, row)

    to_fetch = [
        row for slug, row in by_slug.items()
        if not (existing.get(slug) and existing[slug].settled)
    ]
    if not to_fetch:
        log.info(
            "retro resolve: %s — %d events, all already resolved",
            run_id, len(by_slug),
        )
        return sidecar_path

    sem = asyncio.Semaphore(_RESOLVE_CONCURRENCY)
    async with httpx.AsyncClient(timeout=20.0) as client:
        results = await asyncio.gather(
            *(_fetch_one(client, sem, row) for row in to_fetch)
        )

    # Merge: existing outcomes for slugs we didn't refetch, new outcomes
    # for everything else. Newer attempts overwrite older unsettled ones.
    merged: dict[str, ResolvedOutcome] = dict(existing)
    for outcome in results:
        merged[outcome.slug] = outcome

    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with sidecar_path.open("w") as f:
        for slug in sorted(merged):
            f.write(merged[slug].model_dump_json() + "\n")

    settled_count = sum(1 for o in merged.values() if o.settled)
    correct_count = sum(
        1 for o in merged.values() if o.predicted_correct is True
    )
    log.info(
        "retro resolve: %s — %d events, %d settled, %d correct (%s)",
        run_id, len(merged), settled_count, correct_count, sidecar_path.name,
    )
    return sidecar_path


async def resolve_all_runs() -> list[Path]:
    """Resolve every run log under `logs/runs/`. Returns sidecar paths."""
    from skimsmarkets.retro.jsonl import list_run_files

    paths = list_run_files()
    sidecars: list[Path] = []
    for path in paths:
        run_id = path.stem
        try:
            sidecars.append(await resolve_run(run_id))
        except Exception as e:  # noqa: BLE001
            log.warning("retro resolve: %s failed: %s", run_id, e)
    return sidecars
