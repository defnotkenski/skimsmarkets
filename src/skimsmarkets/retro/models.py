"""Pydantic models for retro / self-improvement persistence + LLM I/O.

Three families:
  - `PredictionRow` — re-parses one `record_type="prediction"` JSONL row
    written by `pipeline._persist_run`. The writer assembles rows inline
    (no single Pydantic dump), so this is the canonical reader; mirrors
    the writer's field list.
  - `ResolvedOutcome` — one row in `logs/runs/<run_id>.resolutions.jsonl`,
    Step 1's output. Idempotent sidecar keyed by `slug`.
  - `EventFeatures` / `RetroFindings` — feature row for Steps 2/3 + the
    LLM's typed pattern-finding output.

`extra="ignore"` everywhere so a future writer that adds fields doesn't
break older retro runs (and vice versa).
"""

from __future__ import annotations

from datetime import date as _date_t
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from skimsmarkets.tennis.models import (
    PerMatchStats,
    TennisGbtContext,
    TennisSimulationContext,
    TennisStatsContext,
)


class PredictionRow(BaseModel):
    """One `record_type="prediction"` JSONL row from a run log.

    Field set mirrors `pipeline._persist_run` exactly. Nested vendor
    payloads (`tennis_stats`, `tennis_simulation`) are eager-validated
    here so downstream code can treat them as typed objects rather than
    dicts.
    """

    model_config = ConfigDict(extra="ignore")

    record_type: Literal["prediction"]
    run_id: str
    logged_at_utc: datetime
    fetcher_provider: str | None = None
    fetcher_model: str | None = None
    event_id: str
    event_title: str | None = None
    sport_type: str | None = None
    lens_set_name: str | None = None
    lens_names: list[str] = Field(default_factory=list)
    market_slug: str
    predicted_winner: str
    predicted_yes_probability: float
    polymarket_implied_probability: float | None = None
    confidence: Literal["low", "medium", "high"]
    headline: str | None = None
    reasoning: str | None = None
    specialist_weights: dict[str, float] = Field(default_factory=dict)
    disagreements_flagged: list[str] = Field(default_factory=list)
    uw_flow_note: str | None = None
    defensibility_score: float | None = None
    defensibility_rationale: str | None = None
    defensibility_flags: list[str] = Field(default_factory=list)
    tennis_stats: TennisStatsContext | None = None
    tennis_simulation: TennisSimulationContext | None = None
    tennis_gbt: TennisGbtContext | None = None
    # `notebooks` and `specialist_reports` are intentionally NOT decoded
    # to typed objects — their concrete shapes vary by lens set and
    # carrying the full `LensSpec.report_schema` resolution into the
    # retro layer would couple it to every sport. Leave as raw dicts;
    # callers that need the typed forms can re-validate via the lens
    # registry themselves.
    notebooks: dict[str, dict[str, Any]] = Field(default_factory=dict)
    specialist_reports: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ResolvedOutcome(BaseModel):
    """One row in `logs/runs/<run_id>.resolutions.jsonl` (Step 1 output).

    `settled` distinguishes "trading window closed" from "outcome paid":
    a market can have `closed=True` while gamma still shows two-sided
    bid/ask, meaning settlement is pending. `predicted_correct` is
    None on unsettled rows — Step 2 drops these from the denominator
    and Step 3 skips them.
    """

    model_config = ConfigDict(extra="ignore")

    event_id: str
    slug: str
    closed: bool
    settled: bool
    # YES/NO refer to gamma's natural orientation (the YES side as
    # ranked by `team_a_name` in the prediction row's `tennis_stats` /
    # event title). For binary head-to-heads expanded into YES + NO
    # clones, both clones share the slug; resolution applies once.
    winning_side: Literal["yes", "no"] | None = None
    winning_team_name: str | None = None
    predicted_correct: bool | None = None
    # Optional human-readable reason when `predicted_correct` is None
    # (e.g. "trading closed but no settled price observed", "side label
    # mismatch — predicted_winner not found in event sides").
    skip_reason: str | None = None
    resolved_at_utc: datetime


class EventFeatures(BaseModel):
    """One per-event feature row consumed by Steps 2 and 3.

    Step 2 aggregates these into hit-rate cuts; Step 3 ships the full
    list (wins + losses) to the LLM in one batched call. Per-player
    fields use `_a_` / `_b_` suffixes following the pipeline's
    `team_a_name` / `team_b_name` convention (favorite first).
    """

    model_config = ConfigDict(extra="ignore")

    # Identity / categorical cuts
    event_id: str
    event_title: str | None = None
    run_id: str
    sport_type: str | None = None
    lens_set_name: str | None = None
    surface: str | None = None
    # Director outputs
    predicted_winner: str
    predicted_prob: float
    market_implied_prob: float | None = None
    confidence: Literal["low", "medium", "high"]
    # Slate-judge outputs
    defensibility_score: float | None = None
    case_bucket: int | None = Field(
        default=None,
        description=(
            "1-5 fire-emoji bucket from `_defensibility_stars` boundaries. "
            "None when defensibility_score wasn't logged (judge failure)."
        ),
    )
    # Step 2's anti-anchoring cut. True when predicted_winner sits on
    # the side gamma's market priced as the favorite (≥0.5 implied).
    market_favorite_pick: bool | None = None
    # Outcome (joined from ResolvedOutcome). `won` is only meaningful
    # when `settled` is True.
    settled: bool
    won: bool | None = None
    # Tennis post-match per-side divergence, populated only when
    # post-match stats successfully fetched. None on non-tennis or
    # vendor miss; callers handle absence as "no divergence signal."
    baseline_first_serve_in_pct_a: float | None = None
    actual_first_serve_in_pct_a: float | None = None
    divergence_first_serve_in_a: float | None = None
    baseline_first_serve_win_pct_a: float | None = None
    actual_first_serve_win_pct_a: float | None = None
    divergence_first_serve_win_a: float | None = None
    baseline_second_serve_win_pct_a: float | None = None
    actual_second_serve_win_pct_a: float | None = None
    divergence_second_serve_win_a: float | None = None
    baseline_bp_convert_pct_a: float | None = None
    actual_bp_convert_pct_a: float | None = None
    divergence_bp_convert_a: float | None = None
    baseline_first_serve_in_pct_b: float | None = None
    actual_first_serve_in_pct_b: float | None = None
    divergence_first_serve_in_b: float | None = None
    baseline_first_serve_win_pct_b: float | None = None
    actual_first_serve_win_pct_b: float | None = None
    divergence_first_serve_win_b: float | None = None
    baseline_second_serve_win_pct_b: float | None = None
    actual_second_serve_win_pct_b: float | None = None
    divergence_second_serve_win_b: float | None = None
    baseline_bp_convert_pct_b: float | None = None
    actual_bp_convert_pct_b: float | None = None
    divergence_bp_convert_b: float | None = None


class RetroPostMatchPair(BaseModel):
    """Per-event pair of `PerMatchStats` (one per player), Step 3 cache shape.

    Cached at `logs/retro/post_match/<tour>/<player_a_id>_<date>.json`
    so reruns don't re-hit MatchStat. Stored as a pair (vs. two files)
    because we always fetch them together and the pair is what
    `extract_features` consumes — plus the cache hit ratio is highest
    when keyed by event-day rather than per-player-day.
    """

    model_config = ConfigDict(extra="ignore")

    event_id: str
    on_date: _date_t
    player_a_name: str
    player_b_name: str
    player_a: PerMatchStats | None = None
    player_b: PerMatchStats | None = None


class RetroFindings(BaseModel):
    """Output of Step 3's batched LLM pattern-finding call (per sport).

    Three buckets so the operator can scan: cross-cutting patterns,
    lens-attribution, and concrete prompt edits to consider. The LLM is
    asked to NOT invent per-event explanations and to NOT prescribe
    auto-editable changes — recommendations are for human review.
    """

    model_config = ConfigDict(extra="ignore")

    sport: str
    n_events: int = Field(ge=0)
    n_wins: int = Field(ge=0)
    n_losses: int = Field(ge=0)
    recurring_patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Patterns overrepresented in losses vs wins, named in plain "
            "language. Each entry is one short bullet."
        ),
    )
    lens_underperformance: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map of lens_name → recurring failure mode. Lens names must "
            "match the sport's registered LensSet (e.g. for tennis: "
            "tennis_form_and_surface, tennis_matchup_and_clutch, "
            "tennis_conditions_and_context)."
        ),
    )
    prompt_recommendations: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete, operator-reviewable suggestions for prompt edits "
            "(DIRECTOR_SPORT_HINTS, lens reasoner_sport_hint, lens-set "
            "director_system_tail). Do NOT recommend auto-editable changes."
        ),
    )
