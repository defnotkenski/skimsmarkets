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
    # True when the director picked the same side as the market but with
    # strictly lower probability than the market priced — agreeing-with-
    # less-conviction. None on older rows (pre-flag) or when market
    # implied is missing; `extract_features` recomputes from the two
    # probability fields as a fallback so older runs benefit too.
    negative_edge: bool | None = None
    confidence: Literal["low", "medium", "high"]
    headline: str | None = None
    reasoning: str | None = None
    specialist_weights: dict[str, float] = Field(default_factory=dict)
    disagreements_flagged: list[str] = Field(default_factory=list)
    uw_flow_note: str | None = None
    defensibility_score: float | None = None
    defensibility_rationale: str | None = None
    defensibility_flags: list[str] = Field(default_factory=list)
    # Deterministic risk classifier output (see `classify.py`). None on
    # older runs predating the classifier; `risk_score` is also None when
    # the judge produced no defensibility score (bucket `Unrated`).
    risk_bucket: str | None = None
    risk_score: float | None = None
    # Calibration audit (see `calibration.py`). `calibration_temperature`
    # is the scalar applied to the classifier's magnitude term this run;
    # `calibrated_winner_probability` is what that term actually saw. Both
    # None on runs predating the calibration layer; temperature is 1.0
    # (identity) on runs made with no committed artefact.
    calibration_temperature: float | None = None
    calibrated_winner_probability: float | None = None
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

    # Promoted derived metrics — see `pipeline._persist_run` for how
    # they're computed. None on older runs (pre-2026-05-09) and on
    # non-tennis events that lack a stacking math.
    prompt_version: str | None = None
    predicted_probability_bucket: str | None = None
    stack_baseline_team_a: float | None = None
    stack_team_a_probability: float | None = None
    team_a_p_final: float | None = None
    stack_vs_final_delta: float | None = None
    # Deterministic director-discipline flags set by
    # `pipeline._persist_run`. Each mirrors a specific rule in
    # `DIRECTOR_SYSTEM_TENNIS_TAIL` and fires when the prediction
    # violates it. Tennis-only; None on non-tennis or partial-failure
    # events; older runs (pre-2026-05-16 fix) lack these fields and
    # parse as None.
    #
    # `override_without_retract`: |stack_vs_final_delta| > 0.01 AND
    #   `retracted_shifts` is empty (silent deviation from stack math).
    # `confidence_should_be_low_injury`: any non-empty injury_concerns
    #   AND confidence != "low" (injury-flag cap rule).
    # `confidence_should_be_low_stacked`: |shift_total| ≥ 0.10 AND ≥2
    #   shifts in override direction each ≥ 0.04 AND confidence !=
    #   "low" (multi-shift stack cap rule).
    # `gbt_sim_split_unjustified`: GBT < 0.50 on pick AND sim ≥ 0.50
    #   on pick AND no GBT top_features name appears in reasoning
    #   (sim-vs-GBT split discipline rule).
    override_without_retract: bool | None = None
    confidence_should_be_low_injury: bool | None = None
    confidence_should_be_low_stacked: bool | None = None
    gbt_sim_split_unjustified: bool | None = None
    gap_to_market_signed: float | None = None
    gap_to_sim_signed: float | None = None
    gap_to_gbt_signed: float | None = None
    lens_coverage: dict[str, str] = Field(default_factory=dict)
    retracted_shifts: list[dict[str, Any]] = Field(default_factory=list)
    token_usage_summary: dict[str, Any] | None = None
    token_usage_calls: list[dict[str, Any]] = Field(default_factory=list)


class TradeRow(BaseModel):
    """One `record_type="trade"` JSONL row in `logs/trades/<run_id>.jsonl`.

    Written by `skims execute`, one per filtered prediction row —
    whether the trade was placed, skipped, or dry-run. The same run's
    log is also read by `executed_event_ids()` for intra-run idempotency
    (don't re-place an order whose event already has an executed audit
    row).

    Identity fields (`event_id`, `market_slug`) link back to the
    source `PredictionRow` for retro joins. Diagnostic fields
    (`kalshi_yes_ask_dollars_at_decision`, `raw_response_excerpt`)
    capture what we saw at decision time so post-hoc analysis can
    spot slippage / venue drift without re-fetching.
    """

    model_config = ConfigDict(extra="ignore")

    record_type: Literal["trade"]
    run_id: str
    audit_timestamp: datetime
    # Source prediction identity
    event_id: str
    market_slug: str
    sport_type: str | None = None
    event_title: str | None = None
    predicted_winner: str
    predicted_yes_probability: float
    confidence: Literal["low", "medium", "high"]
    defensibility_score: float | None = None
    negative_edge: bool | None = None
    # Kalshi venue identity (None when skipped before matching)
    kalshi_event_ticker: str | None = None
    market_ticker: str | None = None
    side: Literal["yes", "no"] = "yes"
    # Cost / size
    bet_size_cents: int
    kalshi_yes_ask_dollars_at_decision: float | None = None
    # Execution outcome
    dry_run: bool
    order_id: str | None = None
    client_order_id: str | None = None
    fill_contracts: int = 0
    # Contract cost only — what we paid the seller for the contracts.
    fill_total_cost_cents: int = 0
    fill_avg_price_cents: int | None = None
    # Kalshi fees on the fill (taker + maker). Charged on top of
    # `fill_total_cost_cents` — the account is debited the sum.
    # The exposure-cap accumulator uses `fill_total_cost_cents` only
    # (matches the budget knob `bet_size_cents` semantics); fees are
    # recorded here for transparency and retro analysis.
    fill_fees_cents: int = 0
    fill_status: Literal[
        "filled", "partial", "skipped_dry_run", "skipped", "submitted",
    ]
    skip_reason: str | None = None
    # Forensic capture of Kalshi's raw response. Helps diagnose field-
    # name drift across SDK versions without re-running the order.
    raw_response_excerpt: dict[str, Any] | None = None


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
            "1-5 bar bucket from `_defensibility_stars` boundaries. "
            "None when defensibility_score wasn't logged (judge failure)."
        ),
    )
    # Deterministic risk classifier output (see `classify.py`) — distinct
    # from `case_bucket`: this is the composite of magnitude + defensibility
    # + market convergence, not the defensibility-only bar. None on runs
    # predating the classifier; `risk_score` is also None when the bucket
    # is `Unrated` (no judge score).
    risk_bucket: str | None = None
    risk_score: float | None = None
    # Step 2's anti-anchoring cut. True when predicted_winner sits on
    # the side gamma's market priced as the favorite (≥0.5 implied).
    market_favorite_pick: bool | None = None
    # True when the director picked the same side as the market but
    # with strictly lower conviction. Mirrors `PredictionRow.negative_edge`
    # — kept here as a feature so calibrate can cut hit rate by it.
    negative_edge: bool | None = None
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

    # Tennis per-shift directional grading. For each of the six signed
    # shifts the tennis lens set emits (form, surface, h2h, clutch,
    # physical, stakes), True when the shift's sign pointed at the
    # eventual winner; False when it pointed away; None when the shift
    # was ~zero (no directional content) or the event isn't settled or
    # team_a couldn't be identified. Aggregated by analyze.py to
    # surface "which lens shift has the worst hit rate" — direct signal
    # for tightening that reasoner. Non-tennis events leave all six
    # null. The tolerance for "zero" is ±0.005 — anything inside that
    # band is treated as no-call.
    form_signed_shift_correct: bool | None = None
    surface_signed_shift_correct: bool | None = None
    h2h_signed_shift_correct: bool | None = None
    clutch_signed_shift_correct: bool | None = None
    physical_signed_shift_correct: bool | None = None
    stakes_signed_shift_correct: bool | None = None
    # Raw shift values copied for analysis convenience — saves the
    # consumer from walking back into `specialist_reports`.
    form_signed_shift_value: float | None = None
    surface_signed_shift_value: float | None = None
    h2h_signed_shift_value: float | None = None
    clutch_signed_shift_value: float | None = None
    physical_signed_shift_value: float | None = None
    stakes_signed_shift_value: float | None = None
    # Stack metrics promoted to features so analysis can ask "did the
    # director's overrides pay off?" by joining `stack_vs_final_delta`
    # against `won`.
    stack_team_a_probability: float | None = None
    stack_vs_final_delta: float | None = None
    gap_to_market_signed: float | None = None
    gap_to_sim_signed: float | None = None
    gap_to_gbt_signed: float | None = None


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


class LensUnderperformance(BaseModel):
    """One lens's recurring failure mode, attributed by the retro analyzer.

    Pairs the lens name with a one-line description of how it's failing.
    Carried as a list-of-records on `RetroFindings` rather than a dict
    (`{lens_name: failure_mode}`) for the same reason `SpecialistWeight`
    is a list: dict-shaped outputs are less reliable than list-shaped
    outputs in Anthropic's structured-output mode (the compiler enforces
    array constraints during generation but not object cardinality).
    """

    model_config = ConfigDict(extra="ignore")

    lens_name: str = Field(
        description=(
            "Exact lens name from the sport's registered LensSet — e.g. "
            "for tennis: 'tennis_form_and_surface', "
            "'tennis_matchup_and_clutch', 'tennis_conditions_and_context'. "
            "Do not invent lens names."
        ),
    )
    failure_mode: str = Field(
        description=(
            "One short sentence describing the recurring failure mode "
            "for this lens (e.g. 'overweights H2H sample of <3 meetings', "
            "'misses surface-specific form on clay-court swing')."
        ),
    )


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
    lens_underperformance: list[LensUnderperformance] = Field(
        default_factory=list,
        description=(
            "One entry per lens with a recurring failure mode. Empty list "
            "when no clear lens attribution emerges from the data. Use the "
            "exact lens names from the sport's registered LensSet."
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
