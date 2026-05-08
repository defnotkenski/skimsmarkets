"""Per-event feature extractor — shared between Steps 2 (calibrate) and 3
(LLM analyze).

Step 2 buckets these rows for hit-rate cuts; Step 3 ships the full
list to the LLM in a batched pattern-finding call. Keeping the
extraction in one place ensures the two steps see identical features
and one cut definition (e.g. "market favorite" semantics) can't drift
between them.

The extractor is pure / sync — fetching of post-match stats happens
upstream in `retro/post_match.py` and the result is passed in as
`post_match`. None when post-match wasn't fetched (non-tennis sports,
unsettled events, vendor miss); the resulting `EventFeatures` then
carries None for all per-side divergence fields.
"""

from __future__ import annotations

from skimsmarkets.reporting import _defensibility_stars
from skimsmarkets.retro.models import (
    EventFeatures,
    PredictionRow,
    ResolvedOutcome,
    RetroPostMatchPair,
)


def _case_bucket(score: float | None) -> int | None:
    """Map a [0,1] defensibility score to its 1-5 bar-bucket.

    Source-of-truth boundaries live in
    `reporting._defensibility_stars` (0.85 / 0.65 / 0.45 / 0.25). We
    derive the bucket here by counting filled glyphs in the rendered
    string rather than re-asserting the boundaries — single source of
    truth, will track future changes.
    """
    if score is None:
        return None
    rendered = _defensibility_stars(score)
    return rendered.count("█") or None


def _market_favorite_pick(
    predicted_prob: float | None,
    market_implied: float | None,
) -> bool | None:
    """True when the predicted side IS the market favorite.

    The prediction row's `polymarket_implied_probability` is the implied
    probability of the predicted side specifically (the director picked
    a winner and the market price for that winner is what gets logged).
    So:
      - market_implied >= 0.5 → predicted side WAS the favorite → True
      - market_implied <  0.5 → predicted side was the underdog → False
      - market_implied is None → no signal → None
    """
    if market_implied is None:
        return None
    return market_implied >= 0.5


def extract_features(
    row: PredictionRow,
    outcome: ResolvedOutcome | None,
    post_match: RetroPostMatchPair | None,
) -> EventFeatures:
    """Build the per-event feature row used by Steps 2 and 3.

    Outcome may be None when no resolution sidecar exists yet for this
    run — the feature row is still produced (with `settled=False` and
    `won=None`) so the caller can decide whether to drop it. Post-match
    is None on non-tennis events and on tennis events where the vendor
    fetch missed.
    """
    settled = bool(outcome and outcome.settled)
    won: bool | None = outcome.predicted_correct if outcome and settled else None

    # Pull surface from the tennis_stats payload when present (it's the
    # most reliable surface signal; the event title rarely carries it
    # for tennis, and the lens reports' surface fields aren't typed).
    surface: str | None = None
    if row.tennis_stats is not None:
        surface = row.tennis_stats.surface

    feats = EventFeatures(
        event_id=row.event_id,
        event_title=row.event_title,
        run_id=row.run_id,
        sport_type=row.sport_type,
        lens_set_name=row.lens_set_name,
        surface=surface,
        predicted_winner=row.predicted_winner,
        predicted_prob=row.predicted_yes_probability,
        market_implied_prob=row.polymarket_implied_probability,
        confidence=row.confidence,
        defensibility_score=row.defensibility_score,
        case_bucket=_case_bucket(row.defensibility_score),
        market_favorite_pick=_market_favorite_pick(
            row.predicted_yes_probability,
            row.polymarket_implied_probability,
        ),
        settled=settled,
        won=won,
    )

    # Tennis-only post-match divergence. Pre-match baseline lives on
    # `tennis_stats.player_a/b`; actuals come from `post_match.player_a/b`.
    # Compute divergence as actual - baseline so positive numbers always
    # mean "outperformed baseline" regardless of which percentage.
    if row.tennis_stats is None or post_match is None:
        return feats

    baseline_a = row.tennis_stats.player_a
    baseline_b = row.tennis_stats.player_b
    actual_a = post_match.player_a
    actual_b = post_match.player_b

    def _div(actual: float | None, baseline: float | None) -> float | None:
        if actual is None or baseline is None:
            return None
        return actual - baseline

    if actual_a is not None:
        feats.baseline_first_serve_in_pct_a = baseline_a.first_serve_in_pct
        feats.actual_first_serve_in_pct_a = actual_a.first_serve_in_pct
        feats.divergence_first_serve_in_a = _div(
            actual_a.first_serve_in_pct, baseline_a.first_serve_in_pct
        )
        feats.baseline_first_serve_win_pct_a = baseline_a.first_serve_win_pct
        feats.actual_first_serve_win_pct_a = actual_a.first_serve_win_pct
        feats.divergence_first_serve_win_a = _div(
            actual_a.first_serve_win_pct, baseline_a.first_serve_win_pct
        )
        feats.baseline_second_serve_win_pct_a = baseline_a.second_serve_win_pct
        feats.actual_second_serve_win_pct_a = actual_a.second_serve_win_pct
        feats.divergence_second_serve_win_a = _div(
            actual_a.second_serve_win_pct, baseline_a.second_serve_win_pct
        )
        feats.baseline_bp_convert_pct_a = baseline_a.break_point_convert_pct
        feats.actual_bp_convert_pct_a = actual_a.break_point_convert_pct
        feats.divergence_bp_convert_a = _div(
            actual_a.break_point_convert_pct, baseline_a.break_point_convert_pct
        )

    if actual_b is not None:
        feats.baseline_first_serve_in_pct_b = baseline_b.first_serve_in_pct
        feats.actual_first_serve_in_pct_b = actual_b.first_serve_in_pct
        feats.divergence_first_serve_in_b = _div(
            actual_b.first_serve_in_pct, baseline_b.first_serve_in_pct
        )
        feats.baseline_first_serve_win_pct_b = baseline_b.first_serve_win_pct
        feats.actual_first_serve_win_pct_b = actual_b.first_serve_win_pct
        feats.divergence_first_serve_win_b = _div(
            actual_b.first_serve_win_pct, baseline_b.first_serve_win_pct
        )
        feats.baseline_second_serve_win_pct_b = baseline_b.second_serve_win_pct
        feats.actual_second_serve_win_pct_b = actual_b.second_serve_win_pct
        feats.divergence_second_serve_win_b = _div(
            actual_b.second_serve_win_pct, baseline_b.second_serve_win_pct
        )
        feats.baseline_bp_convert_pct_b = baseline_b.break_point_convert_pct
        feats.actual_bp_convert_pct_b = actual_b.break_point_convert_pct
        feats.divergence_bp_convert_b = _div(
            actual_b.break_point_convert_pct, baseline_b.break_point_convert_pct
        )

    return feats
