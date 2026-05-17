"""Deterministic risk classifier — grades the slate into risk buckets.

Runs in deterministic post-processing, after the (market-blind) LLM layer
and the judge. Combines three signals into a continuous `risk_score`, then
cuts it into four full-spectrum buckets (plus an `Unrated` sentinel):

  - magnitude     — `predicted_winner_probability`, how lopsided the call is
  - defensibility — the judge's `defensibility_score`, internal soundness
  - convergence   — agreement between the blind estimate and the market price

Convergence is the whole point of blinding the LLM stages: because no LLM
saw the price, `gap_to_market_signed` is a genuinely independent cross-check
rather than "how far the director dared to stray from a prior it was shown".
This module is the only place that signal reaches the live output — it
never enters an LLM prompt.

Every constant below (gap scales, weights, band thresholds) is a first-guess
value, meant to be tuned against `retro` backtest data once enough
blind-mode runs have accumulated. Treat them as a starting point, not a
calibration.
"""

from __future__ import annotations

from skimsmarkets.calibration import apply_temperature

# --- tunables (first-guess; tune against retro backtest data) --------------

# Convergence is asymmetric in the predicted-winner frame. A blind estimate
# that is *more* bullish on its winner than the market (positive gap) decays
# gently — being bolder than the crowd is the intended payoff of blinding. A
# blind estimate that is *less* bullish than the market on its own pick
# (negative gap) decays steeply: being more cautious than even the crowd on
# the side you chose is a weak pick.
POS_GAP_SCALE = 0.40
NEG_GAP_SCALE = 0.20

# Directional-disagreement penalty. When the market's own probability for the
# predicted winner falls below 0.5, the market and the blind estimate disagree
# on *who wins* — categorically worse than a same-side confidence gap, so it
# takes a steep penalty that stacks on top of the gap decay above. This is the
# distance below 0.5 at which that penalty alone fully crushes the convergence
# term (e.g. 0.20 → the market pricing the pick at 0.30 zeroes it on its own).
DISAGREE_SCALE = 0.20

# GBT convergence — same asymmetric structure as the market term but
# cross-checking against the GBT prior instead of the market. The GBT
# (tennis_gbt_spike) is the best calibrated point predictor in the system
# (holdout Brier 0.21112 as of 2026-05-16) and is the director's BASELINE
# ANCHOR per `DIRECTOR_SYSTEM_TENNIS_TAIL`. Without this term the classifier
# is blind to whether the director matched or overrode the GBT — a Lock-band
# pick that disagrees with GBT by 20pp gets the same risk_score treatment as
# one that matches GBT exactly. Backtest on the 20 pre-fix tennis rows + 16
# resolved outcomes showed all demotions under this term were wrong-pick
# demotions (directional evidence; sample too small to be statistically
# conclusive, but the pattern is consistent with the architectural argument
# that material director-vs-GBT divergence is a yellow flag).
POS_GAP_GBT_SCALE = 0.40
NEG_GAP_GBT_SCALE = 0.20
DISAGREE_GBT_SCALE = 0.20

# risk_score = weighted sum of the four [0, 1] terms. Magnitude and
# defensibility are load-bearing (how big / how sound); the two convergence
# terms cross-check (market for blinded-vs-crowd, GBT for director-vs-anchor).
# Weights rebalanced 2026-05-16: defensibility 0.40 → 0.25 (still the
# dominant LLM-side signal), market convergence 0.20 → 0.15, new GBT
# convergence at 0.20. Sum = 1.0. Magnitude kept at 0.40 — the calibrated
# magnitude term is the only one that touches `apply_temperature`, so
# changing its weight would conflate with calibration tuning.
W_MAGNITUDE = 0.40
W_DEFENSIBILITY = 0.25
W_CONVERGENCE = 0.15
W_GBT_CONVERGENCE = 0.20

# Band cuts on the continuous risk_score.
THRESHOLD_LOCK = 0.75
THRESHOLD_LEAN = 0.60
THRESHOLD_COINFLIP = 0.45

# --- bucket labels ---------------------------------------------------------

BUCKET_LOCK = "Lock"
BUCKET_LEAN = "Lean"
BUCKET_COINFLIP = "Coin-flip"
BUCKET_AVOID = "Avoid"
BUCKET_UNRATED = "Unrated"

# Best → worst. `Unrated` is the judge-failure sentinel and always sorts last.
BUCKET_ORDER: tuple[str, ...] = (
    BUCKET_LOCK,
    BUCKET_LEAN,
    BUCKET_COINFLIP,
    BUCKET_AVOID,
    BUCKET_UNRATED,
)


def bucket_rank(bucket: str) -> int:
    """Ordinal for leaderboard sorting — 0 = best (Lock), higher = worse.

    Unknown labels sort after every known bucket.
    """
    try:
        return BUCKET_ORDER.index(bucket)
    except ValueError:
        return len(BUCKET_ORDER)


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _gap_decay(
    gap_winner: float,
    pos_scale: float,
    neg_scale: float,
    disagree_scale: float,
    raw_winner_prob: float,
) -> float:
    """Standard 3-component convergence decay against a prior.

    Shared between the market and the GBT convergence terms; both use the
    same shape (positive gap = decay gently; negative gap = decay steeply;
    directional-disagreement = extra penalty stacked on top). The scales
    differ per prior (see `*_SCALE` and `*_GBT_SCALE` constants).

    `raw_winner_prob` is the director's RAW (pre-temperature) probability
    for the predicted winner — used to derive the prior's implied
    probability for the same side via `raw_winner_prob − gap_winner`. The
    raw value is correct here because the gap was computed against the raw
    director output, not the calibrated one.
    """
    prior_winner_prob = raw_winner_prob - gap_winner
    return _clip01(
        1.0
        - max(0.0, -gap_winner) / neg_scale
        - max(0.0, gap_winner) / pos_scale
        - max(0.0, 0.5 - prior_winner_prob) / disagree_scale
    )


def classify_risk(
    predicted_winner_probability: float,
    defensibility_score: float | None,
    gap_to_market_signed: float | None,
    *,
    predicted_winner_is_team_a: bool,
    temperature: float = 1.0,
    gap_to_gbt_signed: float | None = None,
) -> tuple[str, float | None]:
    """Grade one event into a `(risk_bucket, risk_score)` pair.

    `predicted_winner_probability` is winner-anchored ([0.5, 1.0] in practice
    — the director names the side it thinks wins). `gap_to_market_signed`
    and `gap_to_gbt_signed` are team_a-anchored (positive = the director
    puts team_a above that prior's team_a probability);
    `predicted_winner_is_team_a` re-anchors them to the winner frame.

    `temperature` is the calibration scalar from
    `models/tennis_calibration.json` (the pipeline loads it via
    `calibration.load_temperature`). It is applied to the magnitude term
    ONLY — `apply_temperature` has 0.5 as its fixed point, so it rescales
    confidence without ever flipping the pick. Default 1.0 is the identity
    transform (exact pre-calibration behaviour); the temperature is fit on
    win/loss outcomes, never on price, so the classifier stays market-blind.

    `gap_to_gbt_signed` is the team_a-anchored gap between the director's
    `team_a_p_final` and the GBT prior `p_team_a_wins`. Defaults to None
    for backward-compat (tests and any older callers that don't pass it
    get the prior 3-term behaviour, just with the rebalanced weights —
    when GBT is missing, its weight share renormalizes onto the remaining
    terms). The GBT convergence term IS the answer to "does the director's
    output align with the system's best calibrated prior?" — added
    2026-05-16 so the classifier penalizes large director-vs-GBT
    divergence the same way it already penalized director-vs-market.

    Returns `("Unrated", None)` when the judge produced no
    `defensibility_score` — without it there is no honest composite. When
    EITHER `gap_to_market_signed` OR `gap_to_gbt_signed` is None the
    corresponding term is dropped and the remaining weights renormalize
    rather than penalizing the event for a data gap. Never raises.
    """
    if defensibility_score is None:
        return BUCKET_UNRATED, None

    # Calibrated: `apply_temperature` corrects the director's raw confidence
    # against historically-resolved outcomes. Magnitude is the only term that
    # gets calibrated — the only place the probability acts as a confidence
    # *level* rather than a side indicator. T=1.0 → identity.
    magnitude_term = _clip01(
        apply_temperature(predicted_winner_probability, temperature)
    )
    defensibility_term = _clip01(defensibility_score)

    terms: list[tuple[float, float]] = [
        (W_MAGNITUDE, magnitude_term),
        (W_DEFENSIBILITY, defensibility_term),
    ]
    if gap_to_market_signed is not None:
        # Re-anchor the team_a-framed gap to the predicted-winner frame:
        # positive = director is MORE bullish on the predicted winner than
        # the market; negative = less bullish on that same side.
        gap_winner = (
            gap_to_market_signed
            if predicted_winner_is_team_a
            else -gap_to_market_signed
        )
        convergence_term = _gap_decay(
            gap_winner, POS_GAP_SCALE, NEG_GAP_SCALE, DISAGREE_SCALE,
            predicted_winner_probability,
        )
        terms.append((W_CONVERGENCE, convergence_term))
    if gap_to_gbt_signed is not None:
        gap_winner_gbt = (
            gap_to_gbt_signed
            if predicted_winner_is_team_a
            else -gap_to_gbt_signed
        )
        gbt_convergence_term = _gap_decay(
            gap_winner_gbt,
            POS_GAP_GBT_SCALE, NEG_GAP_GBT_SCALE, DISAGREE_GBT_SCALE,
            predicted_winner_probability,
        )
        terms.append((W_GBT_CONVERGENCE, gbt_convergence_term))

    # Renormalize the active weights so a missing convergence term doesn't
    # silently lower every score by its dropped weight share. With both
    # convergence terms active: divisor = 1.0 (the configured weights sum
    # to 1.0). With one dropped: divisor < 1.0 and the kept terms scale up
    # proportionally. With both dropped (no market, cold-start GBT):
    # `risk_score = (W_M * mag + W_D * def) / (W_M + W_D)`.
    weight_sum = sum(w for w, _ in terms)
    risk_score = sum(w * v for w, v in terms) / weight_sum

    if risk_score >= THRESHOLD_LOCK:
        bucket = BUCKET_LOCK
    elif risk_score >= THRESHOLD_LEAN:
        bucket = BUCKET_LEAN
    elif risk_score >= THRESHOLD_COINFLIP:
        bucket = BUCKET_COINFLIP
    else:
        bucket = BUCKET_AVOID
    return bucket, risk_score
