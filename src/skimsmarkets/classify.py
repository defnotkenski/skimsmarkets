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

# risk_score = weighted sum of the three [0, 1] terms. Magnitude and
# defensibility are load-bearing (how big / how sound); convergence confirms.
W_MAGNITUDE = 0.40
W_DEFENSIBILITY = 0.40
W_CONVERGENCE = 0.20

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


def classify_risk(
    predicted_winner_probability: float,
    defensibility_score: float | None,
    gap_to_market_signed: float | None,
    *,
    predicted_winner_is_team_a: bool,
    temperature: float = 1.0,
) -> tuple[str, float | None]:
    """Grade one event into a `(risk_bucket, risk_score)` pair.

    `predicted_winner_probability` is winner-anchored ([0.5, 1.0] in practice
    — the director names the side it thinks wins). `gap_to_market_signed` is
    team_a-anchored (positive = the blind estimate puts team_a above the
    market's team_a price); `predicted_winner_is_team_a` re-anchors it to the
    winner frame.

    `temperature` is the calibration scalar from
    `models/tennis_calibration.json` (the pipeline loads it via
    `calibration.load_temperature`). It is applied to the magnitude term
    ONLY — `apply_temperature` has 0.5 as its fixed point, so it rescales
    confidence without ever flipping the pick. Default 1.0 is the identity
    transform (exact pre-calibration behaviour); the temperature is fit on
    win/loss outcomes, never on price, so the classifier stays market-blind.

    Returns `("Unrated", None)` when the judge produced no
    `defensibility_score` — without it there is no honest composite. When
    `gap_to_market_signed` is None (no market implied probability, e.g. a
    one-sided book) the convergence term is dropped and the remaining two
    weights are renormalized rather than penalizing the event for a data
    gap. Never raises.
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

    if gap_to_market_signed is None:
        risk_score = 0.5 * magnitude_term + 0.5 * defensibility_term
    else:
        # Re-anchor the team_a-framed gap to the predicted-winner frame:
        # positive = blind estimate is MORE bullish on the predicted winner
        # than the market; negative = less bullish on that same side.
        gap_winner = (
            gap_to_market_signed
            if predicted_winner_is_team_a
            else -gap_to_market_signed
        )
        # The market's own probability for the predicted winner. Below 0.5
        # the market and the blind estimate disagree on *who wins* — a
        # directional disagreement, not just a confidence gap — so it takes
        # a steep extra penalty stacked on top of the gap decay. Uses the
        # RAW probability, not the temperature-scaled one: this subtraction
        # recovers what the market thinks (director_raw − gap = market), a
        # side/sign question temperature's 0.5 fixed point cannot change.
        market_winner_prob = predicted_winner_probability - gap_winner
        convergence_term = _clip01(
            1.0
            - max(0.0, -gap_winner) / NEG_GAP_SCALE
            - max(0.0, gap_winner) / POS_GAP_SCALE
            - max(0.0, 0.5 - market_winner_prob) / DISAGREE_SCALE
        )
        risk_score = (
            W_MAGNITUDE * magnitude_term
            + W_DEFENSIBILITY * defensibility_term
            + W_CONVERGENCE * convergence_term
        )

    if risk_score >= THRESHOLD_LOCK:
        bucket = BUCKET_LOCK
    elif risk_score >= THRESHOLD_LEAN:
        bucket = BUCKET_LEAN
    elif risk_score >= THRESHOLD_COINFLIP:
        bucket = BUCKET_COINFLIP
    else:
        bucket = BUCKET_AVOID
    return bucket, risk_score
