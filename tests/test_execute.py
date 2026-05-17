"""Plain-assert smoke tests for `skims execute` — runnable as a script.

Usage:

    uv run python tests/test_execute.py

No pytest dependency (the project doesn't carry one). Each `_t_*`
function returns the count of assertions it made; `main()` runs them
and prints a one-line tally per group.
"""

from __future__ import annotations

import math
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from skimsmarkets.calibration import apply_temperature  # noqa: E402
from skimsmarkets.classify import (  # noqa: E402
    BUCKET_AVOID,
    BUCKET_COINFLIP,
    BUCKET_LOCK,
    BUCKET_ORDER,
    BUCKET_UNRATED,
    THRESHOLD_COINFLIP,
    THRESHOLD_LEAN,
    THRESHOLD_LOCK,
    bucket_rank,
    classify_risk,
)
from skimsmarkets.execute.filters import filter_rows  # noqa: E402
from skimsmarkets.execute.trader import (  # noqa: E402
    _implied_at_or_above_max,
    sum_exposure_cents,
)
from skimsmarkets.kalshi.matcher import (  # noqa: E402
    extract_match_players,
    find_kalshi_match,
    last_token,
)
from skimsmarkets.kalshi.models import KalshiEvent, MarketPosition  # noqa: E402
from skimsmarkets.retro.metrics import (  # noqa: E402
    MIN_FIT_N,
    brier_score,
    calibration_curve,
    compute_metrics,
    expected_calibration_error,
    fit_temperature,
    log_loss,
)
from skimsmarkets.retro.models import PredictionRow  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _row(
    *,
    predicted_winner: str,
    event_title: str | None = None,
    confidence: str = "high",
    defensibility_score: float | None = 0.8,
    negative_edge: bool | None = False,
    sport_type: str | None = "tennis",
    risk_bucket: str | None = "Lock",
    polymarket_implied_probability: float | None = 0.60,
) -> PredictionRow:
    """Minimal PredictionRow for matcher / filter tests."""
    return PredictionRow.model_validate({
        "record_type": "prediction",
        "run_id": "test-run",
        "logged_at_utc": datetime.now(UTC).isoformat(),
        "event_id": "evt-1",
        "market_slug": "slug-1",
        "predicted_winner": predicted_winner,
        "predicted_yes_probability": 0.65,
        "confidence": confidence,
        "event_title": event_title,
        "defensibility_score": defensibility_score,
        "negative_edge": negative_edge,
        "sport_type": sport_type,
        "risk_bucket": risk_bucket,
        "polymarket_implied_probability": polymarket_implied_probability,
    })


def _kalshi_event(
    *,
    event_ticker: str,
    title: str,
    sub_title: str = "",
    yes_player_full: str,
    no_player_full: str,
    yes_ask: float | None = 0.55,
    status: str = "active",
) -> KalshiEvent:
    """Construct a KalshiEvent with two mutually-exclusive markets."""
    yes_short = yes_player_full.split()[-1].upper()[:3]
    no_short = no_player_full.split()[-1].upper()[:3]
    return KalshiEvent.model_validate({
        "event_ticker": event_ticker,
        "series_ticker": "KXATPMATCH",
        "title": title,
        "sub_title": sub_title,
        "mutually_exclusive": True,
        "markets": [
            {
                "ticker": f"{event_ticker}-{yes_short}",
                "event_ticker": event_ticker,
                "yes_sub_title": yes_player_full,
                "no_sub_title": no_player_full,
                "yes_ask_dollars": yes_ask,
                "yes_bid_dollars": yes_ask - 0.01 if yes_ask else None,
                "status": status,
            },
            {
                "ticker": f"{event_ticker}-{no_short}",
                "event_ticker": event_ticker,
                "yes_sub_title": no_player_full,
                "no_sub_title": yes_player_full,
                "yes_ask_dollars": (
                    round(1.0 - yes_ask, 2) if yes_ask else None
                ),
                "yes_bid_dollars": (
                    round(1.0 - yes_ask, 2) - 0.01 if yes_ask else None
                ),
                "status": status,
            },
        ],
    })


# ---------------------------------------------------------------------------
# last_token
# ---------------------------------------------------------------------------


def _t_last_token() -> int:
    cases = [
        ("Frances Tiafoe", "tiafoe"),
        ("En-Shuo Liang", "liang"),  # hyphen-collapsed to space
        ("Karolína Plíšková", "pliskova"),  # diacritics stripped
        ("Andrea Pellegrino", "pellegrino"),
        ("Liang", "liang"),  # single token
        ("", ""),  # empty
        ("  multiple   spaces  ", "spaces"),
    ]
    for inp, want in cases:
        got = last_token(inp)
        assert got == want, f"last_token({inp!r}) = {got!r}, want {want!r}"
    return len(cases)


# ---------------------------------------------------------------------------
# extract_match_players (via event_title only — no tennis_stats fixture)
# ---------------------------------------------------------------------------


def _t_extract_via_title() -> int:
    n = 0
    # Predicted winner is the YES side player
    r = _row(
        predicted_winner="Frances Tiafoe",
        event_title="ATP Rome: Frances Tiafoe vs. Andrea Pellegrino",
    )
    got = extract_match_players(r)
    assert got == ("Frances Tiafoe", "Andrea Pellegrino"), got
    n += 1
    # Predicted winner is the NO side player — orientation flips
    r = _row(
        predicted_winner="Andrea Pellegrino",
        event_title="ATP Rome: Frances Tiafoe vs. Andrea Pellegrino",
    )
    got = extract_match_players(r)
    assert got == ("Andrea Pellegrino", "Frances Tiafoe"), got
    n += 1
    # Hyphenated winner name — normalisation handles the hyphen
    r = _row(
        predicted_winner="En-Shuo Liang",
        event_title="WTA Madrid: En-Shuo Liang vs. Karolína Plíšková",
    )
    got = extract_match_players(r)
    assert got == ("En-Shuo Liang", "Karolína Plíšková"), got
    n += 1
    # Diacritics in winner — should still match by normalised form
    r = _row(
        predicted_winner="Karolína Plíšková",
        event_title="WTA Madrid: En-Shuo Liang vs. Karolína Plíšková",
    )
    got = extract_match_players(r)
    assert got == ("Karolína Plíšková", "En-Shuo Liang"), got
    n += 1
    # No event title and no tennis_stats — returns None
    r = _row(predicted_winner="Frances Tiafoe", event_title=None)
    got = extract_match_players(r)
    assert got is None, got
    n += 1
    # Title without "vs" — returns None (falls through _parse_h2h_question)
    r = _row(predicted_winner="Player", event_title="Random text no separator")
    got = extract_match_players(r)
    assert got is None, got
    n += 1
    return n


# ---------------------------------------------------------------------------
# find_kalshi_match
# ---------------------------------------------------------------------------


def _t_find_match_basic() -> int:
    events = [
        _kalshi_event(
            event_ticker="KXATPMATCH-26MAY11TIAPEL",
            title="Tiafoe vs Pellegrino",
            sub_title="Tiafoe vs Pellegrino (May 11)",
            yes_player_full="Frances Tiafoe",
            no_player_full="Andrea Pellegrino",
            yes_ask=0.70,
        ),
        _kalshi_event(
            event_ticker="KXATPMATCH-26MAY11SINPOP",
            title="Sinner vs Popyrin",
            sub_title="Sinner vs Popyrin (May 11)",
            yes_player_full="Jannik Sinner",
            no_player_full="Alexei Popyrin",
            yes_ask=0.88,
        ),
    ]
    # Predicted Tiafoe → match the first event, resolve TIA market
    r = _row(
        predicted_winner="Frances Tiafoe",
        event_title="ATP Rome: Frances Tiafoe vs. Andrea Pellegrino",
    )
    outcome = find_kalshi_match(r, events)
    assert outcome.kind == "matched", outcome
    assert outcome.market is not None
    assert outcome.market.ticker == "KXATPMATCH-26MAY11TIAPEL-TIA", outcome
    assert outcome.market.yes_ask_dollars == 0.70
    assert outcome.event_ticker == "KXATPMATCH-26MAY11TIAPEL"

    # Predicted Pellegrino → same event, PEL market
    r = _row(
        predicted_winner="Andrea Pellegrino",
        event_title="ATP Rome: Frances Tiafoe vs. Andrea Pellegrino",
    )
    outcome = find_kalshi_match(r, events)
    assert outcome.kind == "matched", outcome
    assert outcome.market is not None
    assert outcome.market.ticker == "KXATPMATCH-26MAY11TIAPEL-PEL", outcome
    return 6


def _t_find_match_no_kalshi() -> int:
    events = [
        _kalshi_event(
            event_ticker="KXATPMATCH-26MAY11SINPOP",
            title="Sinner vs Popyrin",
            sub_title="Sinner vs Popyrin (May 11)",
            yes_player_full="Jannik Sinner",
            no_player_full="Alexei Popyrin",
        ),
    ]
    r = _row(
        predicted_winner="Frances Tiafoe",
        event_title="ATP Rome: Frances Tiafoe vs. Andrea Pellegrino",
    )
    outcome = find_kalshi_match(r, events)
    assert outcome.kind == "no_kalshi_match", outcome
    return 1


def _t_find_match_ambiguous() -> int:
    # Two events with the same surname pair — shouldn't happen in
    # practice but the matcher should refuse to guess.
    events = [
        _kalshi_event(
            event_ticker="KXATPMATCH-26MAY11TIAPEL",
            title="Tiafoe vs Pellegrino",
            yes_player_full="Frances Tiafoe",
            no_player_full="Andrea Pellegrino",
        ),
        _kalshi_event(
            event_ticker="KXATPMATCH-26MAY12TIAPEL",  # different day, same pair
            title="Tiafoe vs Pellegrino",
            yes_player_full="Frances Tiafoe",
            no_player_full="Andrea Pellegrino",
        ),
    ]
    r = _row(
        predicted_winner="Frances Tiafoe",
        event_title="ATP Rome: Frances Tiafoe vs. Andrea Pellegrino",
    )
    outcome = find_kalshi_match(r, events)
    assert outcome.kind == "ambiguous_match", outcome
    return 1


def _t_find_match_closed_market() -> int:
    events = [
        _kalshi_event(
            event_ticker="KXATPMATCH-26MAY11TIAPEL",
            title="Tiafoe vs Pellegrino",
            yes_player_full="Frances Tiafoe",
            no_player_full="Andrea Pellegrino",
            status="settled",  # no longer active
        ),
    ]
    r = _row(
        predicted_winner="Frances Tiafoe",
        event_title="ATP Rome: Frances Tiafoe vs. Andrea Pellegrino",
    )
    outcome = find_kalshi_match(r, events)
    assert outcome.kind == "market_closed", outcome
    assert outcome.event_ticker == "KXATPMATCH-26MAY11TIAPEL"
    return 2


def _t_find_match_no_ask() -> int:
    events = [
        _kalshi_event(
            event_ticker="KXATPMATCH-26MAY11TIAPEL",
            title="Tiafoe vs Pellegrino",
            yes_player_full="Frances Tiafoe",
            no_player_full="Andrea Pellegrino",
            yes_ask=None,  # no ask available
        ),
    ]
    r = _row(
        predicted_winner="Frances Tiafoe",
        event_title="ATP Rome: Frances Tiafoe vs. Andrea Pellegrino",
    )
    outcome = find_kalshi_match(r, events)
    assert outcome.kind == "market_closed", outcome  # mapped to same status
    return 1


def _t_find_match_unparseable() -> int:
    r = _row(
        predicted_winner="Solo Player",
        event_title=None,  # nothing to parse
    )
    outcome = find_kalshi_match(r, [])
    assert outcome.kind == "unparseable_players", outcome
    return 1


# ---------------------------------------------------------------------------
# filter_rows
# ---------------------------------------------------------------------------


def _t_filter_confidence() -> int:
    rows = [
        _row(predicted_winner="A", confidence="high"),
        _row(predicted_winner="B", confidence="medium"),
        _row(predicted_winner="C", confidence="low"),
    ]
    got = list(filter_rows(rows, confidence=["high"]))
    assert len(got) == 1 and got[0].predicted_winner == "A", got
    got = list(filter_rows(rows, confidence=["high", "medium"]))
    assert len(got) == 2, got
    got = list(filter_rows(rows, confidence=None))
    assert len(got) == 3, got
    return 3


def _t_filter_defensibility() -> int:
    rows = [
        _row(predicted_winner="A", defensibility_score=0.9),
        _row(predicted_winner="B", defensibility_score=0.5),
        _row(predicted_winner="C", defensibility_score=None),  # judge failure
    ]
    got = list(filter_rows(rows, min_defensibility=0.7))
    assert [r.predicted_winner for r in got] == ["A"], got
    # None defensibility ALWAYS fails the gate when cutoff is set.
    got = list(filter_rows(rows, min_defensibility=0.0))
    assert [r.predicted_winner for r in got] == ["A", "B"], got
    # No cutoff → all rows pass.
    got = list(filter_rows(rows, min_defensibility=None))
    assert len(got) == 3, got
    return 3


def _t_filter_negative_edge() -> int:
    rows = [
        _row(predicted_winner="A", negative_edge=False),
        _row(predicted_winner="B", negative_edge=True),
        _row(predicted_winner="C", negative_edge=None),
    ]
    got = list(filter_rows(rows, no_negative_edge=True))
    assert [r.predicted_winner for r in got] == ["A"], got
    got = list(filter_rows(rows, no_negative_edge=False))
    assert len(got) == 3, got
    return 2


def _t_filter_sport() -> int:
    rows = [
        _row(predicted_winner="A", sport_type="tennis"),
        _row(predicted_winner="B", sport_type="soccer"),
        _row(predicted_winner="C", sport_type=None),
    ]
    got = list(filter_rows(rows, sports=["tennis"]))
    assert [r.predicted_winner for r in got] == ["A"], got
    got = list(filter_rows(rows, sports=["TENNIS"]))  # case-insensitive
    assert [r.predicted_winner for r in got] == ["A"], got
    return 2


def _t_filter_risk_bucket() -> int:
    rows = [
        _row(predicted_winner="A", risk_bucket="Lock"),
        _row(predicted_winner="B", risk_bucket="Lean"),
        _row(predicted_winner="C", risk_bucket="Coin-flip"),
        _row(predicted_winner="D", risk_bucket="Avoid"),
        _row(predicted_winner="E", risk_bucket=None),  # classifier failure
    ]
    # Default policy: Lock + Lean only.
    got = list(filter_rows(rows, risk_buckets=["Lock", "Lean"]))
    assert [r.predicted_winner for r in got] == ["A", "B"], got
    # Strictest: Lock only.
    got = list(filter_rows(rows, risk_buckets=["Lock"]))
    assert [r.predicted_winner for r in got] == ["A"], got
    # None bucket ALWAYS fails the gate when filter is active.
    got = list(filter_rows(rows, risk_buckets=["Lock", "Lean", "Coin-flip", "Avoid"]))
    assert [r.predicted_winner for r in got] == ["A", "B", "C", "D"], got
    # No filter → every row passes (None bucket included).
    got = list(filter_rows(rows, risk_buckets=None))
    assert len(got) == 5, got
    return 4


def _t_filter_market_implied() -> int:
    rows = [
        _row(predicted_winner="A", polymarket_implied_probability=0.65),  # agree
        _row(predicted_winner="B", polymarket_implied_probability=0.50),  # exactly threshold
        _row(predicted_winner="C", polymarket_implied_probability=0.41),  # directional disagree
        _row(predicted_winner="D", polymarket_implied_probability=None),  # missing
    ]
    # Default 0.50: keeps strict agreement (>=0.50), drops directional disagree
    # and missing.
    got = list(filter_rows(rows, min_market_implied_prob=0.50))
    assert [r.predicted_winner for r in got] == ["A", "B"], got
    # Stricter 0.55: drops the boundary-pass row too.
    got = list(filter_rows(rows, min_market_implied_prob=0.55))
    assert [r.predicted_winner for r in got] == ["A"], got
    # None implied prob ALWAYS fails the gate when filter is active.
    got = list(filter_rows(rows, min_market_implied_prob=0.0))
    assert [r.predicted_winner for r in got] == ["A", "B", "C"], got
    # No filter → every row passes (None implied included).
    got = list(filter_rows(rows, min_market_implied_prob=None))
    assert len(got) == 4, got
    return 4


# ---------------------------------------------------------------------------
# sum_exposure_cents — open-exposure gate input
# ---------------------------------------------------------------------------


def _t_sum_exposure_cents() -> int:
    # Kalshi sends `market_exposure_dollars` as a FixedPointDollars string
    # (e.g. "22.540000"); the Pydantic validator coerces to float. The
    # summer multiplies by 100 and rounds — exercise both the string-input
    # path and the None-skip path.
    positions = [
        MarketPosition.model_validate(
            {"ticker": "A", "position_fp": "5.00", "market_exposure_dollars": "22.540000"},
        ),
        MarketPosition.model_validate(
            {"ticker": "B", "position_fp": "-3.00", "market_exposure_dollars": "0.450000"},
        ),
        # Malformed / missing exposure — skipped, not counted as 0.
        MarketPosition.model_validate(
            {"ticker": "C", "position_fp": "1.00", "market_exposure_dollars": None},
        ),
    ]
    got = sum_exposure_cents(positions)
    # 22.54 + 0.45 = 22.99 → 2299 cents
    assert got == 2299, got
    # Empty list → 0
    assert sum_exposure_cents([]) == 0
    # All-None list → 0
    all_none = [
        MarketPosition.model_validate(
            {"ticker": "X", "market_exposure_dollars": None},
        ),
    ]
    assert sum_exposure_cents(all_none) == 0
    return 3


# ---------------------------------------------------------------------------
# classify_risk
# ---------------------------------------------------------------------------


def _t_classify() -> int:
    """classify_risk — bucket boundaries, asymmetric convergence, anchor flip."""
    n = 0

    # Clear Lock — lopsided, well-defended, blind estimate agrees with market.
    bucket, score = classify_risk(0.88, 0.90, 0.03, predicted_winner_is_team_a=True)
    assert bucket == BUCKET_LOCK, (bucket, score)
    assert score is not None and score >= THRESHOLD_LOCK, score
    n += 2

    # Coin-flip — middling magnitude + defensibility; the blind estimate and
    # the market agree on the winner with only a small same-side gap.
    bucket, score = classify_risk(0.55, 0.45, 0.05, predicted_winner_is_team_a=True)
    assert bucket == BUCKET_COINFLIP, (bucket, score)
    assert score is not None and THRESHOLD_COINFLIP <= score < THRESHOLD_LEAN, score
    n += 2

    # Directional disagreement — the market prices the predicted winner below
    # 0.5 (blind 0.61 vs market 0.41), so it favors the *other* side. The
    # disagreement penalty crushes convergence; what magnitude + defensibility
    # alone would have made a Lean drops to Coin-flip.
    bucket, score = classify_risk(0.61, 0.75, 0.20, predicted_winner_is_team_a=True)
    assert bucket == BUCKET_COINFLIP, (bucket, score)
    n += 1

    # Avoid — blind 0.56 vs market 0.30: the gap penalty *and* the
    # directional-disagreement penalty together crush convergence to 0, and
    # thin defensibility does the rest.
    bucket, score = classify_risk(0.56, 0.30, 0.26, predicted_winner_is_team_a=True)
    assert bucket == BUCKET_AVOID, (bucket, score)
    assert score is not None and score < THRESHOLD_COINFLIP, score
    n += 2

    # Unrated — the judge produced no defensibility score.
    bucket, score = classify_risk(0.80, None, 0.05, predicted_winner_is_team_a=True)
    assert bucket == BUCKET_UNRATED and score is None, (bucket, score)
    n += 1

    # No market implied AND no GBT prior → both convergence terms dropped,
    # weights renormalize to magnitude / defensibility share only. Under
    # the rebalanced post-GBT-convergence weights (M=0.40, D=0.25), the
    # surviving 0.65 weight share normalizes to ~0.615 M / ~0.385 D.
    from skimsmarkets.classify import W_MAGNITUDE, W_DEFENSIBILITY
    bucket, score = classify_risk(0.82, 0.72, None, predicted_winner_is_team_a=True)
    expected = (W_MAGNITUDE * 0.82 + W_DEFENSIBILITY * 0.72) / (
        W_MAGNITUDE + W_DEFENSIBILITY
    )
    assert score is not None and abs(score - expected) < 1e-9, score
    assert bucket == BUCKET_LOCK, (bucket, score)
    n += 2

    # GBT convergence — when the director matches GBT closely on the picked
    # side, the gbt_convergence term ~= 1.0 and lifts a borderline pick. A
    # 0.65/0.60/+0.05_market pick that's also 0.0 from GBT lands LEAN (the
    # GBT-convergence boost over the no-GBT case).
    bucket_with_gbt, score_with_gbt = classify_risk(
        0.65, 0.60, 0.05,
        predicted_winner_is_team_a=True,
        gap_to_gbt_signed=0.0,
    )
    bucket_without_gbt, score_without_gbt = classify_risk(
        0.65, 0.60, 0.05,
        predicted_winner_is_team_a=True,
    )
    # GBT-aligned pick should score STRICTLY higher than the same pick
    # without a GBT cross-check — the convergence term contributes a
    # positive value and the renormalization absorbs the dropped weight.
    assert score_with_gbt is not None and score_without_gbt is not None
    assert score_with_gbt > score_without_gbt, (
        score_with_gbt, score_without_gbt,
    )
    n += 2

    # GBT directional disagreement — director picks team_a at 0.55 but GBT
    # says team_a wins at 0.30 (gap = +0.25). That's a directional
    # disagreement on top of a large positive gap; the GBT convergence
    # term should collapse and demote the score noticeably vs the same
    # call with no GBT data.
    _, score_gbt_disagree = classify_risk(
        0.55, 0.65, 0.05,
        predicted_winner_is_team_a=True,
        gap_to_gbt_signed=0.25,
    )
    _, score_no_gbt = classify_risk(
        0.55, 0.65, 0.05,
        predicted_winner_is_team_a=True,
    )
    assert score_gbt_disagree is not None and score_no_gbt is not None
    assert score_gbt_disagree < score_no_gbt, (
        score_gbt_disagree, score_no_gbt,
    )
    n += 2

    # team_b winner — gap_to_market_signed is team_a-anchored, so the anchor
    # flip turns a -0.05 team_a gap into a +0.05 winner-frame gap (blind
    # estimate is *more* bullish on team_b than the market).
    bucket, score = classify_risk(0.70, 0.80, -0.05, predicted_winner_is_team_a=False)
    assert bucket == BUCKET_LOCK, (bucket, score)
    # The same raw gap read as a team_a winner means the blind estimate is
    # *less* bullish than the market → steeper penalty → strictly lower score.
    _, score_team_a = classify_risk(0.70, 0.80, -0.05, predicted_winner_is_team_a=True)
    assert score is not None and score_team_a is not None and score_team_a < score
    n += 2

    # bucket_rank ordering — Lock best (0), Unrated worst, unknown sorts last.
    assert bucket_rank(BUCKET_LOCK) == 0
    assert bucket_rank(BUCKET_AVOID) == 3
    assert bucket_rank(BUCKET_UNRATED) == len(BUCKET_ORDER) - 1
    assert bucket_rank("not-a-bucket") == len(BUCKET_ORDER)
    n += 4

    return n


# ---------------------------------------------------------------------------
# execute implied-probability gate
# ---------------------------------------------------------------------------


def _t_implied_gate() -> int:
    """_implied_at_or_above_max — the execute-time live Kalshi-price gate."""
    n = 0
    # No ceiling configured → gate inactive, every ask passes.
    assert _implied_at_or_above_max(0.95, None) is False
    n += 1
    # Live ask above the ceiling → gate fires (the trade is skipped).
    assert _implied_at_or_above_max(0.72, 0.60) is True
    n += 1
    # Live ask below the ceiling → gate passes, the trade proceeds.
    assert _implied_at_or_above_max(0.55, 0.60) is False
    n += 1
    # Exactly at the ceiling → fires, matching the rank slate's `>=`.
    assert _implied_at_or_above_max(0.60, 0.60) is True
    n += 1
    return n


# ---------------------------------------------------------------------------
# retro scoring metrics
# ---------------------------------------------------------------------------


def _t_metrics() -> int:
    """brier_score / log_loss / ECE / calibration_curve — formulas vs
    hand computation, empty-input and p∈{0,1} edge cases.
    """
    n = 0

    pairs = [(0.8, True), (0.6, False), (0.5, True)]
    # Brier = mean((p - y)^2), computed independently of the impl.
    exp_brier = ((0.8 - 1) ** 2 + (0.6 - 0) ** 2 + (0.5 - 1) ** 2) / 3
    got = brier_score(pairs)
    assert got is not None and abs(got - exp_brier) < 1e-9, got
    n += 1
    # log-loss = mean(-[y log p + (1-y) log(1-p)]).
    exp_ll = -(math.log(0.8) + math.log(1 - 0.6) + math.log(0.5)) / 3
    got = log_loss(pairs)
    assert got is not None and abs(got - exp_ll) < 1e-9, got
    n += 1

    # Empty input → None for both scoring rules.
    assert brier_score([]) is None
    assert log_loss([]) is None
    n += 2

    # p ∈ {0, 1} must not blow log-loss to -inf (the eps clamp).
    ll = log_loss([(1.0, False), (0.0, True)])
    assert ll is not None and math.isfinite(ll), ll
    n += 1

    # Perfectly-calibrated synthetic set → ECE ≈ 0: 10 rows at p=0.7 with
    # exactly 7 wins, 10 at p=0.4 with exactly 4 wins.
    well = (
        [(0.7, True)] * 7 + [(0.7, False)] * 3
        + [(0.4, True)] * 4 + [(0.4, False)] * 6
    )
    ece_well = expected_calibration_error(well)
    assert ece_well is not None and ece_well < 0.05, ece_well
    n += 1

    # Deliberately miscalibrated → large ECE: all p=0.9, only half win.
    bad = [(0.9, True)] * 5 + [(0.9, False)] * 5
    ece_bad = expected_calibration_error(bad)
    assert ece_bad is not None and ece_bad > 0.3, ece_bad
    n += 1

    # calibration_curve: exactly n_bins bins, coverage sums to len(pairs),
    # empty bins carry n=0 and None rates.
    curve = calibration_curve(well, n_bins=10)
    assert len(curve) == 10, len(curve)
    assert sum(b.n for b in curve) == len(well)
    assert all(
        b.mean_predicted is None and b.observed_freq is None
        for b in curve
        if b.n == 0
    )
    n += 3

    # ECE with far more bins than data is still finite (empty bins → 0).
    ece_sparse = expected_calibration_error(pairs, n_bins=50)
    assert ece_sparse is not None and math.isfinite(ece_sparse)
    n += 1

    # compute_metrics bundles everything; empty input → n=0, None metrics,
    # still a full n_bins curve.
    m = compute_metrics([])
    assert m.n == 0 and m.brier is None and m.log_loss is None
    assert m.ece is None and len(m.curve) == 10
    n += 2

    return n


# ---------------------------------------------------------------------------
# temperature-scaling calibration
# ---------------------------------------------------------------------------


def _t_calibration() -> int:
    """apply_temperature invariants (identity / 0.5 fixed point / no pick
    flip), fit_temperature recovery + refusal guards, classify_risk
    backward compatibility at T=1.0.
    """
    n = 0

    # Identity at T=1.0.
    for p in (0.01, 0.3, 0.5, 0.72, 0.99):
        assert abs(apply_temperature(p, 1.0) - p) < 1e-12, p
    n += 1

    # 0.5 is the fixed point for every temperature.
    for t in (0.25, 0.5, 1.0, 2.5, 5.0):
        assert abs(apply_temperature(0.5, t) - 0.5) < 1e-12, t
    n += 1

    # Never crosses 0.5 — the "can't flip the pick" guarantee.
    for t in (0.25, 0.5, 2.0, 5.0):
        assert apply_temperature(0.72, t) > 0.5, t
        assert apply_temperature(0.31, t) < 0.5, t
    n += 1

    # Finite at p ∈ {0, 1} — the _P_CLAMP guard.
    for t in (0.25, 1.0, 5.0):
        assert math.isfinite(apply_temperature(0.0, t)), t
        assert math.isfinite(apply_temperature(1.0, t)), t
    n += 1

    # T > 1 pulls toward 0.5 (de-confidence); T < 1 pushes away.
    assert apply_temperature(0.9, 2.0) < 0.9
    assert apply_temperature(0.9, 0.5) > 0.9
    n += 1

    # Round-trip recovery: emit overconfident scores from a known
    # T_true, draw outcomes from the *true* probabilities, and check
    # fit_temperature recovers T_true. Composition multiplies
    # temperatures, so emitting `apply_temperature(p_true, 1/T_true)`
    # means `apply_temperature(emitted, T_true) == p_true`. Fixed-seed
    # stdlib RNG — deterministic, no numpy dependency.
    rng = random.Random(42)
    t_true = 1.6
    pairs: list[tuple[float, bool]] = []
    for _ in range(1000):
        p_true = rng.uniform(0.5, 0.95)
        emitted = apply_temperature(p_true, 1.0 / t_true)
        won = rng.random() < p_true
        pairs.append((emitted, won))
    t_fit = fit_temperature(pairs)
    assert t_fit is not None and abs(t_fit - t_true) < 0.2, t_fit
    n += 1

    # Refusal guards: too few rows (both classes present, so it's the N
    # guard specifically), and single-class input.
    too_few = [(0.7, True), (0.6, False)] * ((MIN_FIT_N - 1) // 2)
    assert len(too_few) < MIN_FIT_N
    assert fit_temperature(too_few) is None
    assert fit_temperature([(0.7, True)] * 200) is None  # all wins
    assert fit_temperature([(0.6, False)] * 200) is None  # all losses
    n += 3

    # classify_risk: temperature=1.0 is byte-identical to the no-param
    # call — proves the new param is backward compatible.
    base = classify_risk(0.88, 0.90, 0.03, predicted_winner_is_team_a=True)
    with_t = classify_risk(
        0.88, 0.90, 0.03, predicted_winner_is_team_a=True, temperature=1.0
    )
    assert base == with_t, (base, with_t)
    n += 1

    # T > 1 shrinks the magnitude term → strictly lower risk_score on a
    # high-magnitude case.
    _, score_raw = classify_risk(
        0.88, 0.90, 0.03, predicted_winner_is_team_a=True, temperature=1.0
    )
    _, score_cooled = classify_risk(
        0.88, 0.90, 0.03, predicted_winner_is_team_a=True, temperature=2.0
    )
    assert score_raw is not None and score_cooled is not None
    assert score_cooled < score_raw, (score_raw, score_cooled)
    n += 1

    return n


# ---------------------------------------------------------------------------
# Unusual Whales (Hashdive) client schema — regression test against a
# real `/detail_agg` response captured at 2026-05-17 (the API migration
# from `api.unusualwhales.com/api/predictions/market/{id}` to
# `phx.unusualwhales.com/hashdive/api/assets/{id}/detail_agg`). Verifies
# the field renames (`tags_score` → `unusual_score`, `whale_trades` from
# the new fanned-out shape, `size`+`price` on trades instead of the old
# maker/taker amount pair, new insider signals `pnl_percent`/
# `invested_zscore`/`n_positions`) all decode without losing data the
# downstream consumers expect. If the vendor changes the response shape
# again, this test fires before the live pipeline silently degrades to
# `attached unusual-whales context to 0/N events`.
# ---------------------------------------------------------------------------


def _t_uw_schema_decode() -> int:
    import json

    from skimsmarkets.unusual_whales.client import _context_from_detail
    from skimsmarkets.unusual_whales.rendering import render_uw_block

    fixture = (
        REPO_ROOT
        / "tests/fixtures/uw_detail_agg_tennis_sinner_french_open_2026.json"
    )
    payload = json.loads(fixture.read_text())
    ctx = _context_from_detail(payload["asset_id"], payload)
    assert ctx is not None, "fixture should decode to a context"
    n = 1

    # Core identifiers + the renamed score field (tags_score → unusual_score).
    assert ctx.asset_id == payload["asset_id"]
    assert ctx.question and "Sinner" in ctx.question
    # outcome_label resolves outcomes[outcome_index] — `Yes` for this fixture.
    assert ctx.outcome_label == "Yes"
    # The new `tags_score` field carries the value the old client expected
    # under `unusual_score`. Float coercion off the JSON string must work.
    assert ctx.unusual_score is not None and ctx.unusual_score > 0
    n += 4

    # MCI scalar pair — same shape across the API migration.
    assert ctx.mci is not None
    assert ctx.mci.value is not None and ctx.mci.delta is not None
    n += 2

    # Liquidity block — best_bid / best_ask / mid_price / spread /
    # total_liquidity all preserved across the migration; the new
    # ask_liquidity / bid_liquidity fields are optional and harmless if
    # ignored.
    assert ctx.liquidity is not None
    assert ctx.liquidity.best_bid is not None and ctx.liquidity.best_ask is not None
    assert ctx.liquidity.total_liquidity is not None
    n += 3

    # Smart trades — fixture has at least one fill. Verifies the new
    # `size` + `price` shape lands cleanly (old shape was maker/taker
    # amount pair) and the derived `usdc_notional` math works.
    assert ctx.smart_trades, "fixture should carry at least one smart trade"
    trade = ctx.smart_trades[0]
    assert trade.size is not None and trade.size > 0
    assert trade.price is not None and 0 < trade.price < 1
    assert trade.usdc_notional is not None
    assert abs(trade.usdc_notional - trade.size * trade.price) < 1e-9
    n += 5

    # Insiders — fixture has multiple; the top insider carries the new
    # Hashdive-only signal fields. invested_zscore is the most directly
    # actionable (z>2 = "this wallet sized this market unusually large
    # vs its own trading-history baseline").
    assert ctx.insiders, "fixture should carry at least one insider"
    top = ctx.insiders[0]
    assert top.user_address and top.user_address.startswith("0x")
    assert top.total_invested_usd is not None
    assert top.invested_zscore is not None  # NEW field — must decode
    assert top.pnl_percent is not None  # NEW field
    assert top.n_positions is not None  # NEW field
    n += 5

    # has_actionable_signal — insiders present, so context is actionable
    # even though smart_trades is single-fill and unusual_score is < 5.0.
    assert ctx.has_actionable_signal() is True
    n += 1

    # Rendering — no crashes, contains the key field markers the
    # director prompt reads off. If a field rename breaks the render,
    # the director sees `?` placeholders and we want to catch that here.
    rendered = render_uw_block(ctx)
    assert "Flow signals" in rendered
    assert "Sinner" in rendered
    assert "tag weights:" in rendered
    assert "MCI:" in rendered
    n += 4

    return n


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    groups = [
        ("last_token", _t_last_token),
        ("extract_match_players via title", _t_extract_via_title),
        ("find_kalshi_match basic", _t_find_match_basic),
        ("find_kalshi_match no_kalshi_match", _t_find_match_no_kalshi),
        ("find_kalshi_match ambiguous", _t_find_match_ambiguous),
        ("find_kalshi_match market_closed", _t_find_match_closed_market),
        ("find_kalshi_match no_ask", _t_find_match_no_ask),
        ("find_kalshi_match unparseable", _t_find_match_unparseable),
        ("filter_rows confidence", _t_filter_confidence),
        ("filter_rows defensibility", _t_filter_defensibility),
        ("filter_rows negative_edge", _t_filter_negative_edge),
        ("filter_rows sport", _t_filter_sport),
        ("filter_rows risk_bucket", _t_filter_risk_bucket),
        ("filter_rows market_implied", _t_filter_market_implied),
        ("sum_exposure_cents", _t_sum_exposure_cents),
        ("classify_risk", _t_classify),
        ("execute implied-prob gate", _t_implied_gate),
        ("retro scoring metrics", _t_metrics),
        ("temperature calibration", _t_calibration),
        ("uw hashdive schema decode", _t_uw_schema_decode),
    ]
    failures = 0
    for name, fn in groups:
        try:
            n = fn()
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok    {name} ({n} asserts)")
    if failures:
        print(f"\n{failures} group(s) failed")
        return 1
    print(f"\nall {len(groups)} groups passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
