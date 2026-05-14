"""Plain-assert smoke tests for `skims execute` — runnable as a script.

Usage:

    uv run python tests/test_execute.py

No pytest dependency (the project doesn't carry one). Each `_t_*`
function returns the count of assertions it made; `main()` runs them
and prints a one-line tally per group.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

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

    # No market implied → convergence term dropped, weights renormalize to
    # 0.5 magnitude / 0.5 defensibility.
    bucket, score = classify_risk(0.82, 0.72, None, predicted_winner_is_team_a=True)
    assert score is not None and abs(score - (0.5 * 0.82 + 0.5 * 0.72)) < 1e-9, score
    assert bucket == BUCKET_LOCK, (bucket, score)
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
        ("sum_exposure_cents", _t_sum_exposure_cents),
        ("classify_risk", _t_classify),
        ("execute implied-prob gate", _t_implied_gate),
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
