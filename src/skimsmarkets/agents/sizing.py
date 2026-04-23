from __future__ import annotations

from skimsmarkets.agents.schemas import (
    KELLY_BANKROLL_CAP,
    MarketPrediction,
    PositionSizing,
    SizedMarketPrediction,
)
from skimsmarkets.kalshi.models import KalshiMarket
from skimsmarkets.polymarket.models import PolymarketMarket


def _kelly_fraction(win_prob: float, entry_price: float) -> float:
    """Kelly-optimal fraction of bankroll for a binary contract.

    For a contract costing `entry_price` (0-1) that pays 1 on win:
        edge = win_prob - entry_price
        kelly = edge / (1 - entry_price)   when edge > 0
    """
    if entry_price <= 0.0 or entry_price >= 1.0:
        return 0.0
    edge = win_prob - entry_price
    if edge <= 0:
        return 0.0
    return edge / (1.0 - entry_price)


def compute_sizing(
    prediction: MarketPrediction,
    kalshi_market: KalshiMarket,
    polymarket_market: PolymarketMarket | None,
) -> PositionSizing:
    """Translate a director's MarketPrediction into a Kelly-sized position at the
    venue with the better entry (lower yes_ask).

    Respects the director's recommendation (never flips sides). When both venues
    are illiquid or both are -EV against the predicted probability, clamps to
    zero with a note — the director's call is honored but the math decides size.
    """
    notes: list[str] = []

    if prediction.recommendation == "pass":
        return PositionSizing(
            side="none",
            venue="none",
            venue_market_id=None,
            entry_price_dollars=None,
            win_probability=None,
            edge=0.0,
            full_kelly_fraction=0.0,
            half_kelly_fraction=0.0,
            capped_half_kelly_fraction=0.0,
            notes=notes,
        )

    # Only remaining recommendation is buy_yes — the director never produces buy_no.
    win_prob = prediction.predicted_yes_probability

    # Candidate set, filtered to venues with valid asks. One-to-one with venue labels
    # in PositionSizing.venue. Walrus-bind each ask so the 0.0<ask<1.0 guard
    # narrows `float | None` to `float` without a cast.
    candidates: list[tuple[str, float, str]] = []
    if (k_ask := kalshi_market.yes_ask_dollars) is not None and 0.0 < k_ask < 1.0:
        candidates.append(("kalshi", k_ask, kalshi_market.ticker))
    if polymarket_market is not None and (
        (p_ask := polymarket_market.yes_ask_dollars) is not None and 0.0 < p_ask < 1.0
    ):
        candidates.append(("polymarket", p_ask, polymarket_market.slug))

    if not candidates:
        kalshi_note = (
            f"kalshi_ask={kalshi_market.yes_ask_dollars}"
            if kalshi_market.yes_ask_dollars is not None
            else "kalshi_ask=missing"
        )
        poly_note = (
            f"polymarket_ask={polymarket_market.yes_ask_dollars}"
            if polymarket_market and polymarket_market.yes_ask_dollars is not None
            else "polymarket=unavailable"
        )
        notes.append(
            f"Director recommended buy_yes but no venue has a tradeable ask "
            f"({kalshi_note}, {poly_note}); cannot size a position."
        )
        return PositionSizing(
            side="yes",
            venue="none",
            venue_market_id=None,
            entry_price_dollars=None,
            win_probability=win_prob,
            edge=0.0,
            full_kelly_fraction=0.0,
            half_kelly_fraction=0.0,
            capped_half_kelly_fraction=0.0,
            notes=notes,
        )

    # Pick the venue with the lowest ask — better entry for a YES buyer.
    candidates.sort(key=lambda c: c[1])
    venue, entry, venue_market_id = candidates[0]

    edge = win_prob - entry
    full = _kelly_fraction(win_prob, entry)
    if full == 0.0 and edge <= 0:
        if len(candidates) == 2:
            other = candidates[1]
            notes.append(
                f"Director recommended buy_yes but both venues are -EV at current "
                f"asks (kalshi={'%.4f' % k if (k := kalshi_market.yes_ask_dollars) is not None else '—'}, "
                f"polymarket={'%.4f' % p if polymarket_market and (p := polymarket_market.yes_ask_dollars) is not None else '—'}) "
                f"vs predicted win probability {win_prob:.4f}; clamped to 0. "
                f"Better-priced venue was {venue} at {entry:.4f}; other was {other[0]} at {other[1]:.4f}."
            )
        else:
            notes.append(
                f"Director recommended buy_yes but {venue}_ask={entry:.4f} "
                f"makes this -EV against predicted win probability {win_prob:.4f}; clamped to 0."
            )

    half = full * 0.5
    capped = min(half, KELLY_BANKROLL_CAP)

    return PositionSizing(
        side="yes",
        venue=venue,  # type: ignore[arg-type]
        venue_market_id=venue_market_id,
        entry_price_dollars=entry,
        win_probability=win_prob,
        edge=edge,
        full_kelly_fraction=full,
        half_kelly_fraction=half,
        capped_half_kelly_fraction=capped,
        notes=notes,
    )


def wrap_with_sizing(
    prediction: MarketPrediction,
    kalshi_market: KalshiMarket,
    polymarket_market: PolymarketMarket | None = None,
) -> SizedMarketPrediction:
    return SizedMarketPrediction(
        prediction=prediction,
        sizing=compute_sizing(prediction, kalshi_market, polymarket_market),
    )
