from __future__ import annotations

from skimsmarkets.agents.schemas import (
    KELLY_BANKROLL_CAP,
    MarketPrediction,
    PositionSizing,
    SizedMarketPrediction,
)
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
    polymarket_market: PolymarketMarket,
) -> PositionSizing:
    """Kelly-size the director's prediction against Polymarket's current yes_ask.

    Sizing is reference only — there's no upstream buy/pass gate, so a zero
    fraction just means "not +EV at the current ask." The leaderboard ranks on
    predicted probability regardless.
    """
    notes: list[str] = []
    win_prob = prediction.predicted_yes_probability

    # Walrus-bind the ask so the 0.0 < ask < 1.0 guard narrows float|None to float
    # without a cast (CLAUDE.md's static-checker note).
    if (ask := polymarket_market.yes_ask_dollars) is None or not (0.0 < ask < 1.0):
        notes.append(
            f"Polymarket ask unavailable or out of range "
            f"(yes_ask={polymarket_market.yes_ask_dollars}); cannot size a position."
        )
        return PositionSizing(
            entry_price_dollars=None,
            win_probability=win_prob,
            edge=0.0,
            full_kelly_fraction=0.0,
            half_kelly_fraction=0.0,
            capped_half_kelly_fraction=0.0,
            notes=notes,
        )

    edge = win_prob - ask
    full = _kelly_fraction(win_prob, ask)
    if full == 0.0 and edge <= 0:
        notes.append(
            f"Predicted win probability {win_prob:.4f} is not above Polymarket ask "
            f"{ask:.4f}; clamped Kelly to 0."
        )

    half = full * 0.5
    capped = min(half, KELLY_BANKROLL_CAP)

    return PositionSizing(
        entry_price_dollars=ask,
        win_probability=win_prob,
        edge=edge,
        full_kelly_fraction=full,
        half_kelly_fraction=half,
        capped_half_kelly_fraction=capped,
        notes=notes,
    )


def wrap_with_sizing(
    prediction: MarketPrediction,
    polymarket_market: PolymarketMarket,
) -> SizedMarketPrediction:
    return SizedMarketPrediction(
        prediction=prediction,
        sizing=compute_sizing(prediction, polymarket_market),
    )
