from __future__ import annotations

from skimsmarkets.agents.schemas import (
    KELLY_BANKROLL_CAP,
    MarketPrediction,
    PositionSizing,
    SizedMarketPrediction,
)
from skimsmarkets.kalshi.models import KalshiMarket


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


def compute_sizing(prediction: MarketPrediction, market: KalshiMarket) -> PositionSizing:
    """Translate a director's MarketPrediction into a position size under full + half Kelly.

    Respects the director's recommendation (does not flip sides), but clamps to zero and
    adds a note when the recommended side is -EV against the current ask price.
    """
    notes: list[str] = []

    if prediction.recommendation == "pass":
        return PositionSizing(
            side="none",
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
    entry = market.yes_ask_dollars

    if entry is None:
        notes.append(
            "Director recommended buy_yes but yes_ask is unavailable "
            "(illiquid market); cannot size a position."
        )
        return PositionSizing(
            side="yes",
            entry_price_dollars=None,
            win_probability=win_prob,
            edge=0.0,
            full_kelly_fraction=0.0,
            half_kelly_fraction=0.0,
            capped_half_kelly_fraction=0.0,
            notes=notes,
        )

    edge = win_prob - entry
    full = _kelly_fraction(win_prob, entry)
    if full == 0.0 and edge <= 0:
        notes.append(
            f"Director recommended buy_yes but yes_ask={entry:.4f} "
            f"makes this -EV against predicted win probability {win_prob:.4f}; clamped to 0."
        )

    half = full * 0.5
    capped = min(half, KELLY_BANKROLL_CAP)

    return PositionSizing(
        side="yes",
        entry_price_dollars=entry,
        win_probability=win_prob,
        edge=edge,
        full_kelly_fraction=full,
        half_kelly_fraction=half,
        capped_half_kelly_fraction=capped,
        notes=notes,
    )


def wrap_with_sizing(
    prediction: MarketPrediction, market: KalshiMarket,
) -> SizedMarketPrediction:
    return SizedMarketPrediction(
        prediction=prediction,
        sizing=compute_sizing(prediction, market),
    )
