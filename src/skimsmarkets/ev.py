"""Expected-value math shared across ranker, executor, and retro scripts.

Tiny module — one pure function. Exists because both `pipeline.py` (rank-
time persistence of `ev_per_dollar`) and `execute/trader.py` (trade-time
EV gate + Kelly sizing) need the same formula, and pulling pipeline.py
into the executor would drag in the entire LLM stack (anthropic, agents,
sports modules) for a 5-line math helper. Keeping it standalone preserves
the executor's lean import surface — `skims execute --dry-run` runs
without any LLM client loaded.
"""

from __future__ import annotations


def compute_ev_per_dollar(
    model_p: float | None, market_p: float | None,
) -> float | None:
    """Expected value per $1 staked on the predicted side.

    Math:
      payoff_ratio = (1 - market_p) / market_p   # $ you win per $1 risked
      ev_per_dollar = model_p * payoff_ratio - (1 - model_p)

    Positive = the bet has asymmetric edge (pays more than fair); negative
    = market offers worse-than-fair odds for the predicted side. Returns
    None when either probability is missing OR market_p is at the
    degenerate edges 0 / 1 (payoff is undefined). Both `model_p` and
    `market_p` must be on the SAME side frame — for this codebase, that's
    the predicted-winner frame (where `polymarket_implied_probability`
    lives per the pipeline docstring).
    """
    if model_p is None or market_p is None:
        return None
    if not (0.0 < market_p < 1.0):
        return None
    if not (0.0 <= model_p <= 1.0):
        return None
    payoff_ratio = (1.0 - market_p) / market_p
    return model_p * payoff_ratio - (1.0 - model_p)
