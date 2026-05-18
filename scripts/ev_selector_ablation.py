"""EV-mode pre-LLM tennis selector ablation harness.

Sibling to the implicit ablation pattern in
`memory/project_pre_llm_tennis_algo_overhaul.md` (which targets
Lock-precision). This script targets the EV-side metrics added to
`tennis/selection_backtest.py:SlateMetrics` in the 2026-05-17 EV-mode
work:

  - mean_realized_return_at_k  — \\$ return per \\$1 staked on GBT's pick
                                at rank-implied market_p. THE headline.
  - precision_at_k_ev_prime    — % picks landing in the Prime EV bucket
                                (EV ≥ 0.30 by `classify.classify_ev`).
  - precision_at_k_ev_edge_or  — % picks in Edge OR Prime (EV ≥ 0.15,
    better                       i.e. above the production gate).

Run from project root:

    uv run python scripts/ev_selector_ablation.py

Walks the 127k-match parquet via `run_selection_backtest_multi` (one
multi-scorer single-pass amortizes GBT inference). Holdout = matches
on/after `--start-date` (default 2025-01-01). ~30s wall.

Scorecards land in `models/selection_backtest/<scorer_name>.metrics.json`
on `--persist`. Without `--persist` the script just prints the
comparison table to stdout.
"""

from __future__ import annotations

import argparse
from datetime import date

from skimsmarkets.tennis.selection_backtest import (
    make_score_random,
    run_selection_backtest_multi,
    score_rank_points_ratio,
    write_scorecard,
)
from skimsmarkets.tennis.selection_scorers import (
    TIER_REGISTRY,
    TierContribution,
    make_composite_scorer,
    score_ev_v1_selection,
    score_v1_selection,
)


def _scale(tier_fn, mult):
    def scaled(**kw):
        c = tier_fn(**kw)
        return TierContribution(c.name, c.value * mult, c.cap * mult)
    return scaled


def _make_with_base(name, spec, base):
    fns = []
    for s in spec:
        if isinstance(s, tuple):
            nm, m = s
            fns.append(_scale(TIER_REGISTRY[nm], m))
        else:
            fns.append(TIER_REGISTRY[s])
    return make_composite_scorer(name, fns, base_fn=lambda **_: base)[1]


def _build_scorers() -> dict:
    """The full ablation table: baselines + every documented variant
    from the v2…v5 iteration chain, plus the locked-in v1 EV scorer
    pulled directly from `selection_scorers`.

    Add a new row here when you want to test a candidate; re-running
    overwrites the same scorecard names so diffs are easy to spot.
    """
    TOP6 = [
        "ev_elo_rank_gap",
        "ev_underdog_form",
        "ev_underdog_serve",
        "ev_underdog_surface",
        "ev_h2h_underdog",
        "ev_competitive_floor",
    ]
    return {
        # Baselines.
        "baseline_rank_points_ratio": score_rank_points_ratio,
        "baseline_random_seed42": make_score_random(42)[1],
        "v1_confidence_lock_tuned": score_v1_selection,
        # The winner — same composition as `score_ev_v1_selection` for
        # apples-to-apples comparison through the driver.
        "v1_ev_winner_locked_in": score_ev_v1_selection,
        # Reference variants from the iteration chain (kept so the diff
        # against `v1_ev_winner_locked_in` re-validates the choices).
        "v1_ev_all_tiers": _make_with_base(
            "v1_ev_all_tiers",
            TOP6 + ["ev_underdog_elo"],
            0.5,
        ),
        "v1_ev_minus_underdog_elo": _make_with_base(
            "v1_ev_minus_underdog_elo", TOP6, 0.5,
        ),
        "v1_ev_minimal_elo_plus_floor": _make_with_base(
            "v1_ev_minimal_elo_plus_floor",
            ["ev_elo_rank_gap", "ev_competitive_floor"],
            0.4,
        ),
    }


def _print_table(results: dict) -> None:
    print()
    print(
        f"{'scorer':38s}  "
        f"realized   ev_prime  ev_edge+   mean_ev  lock     LoL     n_ev"
    )
    print("-" * 105)
    # Pull base rates from any scorer (they're slate-pool-wide).
    h = next(iter(results.values())).holdout
    print(
        f"{'BASE RATE (pool average)':38s}  "
        f"${h.base_mean_realized_return:+.4f}    "
        f"{h.base_rate_ev_prime:.4f}    "
        f"{h.base_rate_ev_edge_or_better:.4f}    -        "
        f"{h.base_rate_lock:.4f}   {h.base_rate_lock_or_lean:.4f}    -"
    )
    print()
    for name, res in results.items():
        h = res.holdout
        print(
            f"{name:38s}  "
            f"${h.mean_realized_return_at_k:+.4f}    "
            f"{h.precision_at_k_ev_prime:.4f}    "
            f"{h.precision_at_k_ev_edge_or_better:.4f}    "
            f"${h.mean_ev_at_k:+.4f}  "
            f"{h.precision_at_k_lock:.4f}   "
            f"{h.precision_at_k_lock_or_lean:.4f}   "
            f"{h.n_ev_labelable}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=date(2025, 1, 1),
        help="Holdout cutoff (matches >= this date are scored). "
        "Default 2025-01-01 matches the confidence-mode ablation chain.",
    )
    ap.add_argument(
        "--slate-cap",
        type=int,
        default=5,
        help="Top-K per slate. Default 5 = `MAX_SLATE_EVENTS`.",
    )
    ap.add_argument(
        "--persist",
        action="store_true",
        help="Write per-scorer scorecards to "
        "models/selection_backtest/<name>.metrics.json.",
    )
    args = ap.parse_args()

    scorers = _build_scorers()
    results = run_selection_backtest_multi(
        scorers, slate_cap=args.slate_cap, start_date=args.start_date,
    )
    _print_table(results)
    if args.persist:
        for res in results.values():
            path = write_scorecard(res)
            print(f"  → wrote {path}")


if __name__ == "__main__":
    main()
