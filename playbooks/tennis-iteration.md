# Tennis selector / GBT iteration playbook

Methodology + iteration toolkit for the tennis pre-LLM selector (`tennis/selection_scorers.py`), the outcome GBT (`tennis/gbt.py`), and any future per-lens variants. Anti-patterns and rejected approaches live in memory (`project_tennis_iteration_archive.md`); this file is the HOW.

## Where the code lives

| Concern | File |
|---|---|
| Confidence-mode selector (v1) | `tennis/selection_scorers.py:score_v1_selection` |
| EV-mode selector | `tennis/selection_scorers.py:score_ev_v1_selection` |
| Production wrappers | `selection.py:_tennis_imbalance_v1`, `_tennis_imbalance_ev_v1` |
| Mode dispatch | `selection.py:imbalance_score` (reads per-call `mode=` arg, falls through to `cfg.KALSHI_DEFAULT_TRADE_MODE`) |
| Backtest harness | `tennis/selection_backtest.py:run_selection_backtest_multi` |
| EV-mode ablation driver | `scripts/ev_selector_ablation.py` (re-runnable, `--persist` for scorecards) |
| GBT predictor | `tennis/gbt.py:predict_for_event` |
| GBT trainer | `tennis/gbt_train.py` |
| GBT features | `tennis/gbt_features.py` (`NUMERIC_FEATURE_COLUMNS`, `CATEGORICAL_FEATURE_COLUMNS`, `HistoryStore`, `PlayerHistory`) |
| Parquet backfill | `tennis/gbt_backfill.py`, `tennis/gbt_rankings_backfill.py` |

## Selector ablation (inline pattern)

No CLI subcommand — use this inline pattern. Multi-scorer single-pass amortizes GBT inference across all variants; ~30s wall on the 2025+ holdout.

```python
from datetime import date
from skimsmarkets.tennis.selection_backtest import (
    run_selection_backtest_multi, score_rank_points_ratio, make_score_random,
)
from skimsmarkets.tennis.selection_scorers import (
    make_composite_scorer, TIER_REGISTRY, TierContribution,
    score_v1_selection, score_ev_v1_selection,
)

T = TIER_REGISTRY

def scale(tier_fn, mult):
    def scaled(**kw):
        c = tier_fn(**kw); return TierContribution(c.name, c.value * mult, c.cap * mult)
    return scaled

def make(name, tier_specs, base_mult=1.0):
    """tier_specs = list of str (cap 1.0) or (str, mult) tuples."""
    fns = []
    for spec in tier_specs:
        if isinstance(spec, tuple):
            nm, mult = spec; fns.append(scale(T[nm], mult))
        else:
            fns.append(T[spec])
    if base_mult == 1.0:
        return make_composite_scorer(name, fns)
    def base_fn(**kw): return score_rank_points_ratio(**kw) * base_mult
    return make_composite_scorer(name, fns, base_fn=base_fn)

scorers = {
    'baseline_rank': score_rank_points_ratio,
    'random':        make_score_random(42)[1],
    'v1_confidence': score_v1_selection,
    'v1_ev':         score_ev_v1_selection,
    # add candidates here
}
results = run_selection_backtest_multi(scorers, start_date=date(2025, 1, 1))
for name, res in results.items():
    h = res.holdout
    print(f'{name}: lock={h.precision_at_k_lock:.4f}  '
          f'realized=${h.mean_realized_return_at_k:+.4f}  '
          f'ev_prime={h.precision_at_k_ev_prime:.4f}  (n_ev={h.n_ev_labelable})')
```

For EV-mode iteration specifically, `scripts/ev_selector_ablation.py --persist` runs the canonical baseline + candidate set and writes scorecards to `models/selection_backtest/<name>.metrics.json`.

## Scorer interface contract (for new tier functions)

```python
def _tier_my_new_signal(*, a_stats, b_stats, surface=None, **_: Any) -> TierContribution:
    """One-line docstring."""
    # ... compute signed contribution ...
    return TierContribution("my_new_signal", value, cap)
```

Required kwargs the harness passes (accept what you need, use `**_: Any` for forward-compat):

- `a_stats`, `b_stats` — `TennisPlayerStats` projections (rank, form, surface W/L, serve %, etc.)
- `a_history`, `b_history` — raw `PlayerHistory` (richer than stats; gives Elo, h2h_against, by_surface aggregates). **None in production** unless your production wrapper plumbs it (see `_tennis_imbalance_ev_v1` for the GBT-bundle-Elo pattern). Handle None gracefully.
- `surface`, `best_of`, `tour`, `match_date`, `round_id`, `rank_id` — match-level context
- Harness primitives: `h2h_total_meetings`, `a_total_matches`, `b_total_matches`, `a_surface_matches`, `b_surface_matches`

Return: `TierContribution(name, value, cap)` with `value` already capped to `[-cap, +cap]` (or `[0, cap]` for bonus-only tiers). Register in `TIER_REGISTRY` to make composable via `make()` above.

## Iteration discipline

**Noise floor**:
- Lock-precision gains under ~0.005 are noise.
- EV realized-return lifts under ~$0.005 are noise.
- Sub-noise-floor changes need verification with a different `start_date` cut or a different `make_score_random` seed.

**Per-tier isolation first, composite ablation second**:
1. Build a single-tier scorer with `base=0.5` neutral.
2. Measure each candidate signal in isolation.
3. Compose the top performers.
4. Minus-one ablation on the composite — drops redundant tiers (e.g. `ev_underdog_elo` was dropped this way for being redundant with `ev_elo_rank_gap`).

**Weight sweep with care**:
- Amplifying a load-bearing tier often hurts — signal saturates at the existing cap. Start with downscales (0.5×) before upscales (1.5×, 2×).
- Base sweeps in [0.0, 0.4] are usually equivalent for additive composites (no clipping). Base > 0.4 introduces clipping at 1.0 and loses differentiation.

**Holdout composition artifact** (CRITICAL when comparing across parquet versions): any change to the parquet (wider top_n, deeper pages, lowered priors gate) changes which matches qualify under the cold-start gate. Wrong comparison: "new parquet gave Brier 0.208 vs old 0.211 — better." Right comparison: compute the metric on the INTERSECTION of matches qualifying in both parquets, AND report the full-pool number separately as a coverage diagnostic.

**K-robustness + per-tour breakdown**: always verify lifts hold at K=3 / K=5 / K=10 and on ATP / WTA independently. ATP and WTA can move in opposite directions on the same change.

## GBT iteration methodology

Apply this pattern to any future GBT iteration:

1. **Data-quality audit FIRST**. Inspect per-feature null rates + mean/std on the training table. The iter5 session found two 100%-NaN features stalled on a `parse_score_details(best_of=None)` bug because the parquet has 100% null `best_of`. Plus `age_diff` at 85% NaN because profiles cover only ~398 of 11k+ unique player IDs (see `project_parquet_backfill_iteration.md`).
2. **Cheap derived features second**. Look at literature (Klaassen-Magnus TPW%, Buhamra 2025 "Age.30", age × Elo interactions, multi-scale form windows). Each typically adds 1-5% importance and tiny but real Brier lift.
3. **Hparam search third**. ~81-trial grid (depth × lr × l2 × min_data_in_leaf) costs ~30 min wall. Diminishing returns past that.
4. **Ablation pruning last**. Drop features with importance <0.3 and re-measure.

## Mode dispatch reminder

The selector reads its mode from either the caller-supplied `mode=` kwarg (CLI plumbs `--mode ev` to here) or `cfg.KALSHI_DEFAULT_TRADE_MODE`. The executor uses identical semantics. Both default to `"confidence"` until the operator opts in.

To use EV mode end-to-end:

```bash
skims rank --mode ev --sport tennis
skims execute --mode ev
```

Or to make EV the default permanently: edit `config.py:KALSHI_DEFAULT_TRADE_MODE = "ev"`. Recommend doing this only after shadow-validation (run both selectors in parallel on real slates for 1-2 weeks).

## Director-blinding invariant

LLM stages (fetchers, reasoners, director, judge) never see Polymarket bid/ask, implied probability, or price history. Selector + classifier + executor are NOT LLM stages and CAN use market price. Never pipe price data into any agent prompt — verify by greping the agents/ directory before merging.

## Performance ceiling expectations

- **GBT Brier**: at ~0.211, already SOTA-comparable for an iid pre-match prior on truncated historical data. Further lifts are data-bound, not algorithm-bound.
- **Confidence selector**: plateaued around 0.249 Lock with the current 12-tier vocabulary; +29% over rank-points baseline. Next gains likely need (a) new data signals via parquet backfill, (b) different architecture (multiplicative, per-tour models, GBT-stacked-on-algo final layer), or (c) labels closer to production reality once `logs/runs/` accumulates 500+ rows.
- **EV selector**: plateaued around +$0.1475 realized return per $1 (+97% over base rate). Same levers as confidence selector apply; ATP shows stronger lift than WTA, so per-tour tuning is a low-hanging future lever.

See `project_parquet_backfill_iteration.md` for the active backfill plan to break the data-bound ceiling.
