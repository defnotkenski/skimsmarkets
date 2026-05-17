"""One-off ablation script for iter 5 — measures the marginal value
of (a) the new KM hold% features and (b) pruning the low-importance
H2H clutch sub-counts. Three variants trained with the same
hyperparams against the same train/holdout split:

  - baseline_iter2: current ALL_FEATURE_COLUMNS (45 features, includes
    new KM scalars added in iter 5).
  - drop_km: ALL_FEATURE_COLUMNS minus the KM hold-pct and
    serve-point-win-pct features.
  - drop_h2h_clutch: ALL_FEATURE_COLUMNS minus the 8 low-importance
    H2H clutch sub-count features (decider/tiebreak/comeback/
    close_match advantages + their n_priors counterparts; main
    h2h_advantage + n_h2h_priors retained).
  - drop_both: drop_km + drop_h2h_clutch.

Reports per-variant Brier / log-loss / AUC alongside the iter2-baseline
metrics that pre-dated this work. Pick the variant with the best
holdout Brier; if drop_both wins, the H2H clutch sub-counts AND KM
features were both noise. If baseline wins, both were signal.

Not a CLI tool — invoke via `uv run python scripts/gbt_ablation_iter5.py`.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier, Pool

from skimsmarkets.tennis.gbt_backfill import (
    PLAYER_PROFILES_PATH,
    RAW_MATCHES_PATH,
)
from skimsmarkets.tennis.gbt_features import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    build_training_table,
)
from skimsmarkets.tennis.gbt_rankings_backfill import RANKINGS_HISTORY_PATH
from skimsmarkets.tennis.gbt_train import (
    TRAIN_CUTOFF,
    _CATBOOST_PARAMS,
    _auc,
    _brier,
    _log_loss,
)

log = logging.getLogger(__name__)


_KM_FEATURES: tuple[str, ...] = (
    "career_serve_point_win_pct_diff",
    "career_hold_pct_diff",
)

# H2H clutch sub-counts identified as the 8 lowest-importance features
# in iter 2 (combined ~2 importance vs the ~370 total). The main
# h2h_advantage + n_h2h_priors carry more weight (>1 combined) and
# stay in the model.
_H2H_CLUTCH_SUB_COUNTS: tuple[str, ...] = (
    "h2h_decider_advantage",
    "n_h2h_decider_priors",
    "h2h_tiebreak_advantage",
    "n_h2h_tiebreak_priors",
    "h2h_comeback_advantage",
    "n_h2h_comeback_priors",
    "h2h_close_match_advantage",
    "n_h2h_close_match_priors",
)


def _features(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    out = df[list(cols)].copy()
    for col in cols:
        if col in NUMERIC_FEATURE_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = out[col].astype(str)
    return out


def _train_and_score(
    train_df: pd.DataFrame, holdout_df: pd.DataFrame, cols: tuple[str, ...],
) -> dict[str, Any]:
    cat_indices = [
        list(cols).index(c)
        for c in CATEGORICAL_FEATURE_COLUMNS
        if c in cols
    ]
    train_pool = Pool(
        data=_features(train_df, cols),
        label=train_df["target"],
        cat_features=cat_indices,
    )
    holdout_pool = Pool(
        data=_features(holdout_df, cols),
        label=holdout_df["target"],
        cat_features=cat_indices,
    )
    model = CatBoostClassifier(**{**_CATBOOST_PARAMS, "verbose": False})
    model.fit(train_pool, eval_set=holdout_pool)
    p = model.predict_proba(holdout_pool)[:, 1]
    y = holdout_df["target"].to_numpy()
    return {
        "n_features": len(cols),
        "brier": _brier(y, p),
        "log_loss": _log_loss(y, p),
        "auc": _auc(y, p),
        "best_iter": model.get_best_iteration(),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    matches_df = pd.read_parquet(RAW_MATCHES_PATH)
    profiles_df = pd.read_parquet(PLAYER_PROFILES_PATH)
    rankings_df = pd.read_parquet(RANKINGS_HISTORY_PATH)

    log.info("building training table (this is the slow part)")
    table = build_training_table(
        matches_df, profiles_df, rankings_df=rankings_df,
    )
    rows = table.rows
    rows["match_date"] = pd.to_datetime(rows["match_date"]).dt.date
    train_mask = rows["match_date"] <= TRAIN_CUTOFF
    train_df = rows.loc[train_mask].reset_index(drop=True)
    holdout_df = rows.loc[~train_mask].reset_index(drop=True)
    log.info(
        "split: train=%d holdout=%d (cutoff=%s)",
        len(train_df), len(holdout_df), TRAIN_CUTOFF,
    )

    full = tuple(ALL_FEATURE_COLUMNS)
    drop_km = tuple(c for c in full if c not in _KM_FEATURES)
    drop_h2h = tuple(c for c in full if c not in _H2H_CLUTCH_SUB_COUNTS)
    drop_both = tuple(
        c for c in full
        if c not in _KM_FEATURES and c not in _H2H_CLUTCH_SUB_COUNTS
    )

    variants: dict[str, tuple[str, ...]] = {
        "baseline_iter2_plus_km": full,
        "drop_km": drop_km,
        "drop_h2h_clutch": drop_h2h,
        "drop_both": drop_both,
    }
    results: dict[str, dict[str, Any]] = {}
    for name, cols in variants.items():
        log.info("=== training variant: %s (n_features=%d) ===", name, len(cols))
        results[name] = _train_and_score(train_df, holdout_df, cols)
        r = results[name]
        log.info(
            "  brier=%.5f  ll=%.5f  auc=%.5f  best_iter=%d",
            r["brier"], r["log_loss"], r["auc"], r["best_iter"],
        )

    out_path = Path("models/tennis_gbt_iter5_ablation.json")
    out_path.write_text(json.dumps({
        "run_at_utc": datetime.now(UTC).isoformat(),
        "train_cutoff": str(TRAIN_CUTOFF),
        "n_train": len(train_df),
        "n_holdout": len(holdout_df),
        "variants": results,
    }, indent=2))
    log.info("wrote %s", out_path)
    print()
    print("=== Iter 5 ablation results ===")
    for name, r in sorted(results.items(), key=lambda x: x[1]["brier"]):
        print(
            f"  {name:30s}  n_feat={r['n_features']:2d}  "
            f"brier={r['brier']:.5f}  ll={r['log_loss']:.5f}  "
            f"auc={r['auc']:.5f}  best_iter={r['best_iter']}"
        )


if __name__ == "__main__":
    main()
