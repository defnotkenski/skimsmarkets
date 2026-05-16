"""Coarse hyperparameter grid for the tennis GBT.

Walks the parquet once via `build_training_table`, then iterates a
grid of CatBoost params against the same train/holdout split — so
the only thing varying across trials is the model fit, not the
feature derivation. Reports per-trial holdout Brier / log-loss /
AUC plus the early-stopping iteration so a follow-up Optuna run can
seed from the grid neighbourhood.

Not a CLI tool — invoke via `uv run python scripts/gbt_hparam_search.py`
once, archive the output, decide on a winning param set, then update
`_CATBOOST_PARAMS` in `gbt_train.py` accordingly.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, UTC
from itertools import product
from pathlib import Path

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
    TRAIN_CUTOFF, _auc, _brier, _log_loss,
)

log = logging.getLogger(__name__)


def _features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[list(ALL_FEATURE_COLUMNS)].copy()
    for col in NUMERIC_FEATURE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in CATEGORICAL_FEATURE_COLUMNS:
        out[col] = out[col].astype(str)
    return out


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
    cat_indices = [list(ALL_FEATURE_COLUMNS).index(c) for c in CATEGORICAL_FEATURE_COLUMNS]
    train_pool = Pool(
        data=_features(train_df), label=train_df["target"], cat_features=cat_indices,
    )
    holdout_pool = Pool(
        data=_features(holdout_df), label=holdout_df["target"], cat_features=cat_indices,
    )
    y_holdout = holdout_df["target"].to_numpy()

    # Coarse grid. Centred on the current defaults (depth=6, lr=0.03,
    # l2=4) and expanding ±1-2 levels on each axis.
    depths = [4, 6, 8]
    lrs = [0.02, 0.03, 0.05]
    l2s = [2.0, 4.0, 8.0]
    # min_data_in_leaf adds regularization at the leaf level — Catboost
    # default = 1 (no constraint), which can let small leaves overfit.
    # Try a few values; "None" means use the default.
    min_data_in_leafs: list[int | None] = [None, 30, 100]

    results: list[dict] = []
    grid = list(product(depths, lrs, l2s, min_data_in_leafs))
    log.info("running %d trials", len(grid))

    for i, (depth, lr, l2, mdl) in enumerate(grid):
        params = {
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "iterations": 3000,
            "learning_rate": lr,
            "depth": depth,
            "l2_leaf_reg": l2,
            "random_seed": 42,
            "use_best_model": True,
            "od_type": "Iter",
            "od_wait": 100,
            "verbose": False,
            "allow_writing_files": False,
        }
        if mdl is not None:
            params["min_data_in_leaf"] = mdl
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=holdout_pool)
        p = model.predict_proba(holdout_pool)[:, 1]
        brier = _brier(y_holdout, p)
        ll = _log_loss(y_holdout, p)
        auc = _auc(y_holdout, p)
        best_iter = model.get_best_iteration()
        results.append({
            "depth": depth, "lr": lr, "l2": l2, "min_data_in_leaf": mdl,
            "brier": brier, "log_loss": ll, "auc": auc,
            "best_iter": best_iter,
        })
        log.info(
            "trial %2d/%d  depth=%d lr=%.2f l2=%.1f mdl=%s  "
            "brier=%.4f  ll=%.4f  auc=%.4f  iter=%d",
            i + 1, len(grid), depth, lr, l2, mdl, brier, ll, auc, best_iter,
        )

    results.sort(key=lambda r: r["brier"])
    out_path = Path("models/tennis_gbt_hparam_search.json")
    out_path.write_text(json.dumps({
        "run_at_utc": datetime.now(UTC).isoformat(),
        "n_trials": len(results),
        "best": results[0],
        "top_10": results[:10],
        "all": results,
    }, indent=2))
    log.info("wrote %s", out_path)
    print("\n=== Top 10 trials by holdout Brier ===")
    for r in results[:10]:
        print(
            f"  brier={r['brier']:.4f}  ll={r['log_loss']:.4f}  auc={r['auc']:.4f}  "
            f"depth={r['depth']}  lr={r['lr']:.2f}  l2={r['l2']:.1f}  "
            f"mdl={r['min_data_in_leaf']}  iter={r['best_iter']}"
        )


if __name__ == "__main__":
    main()
