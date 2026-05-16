"""Multi-seed bagging + isotonic calibration for the tennis GBT.

Two-stage post-tuning improvement loop:

  1. **Bagging.** Train N CatBoost models with the SAME hyperparams
     but different `random_seed` values, average their holdout
     probabilities. Each fit injects different stochasticity into
     CatBoost's split selection (`random_strength` defaults to 1.0,
     `bagging_temperature` defaults to 1.0); averaging cancels
     per-seed variance and tightens the predictive distribution.

  2. **Isotonic calibration.** Carve a validation slice out of the
     END of train fold (most recent slice — same walk-forward
     discipline as the train/holdout split), fit isotonic on the
     bag's mean prediction over that slice, then apply on holdout.
     The slice is large enough (~10% of train ≈ 12k rows) to learn
     a smooth monotone mapping, while preserving 90% of train for
     model fitting.

Reports per-stage Brier / log-loss / AUC plus a final ensemble +
calibrated combined number, so you can see what each layer adds.

Not a CLI; invoke once via `uv run python scripts/gbt_bag_and_calibrate.py`
after picking hyperparams from `gbt_hparam_search.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, UTC, timedelta
from pathlib import Path

import numpy as np
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
from skimsmarkets.tennis.gbt_train import TRAIN_CUTOFF, _auc, _brier, _log_loss


class _IsotonicCalibrator:
    """Pool-Adjacent-Violators isotonic regression.

    sklearn ships one but it's a heavy dep to pull in for ~30 lines
    of algorithm — this implementation is enough for our calibration
    layer. Fit on (predicted, observed) pairs, predict via
    np.interp on the monotone-non-decreasing reduced curve. Output
    is clipped to [0, 1] since predictions are probabilities.

    PAV walks left-to-right merging adjacent blocks whose means
    violate the monotone constraint (mean[i] > mean[i+1]); merging
    replaces both blocks with one whose weight is the sum of the
    contributing blocks' weights and whose value is the weighted
    mean. O(n) on the sorted input.
    """

    def __init__(self) -> None:
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_IsotonicCalibrator":
        order = np.argsort(x, kind="mergesort")
        xs = x[order].astype(float)
        ys = y[order].astype(float)
        # PAV with stack of (value, weight, start_x). On each new point,
        # push it as its own block; while the stack's top violates
        # monotonicity with the block before it, merge.
        stack: list[tuple[float, float, float]] = []
        for xi, yi in zip(xs, ys, strict=True):
            stack.append((yi, 1.0, xi))
            while len(stack) >= 2 and stack[-2][0] > stack[-1][0]:
                v1, w1, x1 = stack[-2]
                v2, w2, _ = stack[-1]
                merged_v = (v1 * w1 + v2 * w2) / (w1 + w2)
                stack.pop()
                stack.pop()
                stack.append((merged_v, w1 + w2, x1))
        # The reduced curve is one (x, y) per block, where x is the
        # block's first input value. To predict, interpolate; values
        # below the first x or above the last clamp to the boundary
        # value (np.interp's default behaviour).
        self._x = np.array([s[2] for s in stack])
        self._y = np.clip(np.array([s[0] for s in stack]), 0.0, 1.0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._x is None or self._y is None:
            raise RuntimeError("call fit() before predict()")
        return np.clip(np.interp(x, self._x, self._y), 0.0, 1.0)


log = logging.getLogger(__name__)


def _features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[list(ALL_FEATURE_COLUMNS)].copy()
    for col in NUMERIC_FEATURE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in CATEGORICAL_FEATURE_COLUMNS:
        out[col] = out[col].astype(str)
    return out


def _make_pool(df: pd.DataFrame, label: pd.Series | None = None) -> Pool:
    cat_indices = [list(ALL_FEATURE_COLUMNS).index(c) for c in CATEGORICAL_FEATURE_COLUMNS]
    return Pool(data=_features(df), label=label, cat_features=cat_indices)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--l2", type=float, default=4.0)
    ap.add_argument("--min-data-in-leaf", type=int, default=None)
    ap.add_argument("--n-seeds", type=int, default=5,
                    help="Number of bag models — averaged for the ensemble.")
    ap.add_argument("--val-days", type=int, default=90,
                    help="Days BEFORE TRAIN_CUTOFF carved into a validation "
                         "slice for isotonic calibration fit. 90d ~ 10% of "
                         "the train fold at current backfill density.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    matches_df = pd.read_parquet(RAW_MATCHES_PATH)
    profiles_df = pd.read_parquet(PLAYER_PROFILES_PATH)
    rankings_df = pd.read_parquet(RANKINGS_HISTORY_PATH)

    log.info("building training table")
    table = build_training_table(
        matches_df, profiles_df, rankings_df=rankings_df,
    )
    rows = table.rows
    rows["match_date"] = pd.to_datetime(rows["match_date"]).dt.date

    # Three-way split:
    #   train ≤ TRAIN_CUTOFF − val_days days  →  fit bag models
    #   train > cutoff−val_days AND ≤ cutoff   →  fit isotonic
    #   holdout > cutoff                        →  final evaluation
    val_cutoff = TRAIN_CUTOFF - timedelta(days=args.val_days)
    train_mask = rows["match_date"] <= val_cutoff
    val_mask = (rows["match_date"] > val_cutoff) & (rows["match_date"] <= TRAIN_CUTOFF)
    holdout_mask = rows["match_date"] > TRAIN_CUTOFF
    train_df = rows.loc[train_mask].reset_index(drop=True)
    val_df = rows.loc[val_mask].reset_index(drop=True)
    holdout_df = rows.loc[holdout_mask].reset_index(drop=True)

    log.info(
        "split: train=%d (≤ %s), val=%d (%s..%s), holdout=%d (> %s)",
        len(train_df), val_cutoff,
        len(val_df), val_cutoff, TRAIN_CUTOFF,
        len(holdout_df), TRAIN_CUTOFF,
    )

    train_pool = _make_pool(train_df, train_df["target"])
    val_pool = _make_pool(val_df, val_df["target"])
    holdout_pool = _make_pool(holdout_df, holdout_df["target"])
    y_val = val_df["target"].to_numpy()
    y_holdout = holdout_df["target"].to_numpy()

    # Stage 1: bag N models with different seeds.
    seeds = list(range(42, 42 + args.n_seeds))
    val_preds: list[np.ndarray] = []
    holdout_preds: list[np.ndarray] = []
    per_seed_metrics: list[dict] = []
    for seed in seeds:
        params = {
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "iterations": 3000,
            "learning_rate": args.lr,
            "depth": args.depth,
            "l2_leaf_reg": args.l2,
            "random_seed": seed,
            "use_best_model": True,
            "od_type": "Iter",
            "od_wait": 100,
            "verbose": False,
            "allow_writing_files": False,
        }
        if args.min_data_in_leaf is not None:
            params["min_data_in_leaf"] = args.min_data_in_leaf
        m = CatBoostClassifier(**params)
        # Eval-set for early-stopping = the VAL slice. Avoids the
        # "early-stopping on the same holdout we evaluate on" leak.
        m.fit(train_pool, eval_set=val_pool)
        p_v = m.predict_proba(val_pool)[:, 1]
        p_h = m.predict_proba(holdout_pool)[:, 1]
        val_preds.append(p_v)
        holdout_preds.append(p_h)
        per_seed_metrics.append({
            "seed": seed,
            "holdout_brier": _brier(y_holdout, p_h),
            "holdout_ll": _log_loss(y_holdout, p_h),
            "holdout_auc": _auc(y_holdout, p_h),
            "best_iter": m.get_best_iteration(),
        })
        log.info(
            "seed=%d  holdout brier=%.4f  ll=%.4f  auc=%.4f  iter=%d",
            seed, per_seed_metrics[-1]["holdout_brier"],
            per_seed_metrics[-1]["holdout_ll"],
            per_seed_metrics[-1]["holdout_auc"],
            per_seed_metrics[-1]["best_iter"],
        )

    # Stage 1 ensemble: mean prediction across seeds.
    bag_val = np.mean(val_preds, axis=0)
    bag_holdout = np.mean(holdout_preds, axis=0)
    log.info(
        "bag (n=%d)  holdout brier=%.4f  ll=%.4f  auc=%.4f",
        len(seeds), _brier(y_holdout, bag_holdout),
        _log_loss(y_holdout, bag_holdout), _auc(y_holdout, bag_holdout),
    )

    # Stage 2: isotonic on val, applied to holdout.
    iso = _IsotonicCalibrator()
    iso.fit(bag_val, y_val)
    calibrated_holdout = iso.predict(bag_holdout)
    log.info(
        "bag+isotonic  holdout brier=%.4f  ll=%.4f  auc=%.4f",
        _brier(y_holdout, calibrated_holdout),
        _log_loss(y_holdout, calibrated_holdout),
        _auc(y_holdout, calibrated_holdout),
    )

    # Save the full record. Caller decides whether to bake the
    # ensemble + isotonic into a production artifact.
    out = {
        "run_at_utc": datetime.now(UTC).isoformat(),
        "params": {
            "depth": args.depth, "lr": args.lr, "l2": args.l2,
            "min_data_in_leaf": args.min_data_in_leaf,
            "n_seeds": args.n_seeds, "val_days": args.val_days,
        },
        "val_cutoff": str(val_cutoff),
        "train_cutoff": str(TRAIN_CUTOFF),
        "n_train": len(train_df), "n_val": len(val_df), "n_holdout": len(holdout_df),
        "per_seed": per_seed_metrics,
        "bag": {
            "holdout_brier": _brier(y_holdout, bag_holdout),
            "holdout_ll": _log_loss(y_holdout, bag_holdout),
            "holdout_auc": _auc(y_holdout, bag_holdout),
        },
        "bag_plus_isotonic": {
            "holdout_brier": _brier(y_holdout, calibrated_holdout),
            "holdout_ll": _log_loss(y_holdout, calibrated_holdout),
            "holdout_auc": _auc(y_holdout, calibrated_holdout),
        },
    }
    Path("models/tennis_gbt_bag_calib.json").write_text(json.dumps(out, indent=2))
    log.info("wrote models/tennis_gbt_bag_calib.json")


if __name__ == "__main__":
    main()
