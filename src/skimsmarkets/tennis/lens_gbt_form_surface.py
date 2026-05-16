"""Per-lens GBT for the tennis form_and_surface lens.

Trains a CatBoostClassifier on a SUBSET of `ALL_FEATURE_COLUMNS` — the
columns the form_and_surface lens is conceptually responsible for
(form proxies + surface + form-adjacent statics like rank and age).
Career serve/return aggregates, H2H, and clutch are deliberately
excluded — those belong to other lenses.

Shares the feature pipeline (`build_training_table`), train/holdout
cutoff (`TRAIN_CUTOFF = 2024-12-31`), anchor convention (lower-id),
and catboost hyperparameters with `gbt_train.py` so the per-lens model
is directly comparable against `tennis_gbt_spike.cbm` on the same
holdout rows.

The whole point of per-lens specialisation is that this module trains
on a feature SUBSET — the lens is intentionally given less than the
full-feature GBT so its predictions live in the form-and-surface
signal subspace.

Why a separate module: keeps the lens-specific feature list pinned
and versioned alongside its trainer, so a future iteration that
broadens or narrows the subset is a single-file diff that doesn't
touch `gbt_train.py` or `gbt_features.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from skimsmarkets.tennis.gbt_features import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    build_training_table,
)
from skimsmarkets.tennis.gbt_train import TRAIN_CUTOFF, _CATBOOST_PARAMS

log = logging.getLogger(__name__)

# Feature subset for the form_and_surface lens.
#
# Numerics (7): the lens's signal vocabulary per the project memo —
#   - recent-form proxies: last_n_winrate_diff, days_since_diff
#   - biological/static skill proxy: age_diff
#   - surface fit: surface_first_serve_win_pct_diff, surface_record_diff
#   - "general skill" proxies (form-adjacent): rank_diff, rank_points_diff
#
# Deliberately EXCLUDED:
#   - career serve/return aggregates (matchup_and_clutch territory)
#   - H2H (matchup_and_clutch territory)
#   - clutch (matchup_and_clutch territory)
#
# Categoricals (3): same as full-feature GBT — surface is on-topic,
# tier and best_of are match-level context that any per-lens model
# needs to condition on.
LENS_FORM_SURFACE_NUMERIC_COLUMNS: tuple[str, ...] = (
    "last_n_winrate_diff",
    "days_since_diff",
    "age_diff",
    "surface_first_serve_win_pct_diff",
    "surface_record_diff",
    "rank_diff",
    "rank_points_diff",
)

LENS_FORM_SURFACE_CATEGORICAL_COLUMNS: tuple[str, ...] = CATEGORICAL_FEATURE_COLUMNS

LENS_FORM_SURFACE_ALL_COLUMNS: tuple[str, ...] = (
    LENS_FORM_SURFACE_NUMERIC_COLUMNS + LENS_FORM_SURFACE_CATEGORICAL_COLUMNS
)


@dataclass
class LensGbtTrainOutput:
    """Bundle returned from `train_lens_gbt`. Exposes the holdout
    DataFrame with predictions attached so the backtest harness can
    compute combined-prediction metrics + bucket-hit-rate without
    re-walking the parquet.
    """

    model: CatBoostClassifier
    train_df: pd.DataFrame
    holdout_df: pd.DataFrame
    train_p: np.ndarray
    holdout_p: np.ndarray
    feature_importance: dict[str, float]
    feature_columns: tuple[str, ...]


def _features_with_dtypes(
    df: pd.DataFrame, feature_columns: tuple[str, ...]
) -> pd.DataFrame:
    """Lens-agnostic dtype coercion. Iterates over the full numeric +
    categorical column lists from `gbt_features`, only touching columns
    that actually appear in `feature_columns`. Lets any lens-specific
    subset (form_surface, matchup_clutch, …) share this trainer
    without each subset duplicating the coercion logic.
    """
    out = df[list(feature_columns)].copy()
    for col in NUMERIC_FEATURE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in CATEGORICAL_FEATURE_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out


def _make_pool(
    df: pd.DataFrame,
    target: pd.Series | None,
    feature_columns: tuple[str, ...],
) -> Pool:
    X = _features_with_dtypes(df, feature_columns)
    # Categorical indices keyed off the FULL canonical list — every
    # `CATEGORICAL_FEATURE_COLUMNS` entry that appears in this lens's
    # subset gets its position passed to catboost as a cat_feature.
    cat_indices = [
        list(feature_columns).index(c)
        for c in CATEGORICAL_FEATURE_COLUMNS
        if c in feature_columns
    ]
    return Pool(data=X, label=target, cat_features=cat_indices)


def train_lens_gbt(
    matches_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    *,
    rankings_df: pd.DataFrame | None = None,
    train_cutoff: date = TRAIN_CUTOFF,
    feature_columns: tuple[str, ...] = LENS_FORM_SURFACE_ALL_COLUMNS,
    catboost_params: dict[str, Any] | None = None,
) -> LensGbtTrainOutput:
    """End-to-end: build features, walk-forward split, fit per-lens GBT.

    `feature_columns` defaults to the form_and_surface subset; pass an
    alternative tuple for ablation experiments. Must intersect
    `ALL_FEATURE_COLUMNS` from `gbt_features` — the trainer feeds
    whatever subset is given.

    `catboost_params` overrides the defaults from `gbt_train`. Mostly
    used to bump `verbose` down for quieter ablation runs.
    """
    table = build_training_table(matches_df, profiles_df, rankings_df=rankings_df)
    rows = table.rows
    if rows.empty:
        raise RuntimeError("training table is empty")

    rows["match_date"] = pd.to_datetime(rows["match_date"]).dt.date
    train_mask = rows["match_date"] <= train_cutoff
    train_df = rows.loc[train_mask].reset_index(drop=True)
    holdout_df = rows.loc[~train_mask].reset_index(drop=True)
    if train_df.empty or holdout_df.empty:
        raise RuntimeError(
            f"split produced empty fold (train={len(train_df)}, "
            f"holdout={len(holdout_df)})"
        )
    log.info(
        "split: train=%d (≤ %s), holdout=%d (> %s)",
        len(train_df), train_cutoff, len(holdout_df), train_cutoff,
    )

    train_pool = _make_pool(train_df, train_df["target"], feature_columns)
    holdout_pool = _make_pool(holdout_df, holdout_df["target"], feature_columns)

    params = dict(_CATBOOST_PARAMS)
    if catboost_params is not None:
        params.update(catboost_params)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=holdout_pool)

    train_p = model.predict_proba(train_pool)[:, 1]
    holdout_p = model.predict_proba(holdout_pool)[:, 1]

    importance = dict(zip(
        feature_columns,
        [float(x) for x in model.get_feature_importance()],
        strict=True,
    ))

    return LensGbtTrainOutput(
        model=model,
        train_df=train_df,
        holdout_df=holdout_df,
        train_p=train_p,
        holdout_p=holdout_p,
        feature_importance=importance,
        feature_columns=feature_columns,
    )


__all__ = [
    "LENS_FORM_SURFACE_ALL_COLUMNS",
    "LENS_FORM_SURFACE_CATEGORICAL_COLUMNS",
    "LENS_FORM_SURFACE_NUMERIC_COLUMNS",
    "LensGbtTrainOutput",
    "train_lens_gbt",
]
