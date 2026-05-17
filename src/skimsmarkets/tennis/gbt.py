"""Slate-time GBT predictor.

Loads the trained catboost artefact + the historical match parquet
once, builds a `HistoryStore` representing the "as-of-latest-row"
snapshot, and exposes `predict_for_event` for the pipeline's
enrichment stage to call per-event.

Cold-start gate (≥ MIN_PRIORS_PER_SIDE per side) silently degrades
to no prediction. Same posture as the iid sim — the director still
sees the market and (possibly) the sim, and a missing third prior
isn't an error.

Anchor convention is critical: the model was trained anchor-relative
(anchor = lower MatchStat id, target = anchor won). The predict path
applies the same convention, then maps the anchor-relative
probability back to `P(team_a wins)` using the event's canonical
team_a name.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from threading import Lock
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier, Pool

from skimsmarkets.polymarket.models import PolymarketEvent
from skimsmarkets.tennis.gbt_backfill import (
    PLAYER_PROFILES_PATH,
    RAW_MATCHES_PATH,
)
from skimsmarkets.tennis.gbt_rankings_backfill import RANKINGS_HISTORY_PATH
from skimsmarkets.tennis.gbt_features import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURE_COLUMNS,
    MIN_PRIORS_PER_SIDE,
    NUMERIC_FEATURE_COLUMNS,
    HistoryStore,
    build_history_store,
    compute_features,
)
from skimsmarkets.tennis.gbt_train import MODEL_PATH
from skimsmarkets.tennis.models import (
    TennisGbtContext,
    TennisGbtFeatureContribution,
    TennisStatsContext,
)
from skimsmarkets.tennis.simulation import detect_best_of

log = logging.getLogger(__name__)

# Top-N features surfaced in the director-facing context. Tuned to
# match what fits in a one-line summary; catboost can return all 13,
# but only the largest few are actionable for the director.
_TOP_FEATURES_CAP = 5


class _ModelBundle:
    """Lazy-loaded model + history store + birthdate lookup.

    Module-level singleton accessed via `_get_bundle()`. The bundle
    is rebuilt when either the model artifact or the parquet's mtime
    advances — this lets a fresh `skims gbt train` run take effect
    without restarting the live pipeline.
    """

    def __init__(self) -> None:
        self.model: CatBoostClassifier | None = None
        self.history: HistoryStore | None = None
        # Birthdate lookup per (tour, player_id) — gbt_features needs
        # both to compute age_diff at slate time.
        self.birthdates: dict[tuple[str, int], date] = {}
        self.model_mtime: float | None = None
        self.parquet_mtime: float | None = None
        self.model_version: str | None = None

    def is_stale(self) -> bool:
        if self.model is None or self.history is None:
            return True
        if not MODEL_PATH.exists() or not RAW_MATCHES_PATH.exists():
            return True
        return (
            MODEL_PATH.stat().st_mtime != self.model_mtime
            or RAW_MATCHES_PATH.stat().st_mtime != self.parquet_mtime
        )

    def load(self) -> None:
        log.info(
            "loading GBT model + history store: model=%s, parquet=%s",
            MODEL_PATH, RAW_MATCHES_PATH,
        )
        self.model = CatBoostClassifier()
        self.model.load_model(str(MODEL_PATH))
        matches_df = pd.read_parquet(RAW_MATCHES_PATH)
        # Rankings is opt-in — when missing, schedule-strength feature
        # lands NaN at slate time and catboost reads as missing. Same
        # graceful-degrade discipline as the training path.
        rankings_df = (
            pd.read_parquet(RANKINGS_HISTORY_PATH)
            if RANKINGS_HISTORY_PATH.exists()
            else None
        )
        self.history = build_history_store(matches_df, rankings_df=rankings_df)
        self.model_mtime = MODEL_PATH.stat().st_mtime
        self.parquet_mtime = RAW_MATCHES_PATH.stat().st_mtime
        # Hash the artefact bytes so retro grading can detect retraining.
        # blake2b digest size 8 mirrors gbt_train._hash_artifact.
        import hashlib
        self.model_version = (
            f"sha-{hashlib.blake2b(MODEL_PATH.read_bytes(), digest_size=8).hexdigest()}"
        )
        # Birthdate lookup. Profile rows are authoritative when
        # birthdate is present; players without a profile fall back to
        # synthesized birthdate = first_appearance - 19y (matching the
        # training path in `build_training_table`). Without the synth
        # fallback, slate-time predictions for any player not in the
        # current top-50 land age-derived features as NaN, which
        # systematically degrades the age_diff, age_to_30_diff, and
        # age_x_elo_diff signal at predict time vs train time —
        # train/serve drift the model would silently suffer from.
        self.birthdates = {}
        if PLAYER_PROFILES_PATH.exists():
            profiles_df = pd.read_parquet(PLAYER_PROFILES_PATH)
            for _, row in profiles_df.iterrows():
                tour = row.get("tour")
                pid = row.get("player_id")
                bd = row.get("birthdate")
                if tour is None or pid is None or pd.isna(pid):
                    continue
                if isinstance(bd, pd.Timestamp) and not pd.isna(bd):
                    self.birthdates[(str(tour), int(pid))] = bd.date()
        # Synthesize for any unique (tour, player_id) in the match
        # parquet that lacks a real birthdate. 19y = the training-path
        # SYNTH_FIRST_APPEARANCE_AGE_YEARS — kept in lockstep so a
        # missing profile doesn't fork train-vs-serve behaviour.
        _SYNTH_FIRST_APPEARANCE_AGE_YEARS = 19
        if not matches_df.empty:
            first_match: dict[tuple[str, int], pd.Timestamp] = {}
            for _, row in matches_df.iterrows():
                md = pd.Timestamp(row["match_date"])
                tour_v = row.get("tour")
                if tour_v is None:
                    continue
                for pid_col in ("p1_id", "p2_id"):
                    pid_v = row.get(pid_col)
                    if pid_v is None or pd.isna(pid_v):
                        continue
                    key = (str(tour_v), int(pid_v))
                    if key not in first_match or md < first_match[key]:
                        first_match[key] = md
            for key, md in first_match.items():
                if key in self.birthdates:
                    continue
                synth_bd = (md - pd.DateOffset(
                    years=_SYNTH_FIRST_APPEARANCE_AGE_YEARS,
                )).date()
                self.birthdates[key] = synth_bd


_BUNDLE: _ModelBundle | None = None
_LOAD_LOCK = Lock()


def _get_bundle() -> _ModelBundle | None:
    """Return the lazily-loaded bundle, or None when artefacts are
    missing (no model trained yet, parquet not backfilled). The
    enricher treats None the same way as the cold-start gate firing
    — silent degrade, no error.
    """
    global _BUNDLE
    with _LOAD_LOCK:
        if _BUNDLE is None:
            _BUNDLE = _ModelBundle()
        if not MODEL_PATH.exists() or not RAW_MATCHES_PATH.exists():
            return None
        if _BUNDLE.is_stale():
            _BUNDLE.load()
        return _BUNDLE


def reload() -> None:
    """Force a reload on next predict. Test helper — also useful for
    operators who run `skims gbt train` and want the next live run
    to pick up the new artefact without a process restart.
    """
    global _BUNDLE
    with _LOAD_LOCK:
        _BUNDLE = None


def _resolve_player_id(stats: TennisStatsContext, side: str) -> int | None:
    """Pull the MatchStat int id off the player block. The model's
    `api_player_id` field is `str | None`; we convert here.
    """
    player = stats.player_a if side == "a" else stats.player_b
    pid = player.api_player_id
    if pid is None:
        return None
    try:
        return int(pid)
    except (TypeError, ValueError):
        return None


def _features_dataframe(features: dict[str, Any]) -> pd.DataFrame:
    """Single-row DataFrame in the column order catboost was trained on.

    The Pool wrapper requires the categorical columns be string-typed
    and the numeric columns be numeric — same coercion as the trainer.
    """
    df = pd.DataFrame([{c: features.get(c) for c in ALL_FEATURE_COLUMNS}])
    for col in NUMERIC_FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURE_COLUMNS:
        df[col] = df[col].astype(str)
    return df


def _top_feature_contributions(
    model: CatBoostClassifier, pool: Pool
) -> list[TennisGbtFeatureContribution]:
    """Per-row SHAP contributions for the predicted log-odds.

    catboost's `get_feature_importance(type="ShapValues", data=pool)`
    returns one row per example with a (n_features + 1) array — the
    last column is the expected value (bias). We strip the bias and
    sort by |contribution| descending.
    """
    shap_with_bias = model.get_feature_importance(
        data=pool, type="ShapValues"
    )
    shap = shap_with_bias[0, :-1]  # single-row pool → first row, drop bias
    pairs = sorted(
        zip(ALL_FEATURE_COLUMNS, shap, strict=True),
        key=lambda p: abs(p[1]),
        reverse=True,
    )
    return [
        TennisGbtFeatureContribution(name=name, contribution=float(value))
        for name, value in pairs[:_TOP_FEATURES_CAP]
    ]


def predict_for_event(event: PolymarketEvent) -> TennisGbtContext | None:
    """Compute the GBT prediction for one tennis event, or None when
    any pre-condition fails. Designed to mirror `simulate_for_event`
    in shape so the enricher in `pipeline.py` is symmetric.

    Pre-conditions, in order:
      1. The event has a populated `tennis_stats` block.
      2. Both player blocks carry an `api_player_id` we can int-parse.
      3. Both players appear in the historical store.
      4. Both players have ≥ MIN_PRIORS_PER_SIDE prior matches in the
         store (cold-start gate).
      5. The model bundle is loaded (artefacts present).

    On any failure, returns None — silent degrade. The director sees
    no `tennis_gbt` field on that event, exactly the way the sim
    falls back to None when its own gate fails.
    """
    stats = event.tennis_stats
    if stats is None:
        return None

    bundle = _get_bundle()
    if bundle is None or bundle.model is None or bundle.history is None:
        return None

    p_a_id = _resolve_player_id(stats, "a")
    p_b_id = _resolve_player_id(stats, "b")
    if p_a_id is None or p_b_id is None:
        return None

    h_a = bundle.history.get(p_a_id)
    h_b = bundle.history.get(p_b_id)
    if h_a is None or h_b is None:
        return None

    if h_a.matches < MIN_PRIORS_PER_SIDE or h_b.matches < MIN_PRIORS_PER_SIDE:
        return None

    on_date = datetime.now(UTC).date()
    surface = stats.surface
    best_of = detect_best_of(stats)
    # Tier from the most recent recent_match — same source the sim
    # uses for best_of detection. Conservative; defaults to "unknown"
    # and catboost handles unseen categorical levels gracefully.
    tier = _detect_tier(stats)

    a_birth = _lookup_birthdate(bundle, p_a_id)
    b_birth = _lookup_birthdate(bundle, p_b_id)
    cat_idx = [
        list(ALL_FEATURE_COLUMNS).index(c) for c in CATEGORICAL_FEATURE_COLUMNS
    ]

    # Predict both anchor orientations and average — guarantees a
    # symmetric prediction even when the trained model has residual
    # asymmetry (catboost trees can't be perfectly symmetric even with
    # augmented + sign-symmetric features). Without this step,
    # predict(A_anchor) + predict(B_anchor) drifts by up to ~0.07 on
    # 100 random pairs from the spike model. With it, the drift is
    # zero by construction.
    # Rank values come straight off the live `TennisStatsContext`
    # blocks the MatchStats provider populates at slate time. Live
    # ranks are current-week snapshots (not point-in-time historical
    # like the training path), but at prediction time "current rank"
    # IS the point-in-time value, so the semantics are identical.
    a_rank = stats.player_a.rank_singles
    b_rank = stats.player_b.rank_singles
    a_pts = stats.player_a.rank_points
    b_pts = stats.player_b.rank_points

    feats_a = compute_features(
        anchor_history=h_a, opponent_history=h_b,
        anchor_id=p_a_id, opponent_id=p_b_id,
        on_date=on_date, surface=surface, tier=tier, best_of=best_of,
        anchor_birthdate=a_birth, opponent_birthdate=b_birth,
        anchor_rank=a_rank, opponent_rank=b_rank,
        anchor_rank_points=a_pts, opponent_rank_points=b_pts,
    )
    feats_b = compute_features(
        anchor_history=h_b, opponent_history=h_a,
        anchor_id=p_b_id, opponent_id=p_a_id,
        on_date=on_date, surface=surface, tier=tier, best_of=best_of,
        anchor_birthdate=b_birth, opponent_birthdate=a_birth,
        anchor_rank=b_rank, opponent_rank=a_rank,
        anchor_rank_points=b_pts, opponent_rank_points=a_pts,
    )
    pool_a = Pool(data=_features_dataframe(feats_a), cat_features=cat_idx)
    pool_b = Pool(data=_features_dataframe(feats_b), cat_features=cat_idx)
    p_a_as_anchor = float(bundle.model.predict_proba(pool_a)[0, 1])
    p_b_as_anchor = float(bundle.model.predict_proba(pool_b)[0, 1])
    p_team_a = (p_a_as_anchor + (1.0 - p_b_as_anchor)) / 2.0

    # Top-feature contributions taken from the team_a-as-anchor pool
    # so the contribution sign is consistent with team_a's
    # perspective in the rendered block (positive = pushes toward
    # team_a winning).
    top = _top_feature_contributions(bundle.model, pool_a)

    return TennisGbtContext(
        provider="gbt_spike_v1",
        computed_at=datetime.now(UTC),
        model_version=bundle.model_version or "unknown",
        p_team_a_wins=p_team_a,
        n_prior_matches_a=int(h_a.matches),
        n_prior_matches_b=int(h_b.matches),
        top_features=top,
        assumptions=(
            "catboost on point-in-time aggregated career rates + surface "
            "splits + recent form + age + H2H; trained on top-50 ATP/WTA "
            "× ~2y history; cold-start gate requires ≥ "
            f"{MIN_PRIORS_PER_SIDE} prior matches per side"
        ),
    )


def _detect_tier(stats: TennisStatsContext) -> str | None:
    """Read tournament_tier off the most recent recent_match. Same
    pattern as `simulation.detect_best_of`. Falls back to None when
    neither player has populated recent_matches.
    """
    for player in (stats.player_a, stats.player_b):
        if player.recent_matches:
            tier = player.recent_matches[0].tournament_tier
            if tier:
                return tier
    return None


def _lookup_birthdate(bundle: _ModelBundle, player_id: int) -> date | None:
    """Try ATP first, then WTA. The parquet keys by (tour, player_id);
    a same-tour match guarantees one will resolve.
    """
    for tour in ("atp", "wta"):
        bd = bundle.birthdates.get((tour, player_id))
        if bd is not None:
            return bd
    return None


__all__ = ["predict_for_event", "reload"]
