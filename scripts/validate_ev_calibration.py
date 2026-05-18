"""EV-mode calibration validation — checks both calibration AND realized-EV
on settled predictions from `logs/runs/`.

Two complementary checks:

1. **Calibration check** — bins predictions by model confidence within the
   target market_p range, compares mean predicted probability vs actual
   win rate. Answers: "is the director's calibrated probability accurate
   in the range where this mode bets?"

   - For `--mode tail`: filters to market_p < 0.20 (deep underdog
     candidates). Concern: temperature scaling can't fix selective tail
     miscalibration, so a separate empirical check is required before
     trading tail-mode live.
   - For `--mode ev`: filters to market_p in [0.30, 0.65] (moderate-EV
     range where ev-mode picks concentrate). Temperature scaling SHOULD
     address this; if it doesn't, re-fit via `skims retro --step
     calibrate`.
   - For `--mode confidence`: filters to market_p ≥ 0.50 (favorite
     agreement range where confidence-mode picks concentrate).

2. **Realized-EV check** — for each settled prediction, computes the
   realized $-return per $1 staked at the persisted market_p (win =
   payoff_ratio, lose = -1), groups by ev_bucket, compares to the
   bucket's predicted ev_per_dollar. Answers: "if I'd actually traded
   these picks, would the realized EV match the math?"

   This is the most directly useful check for "did the EV system work?" —
   it validates the math end-to-end (model_p + market_p + outcome → $),
   not just probability calibration. Independent of mode (always runs).

Run from project root:

    uv run python scripts/validate_ev_calibration.py                     # default: ev mode
    uv run python scripts/validate_ev_calibration.py --mode tail
    uv run python scripts/validate_ev_calibration.py --mode confidence
    uv run python scripts/validate_ev_calibration.py --market-p-min 0.05 --market-p-max 0.15

Reports INCONCLUSIVE on insufficient data. The SCRIPT is reusable —
as `logs/runs/` accumulates, re-run periodically to track calibration
and realized-EV drift without re-deriving the analysis logic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

RUNS_DIR = Path("logs/runs")
DEFAULT_MIN_SAMPLES = 50  # below this, assessment is noise

# Per-mode market_p range presets. Each mode bets in a different region
# of the market_p distribution; calibration in the OTHER regions doesn't
# matter for that mode's P&L.
MODE_PRESETS: dict[str, tuple[float, float]] = {
    "tail": (0.0, 0.20),
    "ev": (0.30, 0.65),
    "confidence": (0.50, 1.0),
}


def _load_jsonl_rows(path: Path) -> list[dict]:
    """Load all JSON-decodable lines from a JSONL file. Skips malformed
    lines silently (matches the harness's defensive posture)."""
    out: list[dict] = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def load_settled_predictions(runs_dir: Path, sport: str | None = None) -> pd.DataFrame:
    """Join prediction rows with their resolution rows on event_id.

    Returns predictions where the resolution is settled (so
    `predicted_correct` is a meaningful win/loss label, not None).
    """
    preds: list[dict] = []
    for jl in sorted(runs_dir.glob("*.jsonl")):
        if jl.name.endswith(".resolutions.jsonl"):
            continue
        for r in _load_jsonl_rows(jl):
            if r.get("record_type") != "prediction":
                continue
            preds.append(r)
    if not preds:
        return pd.DataFrame()
    df_pred = pd.DataFrame(preds)

    resos: list[dict] = []
    for jl in sorted(runs_dir.glob("*.resolutions.jsonl")):
        resos.extend(_load_jsonl_rows(jl))
    if not resos:
        return pd.DataFrame()
    df_reso = pd.DataFrame(resos)

    df_reso = df_reso[df_reso.get("settled") == True]  # noqa: E712
    df_reso = df_reso[df_reso["predicted_correct"].notna()]

    df = df_pred.merge(
        df_reso[["event_id", "settled", "predicted_correct"]],
        on="event_id", how="inner",
    )
    if sport is not None and "sport_type" in df.columns:
        df = df[df["sport_type"] == sport]
    return df


def calibration_check(
    df: pd.DataFrame,
    *,
    market_p_min: float,
    market_p_max: float,
    n_bins: int,
    min_samples: int,
    label: str,
) -> int:
    """Per-bin predicted-probability vs realized-win-rate within the
    target market_p range. Returns exit-code contribution
    (0 = OK / INCONCLUSIVE, 1 = miscalibration warning)."""
    print(f"\n{'=' * 70}")
    print(f"CALIBRATION CHECK ({label}): market_p in [{market_p_min:.2f}, {market_p_max:.2f}]")
    print("=" * 70)

    needed = {"polymarket_implied_probability", "calibrated_winner_probability", "predicted_correct"}
    missing = needed - set(df.columns)
    if missing:
        print(f"ERROR: missing fields {sorted(missing)}")
        return 2

    df = df[df["polymarket_implied_probability"].notna()].copy()
    df = df[df["calibrated_winner_probability"].notna()]

    in_range = df[
        (df["polymarket_implied_probability"] >= market_p_min)
        & (df["polymarket_implied_probability"] < market_p_max)
    ].copy()
    print(f"Predictions in target market_p range: {len(in_range)}")

    if len(in_range) < min_samples:
        print(
            f"\nINCONCLUSIVE: only {len(in_range)} in-range predictions "
            f"(need >= {min_samples})."
        )
        return 0

    in_range["model_p"] = in_range["calibrated_winner_probability"].astype(float)
    in_range["actual"] = in_range["predicted_correct"].astype(int)

    try:
        in_range["bin"] = pd.qcut(in_range["model_p"], q=n_bins, duplicates="drop")
    except ValueError:
        in_range["bin"] = pd.cut(in_range["model_p"], bins=n_bins)

    print()
    print(f"{'bin (model_p range)':30s}  {'n':>4s}  {'mean_p':>7s}  {'actual':>7s}  {'diff':>7s}")
    print("-" * 70)
    for bin_, group in in_range.groupby("bin", observed=True):
        n = len(group)
        mean_p = group["model_p"].mean()
        actual = group["actual"].mean()
        diff = mean_p - actual
        print(f"{str(bin_):30s}  {n:>4d}  {mean_p:>7.3f}  {actual:>7.3f}  {diff:>+7.3f}")

    overall_p = in_range["model_p"].mean()
    overall_actual = in_range["actual"].mean()
    overall_diff = overall_p - overall_actual
    print()
    print(f"OVERALL: predicted={overall_p:.3f}  actual={overall_actual:.3f}  diff={overall_diff:+.3f}")

    if overall_diff > 0.05:
        print(
            f"\nWARNING: OVER-predicts by {overall_diff:+.3f} in this range. "
            f"Computed EV likely exceeds realized EV -> bets risk bleeding money."
        )
        return 1
    if overall_diff < -0.05:
        print(
            f"\nNOTE: UNDER-predicts by {abs(overall_diff):+.3f}. "
            f"Bets may have hidden upside vs computed EV."
        )
        return 0
    print(f"\nOK: calibration within +/- 0.05 (diff {overall_diff:+.3f}).")
    return 0


def realized_ev_check(df: pd.DataFrame, *, min_samples: int) -> int:
    """For each settled prediction, compute realized $-return per $1
    staked at the persisted market_p (win -> payoff_ratio, lose -> -1),
    group by ev_bucket, compare to mean predicted ev_per_dollar.

    Mode-agnostic — always runs because it validates the EV math
    end-to-end regardless of which selector mode produced the picks."""
    print(f"\n{'=' * 70}")
    print("REALIZED EV CHECK (mode-agnostic): per-bucket realized vs predicted $-return")
    print("=" * 70)

    needed = {"polymarket_implied_probability", "predicted_correct", "ev_bucket", "ev_per_dollar"}
    missing = needed - set(df.columns)
    if missing:
        print(
            f"SKIPPED: missing field(s) {sorted(missing)} — ev_bucket / "
            f"ev_per_dollar likely not persisted in older runs."
        )
        return 0

    df = df[df["ev_per_dollar"].notna()].copy()
    df = df[df["polymarket_implied_probability"].notna()]
    df = df[df["ev_bucket"].notna()]

    if df.empty:
        print("SKIPPED: no settled predictions with EV labels.")
        return 0

    df["market_p"] = df["polymarket_implied_probability"].astype(float)
    df["actual"] = df["predicted_correct"].astype(int)
    # Realized return at the persisted market_p (the predicted-winner-
    # frame market_p — same frame ev_per_dollar was computed against).
    # Win: payoff_ratio = (1 - market_p) / market_p. Lose: -1.
    df["payoff_ratio"] = (1.0 - df["market_p"]) / df["market_p"]
    df["realized_return"] = df.apply(
        lambda r: r["payoff_ratio"] if r["actual"] == 1 else -1.0, axis=1,
    )

    print()
    print(f"{'ev_bucket':12s}  {'n':>4s}  {'mean_ev':>9s}  {'realized':>9s}  {'diff':>9s}")
    print("-" * 60)
    bucket_order = ["Prime", "Edge", "Thin", "Negative", "Unrated"]
    seen = set()
    total_n = 0
    total_pred_ev = 0.0
    total_realized = 0.0
    for bucket in bucket_order:
        group = df[df["ev_bucket"] == bucket]
        if group.empty:
            continue
        seen.add(bucket)
        n = len(group)
        mean_pred = group["ev_per_dollar"].mean()
        mean_real = group["realized_return"].mean()
        diff = mean_pred - mean_real
        total_n += n
        total_pred_ev += mean_pred * n
        total_realized += mean_real * n
        print(f"{bucket:12s}  {n:>4d}  ${mean_pred:>+7.3f}  ${mean_real:>+7.3f}  ${diff:>+7.3f}")

    other = df[~df["ev_bucket"].isin(seen)]
    if not other.empty:
        for bucket, group in other.groupby("ev_bucket"):
            n = len(group)
            mean_pred = group["ev_per_dollar"].mean()
            mean_real = group["realized_return"].mean()
            total_n += n
            total_pred_ev += mean_pred * n
            total_realized += mean_real * n
            print(f"{str(bucket):12s}  {n:>4d}  ${mean_pred:>+7.3f}  ${mean_real:>+7.3f}  ${mean_pred - mean_real:>+7.3f}")

    if total_n == 0:
        print("(no rows)")
        return 0

    overall_pred = total_pred_ev / total_n
    overall_real = total_realized / total_n
    overall_diff = overall_pred - overall_real
    print(f"{'OVERALL':12s}  {total_n:>4d}  ${overall_pred:>+7.3f}  ${overall_real:>+7.3f}  ${overall_diff:>+7.3f}")

    print()
    if total_n < min_samples:
        print(f"INCONCLUSIVE: only {total_n} EV-labeled rows (need >= {min_samples}).")
        return 0

    if overall_diff > 0.05:
        print(
            f"WARNING: predicted EV exceeds realized by ${overall_diff:+.3f}/$1. "
            f"EV math is over-optimistic — possible miscalibration upstream."
        )
        return 1
    if overall_diff < -0.05:
        print(
            f"NOTE: realized EV exceeds predicted by ${abs(overall_diff):+.3f}/$1. "
            f"EV math is under-optimistic (rare; could indicate market_p drift "
            f"between rank-time and settlement)."
        )
        return 0
    print(f"OK: realized vs predicted EV within +/- $0.05 (diff ${overall_diff:+.3f}/$1).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--mode",
        choices=tuple(MODE_PRESETS.keys()),
        default="ev",
        help=(
            "Selector mode the calibration check should target. Sets the "
            f"market_p range filter: tail={MODE_PRESETS['tail']}, "
            f"ev={MODE_PRESETS['ev']}, confidence={MODE_PRESETS['confidence']}. "
            "Default: ev."
        ),
    )
    ap.add_argument(
        "--market-p-min", type=float, default=None,
        help="Override mode preset's lower bound for the market_p filter.",
    )
    ap.add_argument(
        "--market-p-max", type=float, default=None,
        help="Override mode preset's upper bound for the market_p filter.",
    )
    ap.add_argument("--sport", default="tennis", help="Sport filter (default: tennis)")
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument(
        "--min-samples", type=int, default=DEFAULT_MIN_SAMPLES,
        help=f"Min in-range count for a stable verdict (default: {DEFAULT_MIN_SAMPLES}).",
    )
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument(
        "--skip-realized-ev", action="store_true",
        help="Skip the realized-EV check (calibration check only).",
    )
    args = ap.parse_args()

    p_min, p_max = MODE_PRESETS[args.mode]
    if args.market_p_min is not None:
        p_min = args.market_p_min
    if args.market_p_max is not None:
        p_max = args.market_p_max

    df = load_settled_predictions(args.runs_dir, sport=args.sport)
    print(f"Total settled {args.sport} predictions: {len(df)}")
    if df.empty:
        print(f"\nINCONCLUSIVE: no settled predictions in {args.runs_dir}/.")
        return 0

    exit_code = calibration_check(
        df,
        market_p_min=p_min,
        market_p_max=p_max,
        n_bins=args.n_bins,
        min_samples=args.min_samples,
        label=args.mode,
    )
    if not args.skip_realized_ev:
        ev_exit = realized_ev_check(df, min_samples=args.min_samples)
        exit_code = max(exit_code, ev_exit)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
