"""Mine the closed-event dataset for patterns we can lean on for entry.

Each function takes the assembled DataFrame and returns a small text/table
report. None of these "fit a model" — they're empirical groupbys designed to
surface *where* the market is well-calibrated and *where* it isn't, plus
which signals (movement, volatility, liquidity) carry information.

Calibration metric: Brier score = mean((p - y)^2). Lower is better; 0.25 is
the score of always-50/50, 0.0 is perfect.
Log-loss (clipped) is reported alongside as a sanity check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-6


def brier(p: pd.Series, y: pd.Series) -> float:
    return float(((p - y) ** 2).mean())


def log_loss(p: pd.Series, y: pd.Series) -> float:
    pc = p.clip(EPS, 1 - EPS)
    return float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean())


@dataclass
class Section:
    title: str
    body: str

    def render(self) -> str:
        bar = "=" * len(self.title)
        return f"\n{self.title}\n{bar}\n{self.body}\n"


def _bucket_calibration(df: pd.DataFrame, p_col: str, label: str) -> str:
    """Bucket predictions into deciles of `p_col`, report realized rate."""
    sub = df[df[p_col].between(0, 1, inclusive="both")].copy()
    sub["bucket"] = pd.cut(
        sub[p_col],
        bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        include_lowest=True,
    )
    g = sub.groupby("bucket", observed=True).agg(
        n=("settled_yes", "size"),
        mean_p=(p_col, "mean"),
        win_rate=("settled_yes", "mean"),
    )
    g["edge"] = g["win_rate"] - g["mean_p"]
    g["edge_bps"] = (g["edge"] * 10000).round().astype("Int64")
    bs = brier(sub[p_col], sub["settled_yes"])
    ll = log_loss(sub[p_col], sub["settled_yes"])
    return (
        f"{label}: brier={bs:.4f} log_loss={ll:.4f} n={len(sub)}\n"
        + g.round(4).to_string()
    )


def calibration_overall(df: pd.DataFrame) -> Section:
    body = "\n\n".join(
        _bucket_calibration(df, col, name)
        for col, name in [
            ("open_p", "OPEN  (first observed mid)"),
            ("t24h_p", "T-24h"),
            ("t1h_p", "T-1h"),
            ("t1m_p", "T-1m  (closing line)"),
        ]
    )
    return Section("Calibration: market mid vs. realized outcome", body)


def calibration_by_league(df: pd.DataFrame) -> Section:
    rows = []
    for league, sub in df.groupby("league"):
        if len(sub) < 30:
            continue
        rows.append({
            "league": league,
            "n": len(sub),
            "brier_t1m": brier(sub.t1m_p, sub.settled_yes),
            "brier_open": brier(sub.open_p, sub.settled_yes),
            # delta_brier = how much the market sharpened from open → close.
            # positive = closing line is sharper than the open.
            "improve": brier(sub.open_p, sub.settled_yes) - brier(sub.t1m_p, sub.settled_yes),
        })
    g = pd.DataFrame(rows).sort_values("brier_t1m")
    return Section(
        "Calibration by league (Brier of closing mid; lower = sharper)",
        g.round(4).to_string(index=False),
    )


def movement_signal(df: pd.DataFrame) -> Section:
    """Does pre-match price drift predict the outcome?

    Sign of (t1m - t24h) — did the implied prob rise into kickoff? — bucketed
    against realized win rate, holding closing-line probability roughly fixed.
    """
    sub = df.dropna(subset=["movement_24h", "t1m_p"]).copy()
    # Hold opinion fixed by working in the "favorite" frame: only look at
    # outcomes where the closing line is between 0.2 and 0.8 (genuine
    # uncertainty — extreme favorites/longshots don't move much).
    mid = sub[sub.t1m_p.between(0.2, 0.8)].copy()
    mid["drift_bucket"] = pd.cut(
        mid["movement_24h"],
        bins=[-1, -0.05, -0.02, -0.005, 0.005, 0.02, 0.05, 1],
        labels=["<-5%", "-5/-2%", "-2/-0.5%", "flat", "+0.5/2%", "+2/5%", ">+5%"],
    )
    g = mid.groupby("drift_bucket", observed=True).agg(
        n=("settled_yes", "size"),
        mean_close=("t1m_p", "mean"),
        win_rate=("settled_yes", "mean"),
    )
    g["edge_vs_close"] = g["win_rate"] - g["mean_close"]
    g["edge_bps"] = (g["edge_vs_close"] * 10000).round().astype("Int64")
    return Section(
        "Late drift signal (24h → close, mids in 0.2–0.8 only)",
        g.round(4).to_string(),
    )


def threeway_overround(df: pd.DataFrame) -> Section:
    """How far is home+draw+away closing mid from 1.0? Per league."""
    pivot = df.groupby("event_slug").agg(
        league=("league", "first"),
        n_outcomes=("market_slug", "size"),
        sum_close=("t1m_p", "sum"),
        sum_open=("open_p", "sum"),
    )
    pivot = pivot[pivot.n_outcomes == 3]  # 3-way only
    by_league = pivot.groupby("league").agg(
        n=("sum_close", "size"),
        mean_sum_close=("sum_close", "mean"),
        mean_sum_open=("sum_open", "mean"),
    ).round(4)
    body = (
        f"3-way events: {len(pivot)}\n"
        f"overall mean(sum_close)={pivot.sum_close.mean():.4f} "
        f"median={pivot.sum_close.median():.4f}\n\n"
        + by_league.sort_values("mean_sum_close").to_string()
    )
    return Section("Three-way closing-mid overround (sum H+D+A)", body)


def liquidity_buckets(df: pd.DataFrame) -> Section:
    sub = df.dropna(subset=["volume_num", "t1m_p"]).copy()
    sub["liq_bucket"] = pd.qcut(
        sub.volume_num, q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
    )
    g = sub.groupby("liq_bucket", observed=True).agg(
        n=("settled_yes", "size"),
        mean_vol=("volume_num", "mean"),
        brier_t1m=("settled_yes", lambda y: brier(sub.loc[y.index, "t1m_p"], y)),
    )
    return Section(
        "Calibration by liquidity quartile (volumeNum)",
        g.round(4).to_string(),
    )


def vol_signal(df: pd.DataFrame) -> Section:
    """Does pre-match price volatility correlate with outcome predictability?"""
    sub = df.dropna(subset=["vol_24h", "t1m_p"]).copy()
    sub["vol_bucket"] = pd.qcut(sub.vol_24h, q=4, labels=["Q1 (calm)", "Q2", "Q3", "Q4 (vol)"])
    g = sub.groupby("vol_bucket", observed=True).agg(
        n=("settled_yes", "size"),
        mean_vol=("vol_24h", "mean"),
        mean_close=("t1m_p", "mean"),
        win_rate=("settled_yes", "mean"),
        brier_t1m=("settled_yes", lambda y: brier(sub.loc[y.index, "t1m_p"], y)),
    )
    return Section(
        "Calibration by 24h pre-match volatility quartile",
        g.round(4).to_string(),
    )


def favorite_longshot(df: pd.DataFrame) -> Section:
    """Classic favorite-longshot bias check on the closing line."""
    sub = df[df.t1m_p.between(0, 1)].copy()
    sub["band"] = pd.cut(
        sub.t1m_p,
        bins=[0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
    )
    g = sub.groupby("band", observed=True).agg(
        n=("settled_yes", "size"),
        mean_close=("t1m_p", "mean"),
        win_rate=("settled_yes", "mean"),
    )
    g["edge"] = g["win_rate"] - g["mean_close"]
    g["edge_bps"] = (g["edge"] * 10000).round().astype("Int64")
    return Section(
        "Favorite/longshot bias on closing mid",
        g.round(4).to_string(),
    )


def hypothetical_director_lift(df: pd.DataFrame) -> Section:
    """Sanity ceiling: if we had access to the *actual* settlement, what
    would the Brier-improvement-per-unit-confidence look like? Mostly used
    to quantify how much room there is to beat the closing line — i.e. is
    there headroom for the director to be useful at all?
    """
    sub = df.dropna(subset=["t1m_p"]).copy()
    base = brier(sub.t1m_p, sub.settled_yes)
    # Counterfactual: shrink the closing mid 20% toward the truth (assumes
    # an oracle telling us the right direction). This is NOT realistic — it
    # bounds the upside of any signal that nudges in the correct direction.
    nudged = sub.t1m_p + 0.2 * (sub.settled_yes - sub.t1m_p)
    perfect_dir = brier(nudged.clip(0, 1), sub.settled_yes)
    return Section(
        "Headroom check (NOT a strategy — bounds upside)",
        f"baseline brier (closing mid): {base:.4f}\n"
        f"oracle 20%-toward-truth nudge: {perfect_dir:.4f} "
        f"(reduction {(base-perfect_dir)/base*100:.1f}%)\n"
        f"=> any signal that nudges 20% in the right direction recovers "
        f"~{(base-perfect_dir)/base*100:.0f}% of brier. Room exists; whether "
        f"the director can do that is the open question.",
    )


def report(df: pd.DataFrame) -> str:
    sections = [
        calibration_overall(df),
        favorite_longshot(df),
        calibration_by_league(df),
        movement_signal(df),
        threeway_overround(df),
        liquidity_buckets(df),
        vol_signal(df),
        hypothetical_director_lift(df),
    ]
    header = (
        f"Backtest report — n_rows={len(df)} "
        f"events={df['event_slug'].nunique()} leagues={df['league'].nunique()}\n"
    )
    return header + "".join(s.render() for s in sections)
