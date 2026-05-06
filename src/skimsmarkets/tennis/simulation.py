"""Tennis match Monte Carlo simulator — career-baseline iid prior.

Pure-numpy point-by-point sim that takes a `TennisStatsContext`
(player career serve/return percentages already populated by the
upstream MatchStat enrichment) and returns a `TennisSimulationContext`
with `P(team_a wins)`, a 95% sampling-uncertainty CI, and per-side
point-win probabilities.

The sim is a SECOND deterministic prior for the director (alongside
Polymarket bid/ask). Intentionally limited to iid + career-baseline
so it doesn't fight the lenses' jobs:
- Surface effect → owned by `surface_signed_shift` on the form lens
- Recent form → owned by `form_signed_shift` on the form lens
- H2H + style fit → owned by `h2h_signed_shift` on the matchup lens
- Conditions / fatigue / stakes → owned by the conditions lens

The sim's value is being a long-run statistical baseline the director
can sanity-check the synthesized read against. When the director's
final probability deviates materially from BOTH the market AND the
sim, that's a high-conviction call. When it agrees with one but not
the other, it's a calibration check.

Math:
- Per-point (server's win-rate): symmetric average of server's career
  win-rate and 1-minus-returner's career return-win-rate, conditioned
  on first vs second serve via server's first-serve-in %.
- Per-game: closed-form probability given per-point win prob.
- Per-set: simulate game-by-game with alternating servers; tiebreak
  at 6-6 (first to 7, win by 2; ATP rotation: server 1 serves point 1,
  then alternates every 2 points).
- Per-match: bo3 = first to 2 sets; bo5 = first to 3 sets.
- 95% CI via Wilson interval (better than Wald for extreme p, and the
  director will see both extremes when the underlying skill gap is
  large enough).

Performance: 10k-sim default takes ~50-100ms per match in pure Python
loops — fast enough that we don't bother with vectorization. If a
slate ever pushes 100+ tennis events, revisit.
"""

from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime

import numpy as np

from skimsmarkets.tennis.models import (
    TennisPlayerStats,
    TennisSimulationContext,
    TennisStatsContext,
)


# Wilson 95% z-value. Hard-coded since we don't need configurability.
_WILSON_Z = 1.96
# Default sim count. 10k gives ~±1pp Wald precision around p=0.5,
# sub-second runtime. Lift only with evidence.
_DEFAULT_N_SIMS = 10_000


def detect_best_of(stats: TennisStatsContext) -> int:
    """Read tournament tier from the most recent recent_matches row on
    either player; `Grand Slam` → 5, anything else → 3.

    Falls back to bo3 when neither player has populated recent matches
    (the conservative default — most ATP/WTA events are bo3, and a
    bo3-simulated Slam under-predicts the favorite by ~3-5pp, which is
    less wrong than a bo5-simulated Masters event over-predicting by
    the same).
    """
    for player in (stats.player_a, stats.player_b):
        recent = player.recent_matches
        if not recent:
            continue
        # `recent_matches` is newest-first per the MatchStat adapter
        # (`/player/past-matches` reverse-chronological). Read tier off
        # the first row; older rows may be from a different tier.
        tier = recent[0].tournament_tier
        if tier and "grand slam" in tier.lower():
            return 5
    return 3


def _seed_from_slug(slug: str) -> int:
    """Stable 32-bit seed derived from the event slug. Same slug →
    same seed → reproducible sim across re-runs and JSONL re-renders.
    """
    h = hashlib.blake2b(slug.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(h, "big")


def _point_win_prob(
    server_first_in: float,
    server_first_win: float,
    server_second_win: float,
    returner_first_return_win: float,
    returner_second_return_win: float,
) -> float:
    """Probability the server wins a point.

    Symmetric-average form: blend server's career win-rate with
    1-minus-returner's career return-win-rate. Done separately for
    first-serve-in points (using first-serve metrics) and second-serve
    points (using second-serve metrics), weighted by the server's
    career first-serve-in %.

    `second_serve_win_pct` is computed by MatchStat across all
    second-serve points INCLUDING double faults, so this is a faithful
    aggregate without explicit fault modeling.
    """
    p_first_in_win = (server_first_win + (1 - returner_first_return_win)) / 2
    p_second_in_win = (server_second_win + (1 - returner_second_return_win)) / 2
    return server_first_in * p_first_in_win + (1 - server_first_in) * p_second_in_win


def _game_prob(p: float) -> float:
    """Closed-form probability of winning a game given per-point win
    prob `p`. Tennis game = first to 4 points, win by 2 (deuce).

    Path decomposition:
    - 4-0: p^4
    - 4-1: 4 * p^4 * q (lose 1 of first 4, win 5th)
    - 4-2: 10 * p^4 * q^2 (lose 2 of first 5, win 6th)
    - Reach 3-3 deuce: 20 * p^3 * q^3
    - Win from deuce: p^2 / (p^2 + q^2) (geometric)
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    q = 1 - p
    win_straight = (p**4) * (1 + 4 * q + 10 * q * q)
    p_deuce = 20 * (p**3) * (q**3)
    win_from_deuce = (p * p) / (p * p + q * q)
    return win_straight + p_deuce * win_from_deuce


def _simulate_tiebreak(
    p_a_pt_on_a_serve: float,
    p_a_pt_on_b_serve: float,
    rng: np.random.Generator,
    a_serves_first: bool,
) -> bool:
    """Standard ATP/WTA tiebreak — first to 7, win by 2.

    Service rotation: first server serves point 1; thereafter the
    serve alternates every 2 points (so server-2 takes points 2&3,
    server-1 takes 4&5, etc.).
    """
    a_pts = 0
    b_pts = 0
    pt = 0
    while True:
        # Determine current server.
        if pt == 0:
            a_serves = a_serves_first
        else:
            # After point 1, switch every 2 points. `pair_idx` cycles
            # 0,1,0,1,... starting at point 1; pair_idx==0 → server 2,
            # pair_idx==1 → server 1.
            pair_idx = ((pt - 1) // 2) % 2
            a_serves = a_serves_first if pair_idx == 1 else not a_serves_first
        prob = p_a_pt_on_a_serve if a_serves else p_a_pt_on_b_serve
        if rng.random() < prob:
            a_pts += 1
        else:
            b_pts += 1
        if a_pts >= 7 and a_pts - b_pts >= 2:
            return True
        if b_pts >= 7 and b_pts - a_pts >= 2:
            return False
        pt += 1


def _simulate_set(
    p_a_game_a_serves: float,
    p_a_game_b_serves: float,
    p_a_pt_on_a_serve: float,
    p_a_pt_on_b_serve: float,
    rng: np.random.Generator,
    a_serves_first: bool,
) -> bool:
    """Return True iff A wins this set. Games alternate servers; first
    to 6 with 2-game margin wins, otherwise tiebreak at 6-6.
    """
    a_games = 0
    b_games = 0
    a_serves = a_serves_first
    while True:
        prob = p_a_game_a_serves if a_serves else p_a_game_b_serves
        if rng.random() < prob:
            a_games += 1
        else:
            b_games += 1
        if a_games >= 6 and a_games - b_games >= 2:
            return True
        if b_games >= 6 and b_games - a_games >= 2:
            return False
        if a_games == 6 and b_games == 6:
            # Tiebreak: whoever was due to serve the next game serves
            # point 1 of the tiebreak.
            tb_a_serves_first = not a_serves
            return _simulate_tiebreak(
                p_a_pt_on_a_serve, p_a_pt_on_b_serve, rng, tb_a_serves_first
            )
        a_serves = not a_serves


def _gate_required_fields(p: TennisPlayerStats) -> bool:
    """Both players must have ALL FIVE of these populated for the sim
    to compute a meaningful number. Missing any one → no sim."""
    return all(
        v is not None
        for v in (
            p.first_serve_in_pct,
            p.first_serve_win_pct,
            p.second_serve_win_pct,
            p.first_serve_return_win_pct,
            p.second_serve_return_win_pct,
        )
    )


def simulate_match(
    stats: TennisStatsContext,
    *,
    best_of: int,
    n_sims: int = _DEFAULT_N_SIMS,
    seed: int | None = None,
) -> TennisSimulationContext | None:
    """Run a Monte Carlo sim of the match. Returns None when the
    attachment gate fails (either player missing any required career
    serve/return percentage).

    `best_of` must be 3 or 5; callers detect from
    `TennisRecentMatch.tournament_tier` via `detect_best_of`.

    `seed` is None → fresh RNG; pass a stable hash (e.g. of the event
    slug) for reproducible sims across re-runs.
    """
    if best_of not in (3, 5):
        raise ValueError(f"best_of must be 3 or 5, got {best_of}")

    a, b = stats.player_a, stats.player_b
    if not (_gate_required_fields(a) and _gate_required_fields(b)):
        return None

    # Server-side point-win probability for each player on their own
    # serve. The mypy/pylance reading sees these as Optional[float]
    # because TennisPlayerStats fields are Optional, but the gate
    # above guarantees non-None — so the cast is defensible. Use
    # `assert` for runtime safety in the rare case the gate logic
    # drifts from the field set.
    assert a.first_serve_in_pct is not None
    assert a.first_serve_win_pct is not None
    assert a.second_serve_win_pct is not None
    assert a.first_serve_return_win_pct is not None
    assert a.second_serve_return_win_pct is not None
    assert b.first_serve_in_pct is not None
    assert b.first_serve_win_pct is not None
    assert b.second_serve_win_pct is not None
    assert b.first_serve_return_win_pct is not None
    assert b.second_serve_return_win_pct is not None

    p_a_serve = _point_win_prob(
        a.first_serve_in_pct,
        a.first_serve_win_pct,
        a.second_serve_win_pct,
        b.first_serve_return_win_pct,
        b.second_serve_return_win_pct,
    )
    p_b_serve = _point_win_prob(
        b.first_serve_in_pct,
        b.first_serve_win_pct,
        b.second_serve_win_pct,
        a.first_serve_return_win_pct,
        a.second_serve_return_win_pct,
    )
    # Convert "B wins point on B's serve" to "A wins point on B's serve"
    # so all probabilities below are consistently from A's perspective.
    p_a_pt_on_a_serve = p_a_serve
    p_a_pt_on_b_serve = 1.0 - p_b_serve

    p_a_game_a_serves = _game_prob(p_a_pt_on_a_serve)
    p_a_game_b_serves = _game_prob(p_a_pt_on_b_serve)

    sets_to_win = (best_of + 1) // 2  # 2 for bo3, 3 for bo5
    rng = np.random.default_rng(seed)
    a_match_wins = 0
    for _ in range(n_sims):
        a_sets = 0
        b_sets = 0
        # A serves first in set 1 by convention; subsequent sets
        # continue the alternating-game pattern from where the
        # previous set ended. For an iid model this distinction
        # doesn't matter — set outcomes are independent given the
        # per-game probabilities — so we just reset to A-serves-first
        # each set. The bias is symmetric and washes out.
        while a_sets < sets_to_win and b_sets < sets_to_win:
            if _simulate_set(
                p_a_game_a_serves,
                p_a_game_b_serves,
                p_a_pt_on_a_serve,
                p_a_pt_on_b_serve,
                rng,
                a_serves_first=True,
            ):
                a_sets += 1
            else:
                b_sets += 1
        if a_sets == sets_to_win:
            a_match_wins += 1

    p_a = a_match_wins / n_sims

    # Wilson 95% interval. Better than Wald near 0/1 because we'll
    # legitimately see extreme p values when career-skill gaps are
    # large (e.g. top-5 vs unranked qualifier).
    z = _WILSON_Z
    z2 = z * z
    denom = 1 + z2 / n_sims
    centre = (p_a + z2 / (2 * n_sims)) / denom
    half = (
        z
        * math.sqrt(p_a * (1 - p_a) / n_sims + z2 / (4 * n_sims * n_sims))
        / denom
    )
    ci_low = max(0.0, centre - half)
    ci_high = min(1.0, centre + half)

    return TennisSimulationContext(
        provider="monte_carlo_v1",
        computed_at=datetime.now(UTC),
        p_team_a_wins=p_a,
        ci_low=ci_low,
        ci_high=ci_high,
        n_sims=n_sims,
        best_of=best_of,
        point_win_pct_a_serving=p_a_pt_on_a_serve,
        point_win_pct_b_serving=p_b_serve,
        assumptions=(
            "iid points, career-baseline serve/return %; "
            "no surface/form/H2H/conditions adjustment"
        ),
    )


def simulate_for_event(
    stats: TennisStatsContext,
    *,
    slug: str,
) -> TennisSimulationContext | None:
    """Convenience wrapper used by the pipeline enrichment stage.

    Detects bo3 vs bo5 from `TennisRecentMatch.tournament_tier`, derives
    a stable per-event seed from the slug, and runs the sim. Returns
    None on the attachment gate failure (caller leaves
    `event.tennis_simulation` as None and emits no error).
    """
    best_of = detect_best_of(stats)
    seed = _seed_from_slug(slug)
    return simulate_match(stats, best_of=best_of, seed=seed)


__all__ = [
    "detect_best_of",
    "simulate_for_event",
    "simulate_match",
]
