"""Director-side per-sport contingency menus.

The fetcher- and reasoner-side sport hints (`SPORT_HINTS`,
`REASONER_SPORT_HINTS`) used to live here too, keyed by `(lens, sport)`.
With per-sport lens sets (`agents/sports/<sport>/lens_set.py`), each
`LensSpec` now owns its own `fetcher_sport_hint` / `reasoner_sport_hint`
strings — there's only ONE valid sport per lens set, so the dict-of-tuples
collapsed to per-spec fields. Migration was straight: tennis content
moved into `agents/sports/tennis/lens_set.py`.

What survives here is `DIRECTOR_SPORT_HINTS` — sport-keyed (NOT
lens-keyed), applied to every event reaching the director regardless of
whether that sport has a lens set yet. Basketball / baseball / ufc / mma
have menus here even though they don't have bespoke lens sets on PR1
(events drop at lens_dispatch); the menus are kept ready for the day
their lens sets ship.

Same posture as before: rides on the per-event user message in
`director._render_user_message`, NEVER on the cached system block (which
would bust the slate-wide cache hit on `DIRECTOR_SHARED_PREAMBLE` and
the per-sport tail).
"""

from __future__ import annotations

from skimsmarkets.polymarket.models import PolymarketEvent

DIRECTOR_SPORT_HINTS: dict[str, str] = {
    "tennis": """
Tennis contingency menu — when sizing `confidence`, count how many of these
would have to align against your pick:
- Late withdrawal / walkover (most common single contingency in lower tiers).
- Mid-match retirement, especially in best-of-5 Slams or tour finals where
  fatigue surfaces.
- Outdoor weather: wind shifts ball flight on outdoor sessions; heat slows
  hard courts and dehydrates long rallies; rain delays let one player reset.
- Surface drying / roof open-vs-closed change between sets.
- Single-set blowup risk: a tight first set going to a bagel for the
  underdog often triggers a swing the pre-match priors didn't anticipate.
- Best-of-3 vs best-of-5 variance: best-of-3 has higher per-set leverage,
  best-of-5 dampens variance but amplifies fatigue/injury risk.
Calibration anchors:
- ATP/WTA top-100 vs unranked qualifier in R32, healthy, hard court → high
  (would need late withdrawal AND in-match collapse).
- Tour-level R64 between two ranked players within 30 spots, mid-tournament
  with no injury flags → medium (one bad set or a wind shift could flip it).
- Two players coming off back-to-back deciders, one with a fitness scare →
  low (a single mid-match retirement flips the pick).
""".strip(),

    "soccer": """
Soccer contingency menu — when sizing `confidence`, count how many of these
would have to align against your pick:
- Late predicted-XI change (key starter held back, GK rotation, surprise
  rest day).
- Red card in the first 30 minutes (resets the game; favored side often
  can't recover with 10 men).
- Set-piece variance: a single dead-ball goal can flip a tight match,
  especially in low-xG leagues / cup ties.
- Penalty awarded / VAR overturn in either direction.
- Weather: heavy rain on a high-press team's home pitch, wind on aerial
  duels, snow on a possession-passing side.
- Fixture congestion (UCL midweek into a weekend league fixture) driving
  rotation that pre-match line didn't price in.
Calibration anchors:
- Top-of-table side at home vs bottom-half visitor, no injury flags, no
  fixture-congestion overlap → high.
- Mid-table 3-way moneyline between sides separated by 10pts → medium
  (one early red card or a set-piece coin-flip resets it).
- Anyone playing the third match in 7 days vs a rested rival → low (a
  single rotation call or a 30th-minute red card flips it).
""".strip(),

    "basketball": """
Basketball contingency menu — when sizing `confidence`, count how many of
these would have to align against your pick:
- Late scratch (load management, rest day, illness reported at warmup).
- Foul trouble on a star — 2 fouls in Q1 typically removes them for most
  of the half and the underdog can run a backup-heavy lineup against them.
- Hot/cold shooting half from a role player: a 3-of-3 from beyond the arc
  by a 35% shooter is enough variance to flip a 5-pt spread game.
- Pace mismatch: a slow team forcing tempo on a fast team often produces
  results pre-game models don't see.
- Back-to-back fatigue, especially the second leg with travel.
- Garbage-time stat-padding for spreads (n/a for moneyline picks but flag
  the variance source if reasoning leans on a spread-style argument).
Calibration anchors:
- Top-3 conference team at home vs bottom-3, both healthy, neither on a
  back-to-back → high.
- Two playoff-tier teams in a regular-season meeting, one on rest →
  medium (foul trouble or one cold shooting half from the favored side
  flips it).
- Two evenly-matched teams, one starter scratched at warmup → low (the
  scratch alone is the single contingency that flips it).
""".strip(),

    "baseball": """
Baseball contingency menu — when sizing `confidence`, count how many of
these would have to align against your pick:
- Starting pitcher scratch (announced same-day, often missed by pre-game
  lines for hours).
- Weather: wind blowing in/out at hitter parks (Wrigley, Yankee Stadium)
  shifts run totals materially; rain delays reset bullpens.
- Bullpen depletion the day after long extras / a 7+ inning save situation.
- Umpire strike-zone skew (some umps run 2-3 inches above/below average
  on the high strike, swinging swing-rate-driven hitters).
- BABIP variance: a single softly-hit single in a tie game in the 8th
  flips a tightly-priced moneyline.
Calibration anchors:
- Top-5 starter (sub-3.00 ERA) at home vs a sub-.400 win-pct team's #4
  starter → high.
- Two division rivals with comparable starters, no bullpen overhang →
  medium (one mid-game pull or a wind shift flips it).
- Either side starting an opener / bullpen day, or coming off long
  extras → low (one pitching-decision contingency flips the pick).
""".strip(),

    "ufc": """
MMA contingency menu — when sizing `confidence`, count how many of these
would have to align against your pick:
- Weight-cut issues at weigh-in (missed weight, IV-rehydration ban era,
  visible drain on the scales).
- Late opponent change in the week of the fight.
- Judge-scoring variance on close decisions (especially in Nevada / NY
  where judging has historically diverged from MMA media scorecards).
- Cardio drop after round 2: heavy-handed strikers fade hard if the fight
  goes long; conversely grappling-only fighters can stall a striker into
  a decision they'd lose on the feet.
- Single-shot KO variance: even a 3-1 favorite can lose to one clean
  counter at the right angle.
Calibration anchors:
- Champion / top-5 ranked vs bottom-half-of-division opponent, both made
  weight cleanly, decisive style mismatch (e.g. wrestler vs no TDD) → high.
- Two top-15 fighters with reciprocal style threats, both made weight →
  medium (one early cardio fade or a coin-flip judging card flips it).
- Heavy striker favorite vs durable opponent with cardio in a 5-rounder
  → low (one round-3 fade flips the pick on the cards).
""".strip(),

    # `mma` is the gamma slug for non-UFC promotions (Bellator, ONE, PFL).
    # Same contingency menu — the bullets read identically.
    "mma": """
MMA contingency menu — when sizing `confidence`, count how many of these
would have to align against your pick:
- Weight-cut issues at weigh-in.
- Late opponent change in the week of the fight.
- Judge-scoring variance on close decisions.
- Cardio drop after round 2 / heavy-handed strikers fading.
- Single-shot KO variance: even a 3-1 favorite can lose to one clean
  counter at the right angle.
Calibration anchors mirror the UFC menu — apply by ranking and style
mismatch within the promotion's depth chart.
""".strip(),
}


def render_director_sport_hint(event: PolymarketEvent) -> str | None:
    """Return the director's sport-specific contingency menu, or `None` when
    no specialization applies.

    Same posture as before the per-sport-lens-set refactor: rides on the
    per-event user message in `director._render_user_message`, NEVER on
    the cached system block.
    """
    sport = event.sport_type
    if sport is None:
        return None
    body = DIRECTOR_SPORT_HINTS.get(sport)
    if body is None:
        return None
    return f"--- Contingency menu ({sport}) ---\n{body}"
