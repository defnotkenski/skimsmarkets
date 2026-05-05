"""Director-side per-sport contingency menus.

The fetcher- and reasoner-side sport hints used to live here too, keyed
by `(lens, sport)`. With per-sport lens sets
(`agents/sports/<sport>/lens_set.py`), each `LensSpec` now owns its own
`fetcher_sport_hint` / `reasoner_sport_hint` strings — there's only one
valid sport per lens set, so the dict-of-tuples collapsed to per-spec
fields.

What survives here is `DIRECTOR_SPORT_HINTS` — sport-keyed (NOT
lens-keyed), one entry per sport registered in `SPORT_LENS_SETS`. Strict
posture: events for unregistered sports drop at `lens_dispatch` before
the director, so menus only need to exist for sports that actually flow
through. When a new sport ships a lens set, add its menu here at the
same time.

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
