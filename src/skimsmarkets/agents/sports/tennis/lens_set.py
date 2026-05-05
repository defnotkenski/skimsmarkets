"""Tennis lens set — three bespoke specs.

Each `LensSpec` carries:
- `fetcher_system_builder` — the prompt builder for the cached fetcher
  system block, parameterized by the provider's tool prose.
- `reasoner_system` — the cached system prompt for the Claude reasoner.
- `report_schema` — the Pydantic class the reasoner returns.
- `render_extras` — per-lens user-message append. ONLY
  `tennis_form_and_surface` reads the structured tennis_stats block;
  the matchup and conditions lenses see the same data only via the
  fetcher's notebook (they don't get the structured payload separately,
  to keep the lens silos honest).
- `fetcher_sport_hint` / `reasoner_sport_hint` — per-lens sport-specific
  guidance, ride on the user message (never the cached system block).

Why does only `tennis_form_and_surface` see the structured tennis_stats?
The block contains form/surface stats AND matchup-conditioned stats AND
some primitives the conditions lens cares about (last_match_date for
fatigue). Splitting it three ways would either duplicate the rendering
across all three lenses (token-cost overhead, breaks the silo by giving
every lens the same external structured payload) or require a
per-lens-tailored slice of the block (renderer complexity for marginal
gain). The tradeoff: form_and_surface gets the full block (it owns the
plurality of fields anyway); the other two lenses get those fields via
their own search using the same vendor's web pages, which is
substantively the same information through the fetcher's tool path.
This preserves the lens-silo posture from CLAUDE.md.
"""

from __future__ import annotations

from skimsmarkets.agents.sports.base import LensSet, LensSpec
from skimsmarkets.agents.sports.tennis.prompts import (
    DIRECTOR_SYSTEM_TENNIS_TAIL,
    TENNIS_CONDITIONS_AND_CONTEXT_REASONER_SYSTEM,
    TENNIS_FORM_AND_SURFACE_REASONER_SYSTEM,
    TENNIS_MATCHUP_AND_CLUTCH_REASONER_SYSTEM,
    tennis_conditions_and_context_notebook_system,
    tennis_form_and_surface_notebook_system,
    tennis_matchup_and_clutch_notebook_system,
)
from skimsmarkets.agents.sports.tennis.schemas import (
    TennisConditionsContextReport,
    TennisFormSurfaceReport,
    TennisMatchupClutchReport,
)
from skimsmarkets.polymarket.models import PolymarketEvent
from skimsmarkets.tennis import render_tennis_stats_block


def _render_tennis_stats_extras(event: PolymarketEvent) -> str | None:
    """Per-lens user-message append for `tennis_form_and_surface`.

    Returns the rendered structured tennis-stats block when present,
    `None` otherwise. The form_and_surface lens consumes the plurality
    of these fields (rankings, surface splits, recent matches, career
    serve/return %, tier records, career titles); piping the same block
    to the matchup and conditions lenses would breach the silo posture
    without buying material lens-specific data — those lenses pull
    their own primitives via the fetcher's web search.
    """
    if event.tennis_stats is None:
        return None
    return render_tennis_stats_block(event.tennis_stats)


# Per-lens fetcher sport-hint bodies. These migrate the relevant content
# from the legacy `SPORT_HINTS[("statistics", "tennis")]` /
# `("injury", "tennis")` / `("narrative", "tennis")` keys, re-attributed
# by lens job. Form/surface/serve metrics + Log5/Elo method notes →
# form_and_surface; H2H method notes + clutch + handedness → matchup;
# weather/court/medical-timeout/coaching → conditions.

_FETCHER_HINT_FORM_AND_SURFACE = """
Tennis form-and-surface specifics:
- Surface-conditioned form is the load-bearing primitive. ATP/WTA stats are
  surface-segregated for a reason — recent on-surface form (hard / clay /
  grass / carpet) outweighs YTD aggregates. Lift the on-surface W-L
  directly from `surface_win_loss[surface]` in the structured block.
- Tier records (`record_at_grand_slam`, `record_at_masters`,
  `record_vs_top_5`, `record_vs_top_10`) tell you whether form translates
  against quality competition at this stage. Use them when the sample is
  ≥10 matches.
- Career serve metrics (1st-serve in%, 1st/2nd-serve points won%) and
  career return metrics (1st/2nd-serve return won%) are CAREER aggregates
  in the structured block. A player on a hot serving stretch may be
  over-performing them right now; web-search recent results to flag this.
- Sources for color the structured block doesn't carry: tennisabstract.com
  (Elo + surface splits), ATP/WTA official, Infosys ATP stats,
  flashscore for recent results.
- code_execution: surface-conditioned Elo, recent-form-weighted baseline.
  Surface candidates labeled clearly (e.g. `team_a_baseline_elo_clay`,
  `team_a_baseline_log5_recent_8`) so the reasoner can pick the most
  defensible. Log5 is awkward for tennis — surface-conditioned win-rate
  is the cleaner baseline.
- DO NOT push for surface-effect MAGNITUDE in your `form_signed_shift` —
  that's `surface_signed_shift`'s job (also from this lens, but separate
  field). The form shift is about RECENT QUALITY (last 10 matches, loss
  color, training-block disruptions); the surface shift is about
  THIS-SURFACE FIT (career and YTD splits on this surface).
""".strip()


_FETCHER_HINT_MATCHUP_AND_CLUTCH = """
Tennis matchup-and-clutch specifics:
- H2H samples are small but predictive. The structured block ships
  total H2H + per-surface H2H + 3 most recent meetings (date, winner,
  surface, round, score). Cite both lifetime AND on-surface H2H, but
  remember: the form lens owns surface-effect magnitude, so your
  numeric `h2h_signed_shift` should NOT push for surface — surface H2H
  informs your reasoning qualitatively only.
- In-matchup clutch (deciders, tiebreaks, comeback %, closeout %, bo3
  vs bo5 records) is sharper signal than career-wide BP %. The
  structured block ships these per-player conditioned on THIS opponent —
  a player who's 67% in deciders overall but 33% in deciders against
  this specific opponent gives you a real edge.
- Handedness (`plays`) drives game-style fit: lefty crosscourt forehand
  attacks one-handed-backhand defenders. Big-server vs returner: a
  high-1st-serve-in% / low-return-pts-won% server vs a high-1st-serve-
  return-won% returner is a tightly contested matchup; tiebreaks
  decide it.
- Tournament tier shapes variance: Slam best-of-5 reduces upset rate
  (more chances for the higher-quality player to assert), but
  amplifies fatigue/injury risk if either player is marginal. Tour
  best-of-3 has more variance; tiebreak skill is more decisive.
- Sources: tennisabstract.com (H2H + matchup splits), ATP/WTA H2H
  pages, recent-meeting recaps in tennis press.
- code_execution: H2H-conditioned win-rate, decider-record fit
  (binomial confidence intervals when N is small), comeback-rate
  comparison.
""".strip()


_FETCHER_HINT_CONDITIONS_AND_CONTEXT = """
Tennis conditions-and-context specifics:
- Court conditions are quantitative form adjustments, not storylines.
  Heat slows hard / clay; wind on outdoor sessions hurts the bigger
  server; clay drying speed after rain shifts the pace; roof open vs
  closed at Wimbledon Centre / Rod Laver shifts ball flight; altitude
  at Madrid / Indian Wells inflates flat shots. Pull current/forecast
  conditions and surface their expected impact in `computed_numbers`
  (e.g. `wind_serve_drag_<player>`, `heat_hold_pct_adjust_<player>`).
- Tennis injuries are often hidden until match-time. Check for
  medical timeouts in recent matches, retirement frequency in the
  last 90 days, and withdrawals from preceding events on the same
  swing. Body-part / surface interaction shapes magnitude: shoulder /
  wrist issues spike serve breakdown (bigger on hard / grass); lower-
  body issues are bigger on clay (longer rallies, more sliding).
- Fatigue from prior round is real and quantifiable: sets/games/minutes
  played in this tournament so far, time since last match (long
  layoff = rust risk; back-to-back days = cumulative fatigue), travel
  since last event.
- Stakes and motivation: defending finalist points, race-to-Finals
  cutoff (October-November), defending the title, first-time
  finalist nerves, post-Slam letdown, end-of-season tank.
- Coaching changes mid-season often signal technical recalibration in
  progress — flag them in `stakes_summary` (NOT in the physical
  shift; coaching is technical disruption, not availability).
- Sources: ATP/WTA official notices, beat reporters (José Morgado,
  Christopher Clarey, Ben Rothenberg), player social media (Twitter /
  Instagram stories often leak warm-up issues), weather.com /
  Météo-France for outdoor-session forecasts, tennis press for stakes
  context.
- code_execution: tour-baseline retirement rate is ~3-5% of matches;
  flag players above baseline in the trailing 90d as withdrawal-
  risk-elevated.
""".strip()


# Per-lens reasoner sport-hint bodies. Calibration anchors specifically
# for the new tennis schemas' signed-shift bounds and confidence tiers.

_REASONER_HINT_FORM_AND_SURFACE = """
Tennis form-and-surface calibration:
- `team_a_win_probability` BASELINE: anchor on rankings + surface fit
  ALONE. Don't pre-bake H2H, conditions, or stakes — those stack on top
  via the other lenses' shifts. A 600-rank gap between two healthy
  players on a neutral surface is typically a 0.70-0.80 baseline; a
  100-rank gap is typically 0.55-0.62.
- `form_signed_shift` magnitudes (toward team_a):
    * ≥+0.10 / ≤-0.10: clear form divergence (e.g. one player 8-1 with
      titles in the last 6 weeks; the other 1-4 with two retirements).
    * +0.05 to +0.10: meaningful form delta (one player on a winning
      stretch, the other not).
    * 0 to ±0.05: comparable form, no decisive recency edge.
- `surface_signed_shift` magnitudes:
    * ≥+0.07 / ≤-0.07: dominant surface specialist vs weak surface
      record (e.g. clay-courter vs hard-courter on clay).
    * +0.03 to +0.07: clear preference asymmetry but not specialist-
      level (one player 65%+ on this surface, the other 50%).
    * 0 to ±0.03: both players comfortable on this surface.
- `confidence`:
    * 'high': structured stats block is rich, computed candidates
      converge within 5pp, multiple surface seasons of data on both.
    * 'medium': structured block present but candidates span 5-10pp,
      or one player is a surface debutant.
    * 'low': structured block missing OR `coverage='thin'` OR
      candidates span >10pp.
""".strip()


_REASONER_HINT_MATCHUP_AND_CLUTCH = """
Tennis matchup-and-clutch calibration:
- `h2h_signed_shift` magnitudes (toward team_a):
    * ≥+0.10 / ≤-0.10: dominant H2H (≥4 meetings, lopsided record,
      and clear style fit favoring one side, e.g. lefty vs OHB
      defender). REMINDER: this is NOT surface-driven — even a
      surface-asymmetric H2H pushes via the form lens's surface
      shift, not here.
    * +0.05 to +0.10: meaningful H2H edge (3+ meetings, modest skew
      with style fit).
    * 0 to ±0.05: thin H2H or balanced record; rely on style flag and
      qualitative tactical fit.
- `clutch_signed_shift` magnitudes:
    * ≥+0.06 / ≤-0.06: clear clutch divergence — strongly prefer
      matchup-conditioned numbers (in-matchup decider %, in-matchup
      tiebreak %) when present. A 33% vs 67% in-matchup decider
      record is a genuine ~0.06-0.08 shift.
    * +0.02 to +0.06: career BP-save / BP-convert percentages
      diverge by 5pp+; matchup-conditioned data sparse.
    * 0 to ±0.02: comparable clutch, no decisive divergence.
- `confidence`:
    * 'high': H2H ≥7 meetings AND matchup-conditioned clutch present
      AND directionally aligned with H2H.
    * 'medium': H2H 3-6 meetings OR matchup-conditioned data sparse.
    * 'low': first-time meeting OR H2H ≤2 meetings OR
      `coverage='thin'`.
""".strip()


_REASONER_HINT_CONDITIONS_AND_CONTEXT = """
Tennis conditions-and-context calibration:
- `physical_signed_shift` magnitudes (toward team_a, combining
  fitness + court conditions):
    * ±0.13 to ±0.15 (cap): confirmed pre-match withdrawal /
      walkover. The other side gets the full cap toward them.
    * ±0.06 to ±0.12: credible questionable tag with same-day
      warm-up issues; stacked fatigue + adverse weather for one side.
    * ±0.02 to ±0.06: fatigue from a back-to-back-day grind, or
      adverse weather (e.g. cold + wind for a big-server) without
      injury concerns.
    * 0 to ±0.02: comparable physical state, neutral conditions.
- Body-part / surface interaction: shoulder / wrist on hard / grass
  pushes one band stronger; lower-body on clay pushes one band
  stronger.
- Best-of-5 (Slams) amplifies marginal-injury risk vs best-of-3 — push
  one band stronger on a marginal injury at a Slam vs a 250.
- `stakes_signed_shift` magnitudes (toward team_a, MOTIVATION ONLY —
  not physical):
    * ≥+0.06 / ≤-0.06: clear divergence (one player defending huge
      Slam points, the other with nothing on the line; or one
      player's first Slam SF vs the other's 20th).
    * +0.02 to +0.06: modest motivation asymmetry (one player on a
      ranking-bubble, the other comfortable).
    * 0 to ±0.02: comparable stakes; both players motivated similarly.
- `lineup_confidence`:
    * 'confirmed' once both players are on entry list AND have
      practiced same-day.
    * 'probable' when entry list is final but warm-up issues
      reported on either side.
    * 'uncertain' otherwise.
- Coach changes are technical disruption, not availability — keep
  them OUT of the physical shift even when the notebook flags them.
  Note them in `stakes_summary` if relevant.
""".strip()


TENNIS_LENS_SET = LensSet(
    sport="tennis",
    lenses=(
        LensSpec(
            name="tennis_form_and_surface",
            fetcher_system_builder=tennis_form_and_surface_notebook_system,
            reasoner_system=TENNIS_FORM_AND_SURFACE_REASONER_SYSTEM,
            report_schema=TennisFormSurfaceReport,
            render_extras=_render_tennis_stats_extras,
            fetcher_sport_hint=_FETCHER_HINT_FORM_AND_SURFACE,
            reasoner_sport_hint=_REASONER_HINT_FORM_AND_SURFACE,
        ),
        LensSpec(
            name="tennis_matchup_and_clutch",
            fetcher_system_builder=tennis_matchup_and_clutch_notebook_system,
            reasoner_system=TENNIS_MATCHUP_AND_CLUTCH_REASONER_SYSTEM,
            report_schema=TennisMatchupClutchReport,
            fetcher_sport_hint=_FETCHER_HINT_MATCHUP_AND_CLUTCH,
            reasoner_sport_hint=_REASONER_HINT_MATCHUP_AND_CLUTCH,
        ),
        LensSpec(
            name="tennis_conditions_and_context",
            fetcher_system_builder=tennis_conditions_and_context_notebook_system,
            reasoner_system=TENNIS_CONDITIONS_AND_CONTEXT_REASONER_SYSTEM,
            report_schema=TennisConditionsContextReport,
            fetcher_sport_hint=_FETCHER_HINT_CONDITIONS_AND_CONTEXT,
            reasoner_sport_hint=_REASONER_HINT_CONDITIONS_AND_CONTEXT,
        ),
    ),
    director_system_tail=DIRECTOR_SYSTEM_TENNIS_TAIL,
)
