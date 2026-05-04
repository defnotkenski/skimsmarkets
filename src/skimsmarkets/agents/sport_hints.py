"""Sport-specific guidance injected into per-event user messages.

The system prompts in `prompts.py` are sport-agnostic and stay cached
(`CacheControlEphemeralParam` requires fixed strings). Sport-specific
guidance lives here and rides on the per-event user message, so the cache
hit on the system block is preserved.

Two parallel dicts, both keyed by (lens, sport):
- `SPORT_HINTS` shapes the fetcher's search loop ("what to look up, which
  sources, which code_execution recipes"). Consumed by `render_sport_hint`
  and appended to the fetcher's user message.
- `REASONER_SPORT_HINTS` shapes the reasoner's calibration ("how to size
  the signed shift / probability now that the facts are in the notebook").
  Consumed by `render_reasoner_sport_hint` and appended to the reasoner's
  user message. Currently populated for the injury lens only — the key
  shape leaves room for statistics / narrative reasoner specializations
  later without an API change.

Scope is intentionally narrow:
- Tennis and soccer only. These two have the most distinctive search patterns
  (surface splits + serve stats; xG + 3-way pricing + predicted XI) AND the
  most distinctive injury-impact regimes (single-player tennis withdrawals;
  soccer predicted-XI position granularity). Other sports fall through to
  the generic prompt.
- Detection via `event.sport_type` (gamma-tag-derived). Other sources like
  `series_slug` are not consulted — for events where gamma omits the sport
  tag (typically seasonal futures), we don't rank them anyway.
"""

from __future__ import annotations

from skimsmarkets.agents.schemas import LensName
from skimsmarkets.polymarket.models import PolymarketEvent

# Sport keys MUST match values populated by `_gamma_sport_from_tags` in
# `polymarket/models.py` exactly — that's the contract on `event.sport_type`.
SPORT_HINTS: dict[tuple[LensName, str], str] = {
    # ---------------- TENNIS ----------------
    ("statistics", "tennis"): """
Tennis-specific focus:
- Pull serve metrics (1st-serve %, points won on 1st/2nd serve, hold %), return
  metrics (return-points-won %, break-point conversion), and surface-conditioned
  splits. ATP/WTA stats are surface-segregated for a reason — recent on-surface
  form (hard / clay / grass / carpet) outweighs YTD aggregates.
- Tournament tier shapes variance: Slam best-of-5 reduces upset rate; tour
  best-of-3 has more variance. Note round (R64 / R32 / QF / SF / F) — top
  players sometimes play down to the level in early Masters rounds.
- H2H samples are small but predictive. Cite both lifetime AND on-surface H2H.
- Court conditions and weather are quantitative form adjustments, not
  storylines: heat (slower courts, dehydrating long rallies), wind on outdoor
  sessions (US Open day, Roland Garros), clay drying speed after rain, roof
  open/closed shifts ball flight. Pull current/forecast conditions and
  surface their expected impact in `computed_numbers` (e.g.
  `wind_serve_drag_<player>`, `heat_hold_pct_adjust_<player>`).
- Sources: tennisabstract.com (ELO + surface splits), ATP/WTA official, Infosys
  ATP stats, flashscore for recent results, weather.com / Météo-France for
  outdoor-session conditions.
- code_execution: surface-conditioned ELO from tennisabstract or recent matches;
  serve+return-game model. Log5 doesn't apply cleanly to tennis — use
  surface-conditioned win-rate as the baseline instead.
""".strip(),

    ("injury", "tennis"): """
Tennis-specific focus:
- Tennis injuries are often hidden until match-time. Check for medical timeouts
  in recent matches, retirement frequency in the last 90 days, and withdrawals
  from preceding events on the same swing.
- Body-part / surface interaction: shoulder issues spike serve breakdown; lower-
  body issues are bigger on clay (longer rallies, more sliding). Surface that
  interaction in the impact note.
- A coach change mid-season often signals technical recalibration in progress —
  worth flagging even if no injury exists.
- Sources: ATP/WTA official notices, beat reporters (José Morgado, Christopher
  Clarey, Ben Rothenberg), player social media (Twitter / Instagram stories
  often leak warm-up issues).
- code_execution: tour-baseline retirement rate is ~3–5% of matches; flag
  players above baseline in the trailing 90d.
""".strip(),

    ("narrative", "tennis"): """
Tennis-specific focus:
- Tournament weight matters: Slam > Masters/1000 > 500 > 250. Top players manage
  workloads — a 250 final is a different motivation profile than a Slam QF.
- Surface preference: defending champion on this surface vs. surface debutant;
  clay-court specialists vs. hard-court grinders. Gauge from career titles per
  surface.
- Mental factors: recent loss to same opponent, comebacks from injury, end-of-
  season tank, post-Slam letdown.
- Crowd factor: home crowd impact (US Open Americans, Roland Garros French
  players, Australian Open Aussies, Davis Cup ties).
- Weather and court conditions are NOT in scope for this lens — the statistics
  fetcher quantifies them as measurable form adjustments. Do not search for
  them here.
""".strip(),

    # ---------------- SOCCER ----------------
    ("statistics", "soccer"): """
Soccer-specific focus:
- Expected goals (xG) and expected goals against (xGA) are more predictive than
  raw goal differential — pull per-match xG over the last 8–10 matches.
  Sources: fbref.com (most comprehensive), understat.com, opta when accessible.
- Tactical context: pressing intensity (PPDA), set-piece efficiency, late-game
  scoring rate. These shape how a matchup plays out beyond raw form.
- Home/away splits matter MORE in soccer than in most sports — a team's
  home-form-only and away-form-only records are often very different. Use the
  appropriate split for this fixture.
- THREE-WAY MARKET: P(home), P(draw), P(away). DRAWS are ~25–30% of top-5
  league matches. Do NOT collapse to a binary head-to-head — surface a draw
  probability when the event has 3 outcomes, and label your computed numbers
  with which outcome they reference (`p_home`, `p_draw`, `p_away`).
- Weather is a measurable xG modifier, not a storyline: heavy rain favors
  slower technical teams and reduces total goals; strong wind hurts long-ball
  teams and inflates set-piece variance; summer heat reduces high-press
  intensity and total xG. Pull current/forecast conditions and adjust the
  Poisson λ accordingly in `code_execution`, surfacing as
  `weather_xg_adjust_<team>` so the reasoner can attribute the shift.
- code_execution: Poisson model from team xG-for / xGA-against rates, or Dixon-
  Coles for low-scoring leagues. Cite the 3-way fair probabilities side by side.
""".strip(),

    ("injury", "soccer"): """
Soccer-specific focus:
- Predicted XI is the load-bearing artifact. A starting striker out vs. a bench
  player out is a ~5x impact difference. Search for "predicted lineup" and
  "team news" specifically, not just "injury report".
- Suspensions: yellow-card accumulation triggers bans (EPL: 5 yellows by match
  19, 10 by match 32, 15 by season end). Red-card suspensions are typically 1
  match (straight red) or 3 (violent conduct).
- Late fitness tests: starting XI is published ~1h before kickoff. Flag any
  player listed as "doubtful" or "to be assessed" in the 24h pre-match presser.
- Rotation in fixture-congestion windows: midweek UCL / Europa League often
  drives weekend rotation in EPL / La Liga. If the team played 72h ago, expect
  rotation; if they're playing again in 72h, expect rotation now.
- Sources: official club pre-match pressers (24h pre-match), Premier Injuries,
  beat reporters per club, transfermarkt for squad context.
- code_execution: starter-vs-bench impact via xG-share or G+A per 90; squad-
  strength delta in € via transfermarkt valuations.
""".strip(),

    ("narrative", "soccer"): """
Soccer-specific focus:
- Competition stakes shape intensity: title race, top-4 / Champions-League race,
  relegation, cup final, derby (rivalry intensity boosts both teams' output).
- Manager pressure: a recent or rumored managerial change typically yields a
  +5–8pp short-term coaching-bump effect. Flag the bump direction.
- Fixture congestion: 3 matches in 7 days vs. 1 match in 14 days drives
  rotation and fatigue. UCL midweek before a weekend league fixture is the
  classic case.
- Travel: cross-continent vs. domestic — significant for UCL knockout legs
  across European time zones, less relevant for domestic league play.
- Crowd / venue: home advantage, neutral venue (cup finals), behind-closed-
  doors penalty (worth ~3pp of home advantage).
- Weather is NOT in scope for this lens — the statistics fetcher quantifies it
  as a Poisson-λ adjustment. Do not search for forecasts here.
""".strip(),
}


def render_sport_hint(lens: LensName, event: PolymarketEvent) -> str | None:
    """Return a sport-specific hint block to append to the fetcher's user
    message, or `None` when no specialization applies.

    Detection is purely on `event.sport_type` (gamma-derived). Events where
    gamma omits the sport tag (seasonal futures, prop markets) fall through
    to None — those aren't the kind of moneylines we rank anyway.
    """
    sport = event.sport_type
    if sport is None:
        return None
    body = SPORT_HINTS.get((lens, sport))
    if body is None:
        return None
    return f"--- Sport-specific focus ({sport}, lens={lens}) ---\n{body}"


# Reasoner-side calibration anchors. The fetcher's job is to capture facts;
# the reasoner's job is to convert those facts into a typed verdict
# (probability, signed availability shift, motivation_edge). Different sports
# put very different magnitudes on the same status word — a tennis
# "withdrawal" is structural where an NBA "questionable" usually resolves
# available — so the reasoner needs sport-specific bands. Initially populated
# for the injury lens only, where the gap is biggest; statistics and
# narrative reasoners are sport-blind today and the (lens, sport) key shape
# leaves room to specialize them later.
REASONER_SPORT_HINTS: dict[tuple[LensName, str], str] = {
    # ---------------- INJURY · TENNIS ----------------
    ("injury", "tennis"): """
Tennis-specific calibration:
- Singles is structural: each "side" IS a single player. A confirmed pre-match
  withdrawal / walkover is full impact — bound the magnitude near ±0.18 to
  ±0.20 (the schema cap exists for exactly this case), not the lower 0.05–0.10
  band that team sports use.
- "Questionable" 24h pre-match has roughly a 50% out-rate; mid-tournament,
  higher. Don't under-size: a credible questionable tag is typically -0.06 to
  -0.10, not -0.02.
- Body-part / surface interaction shapes magnitude beyond the status word:
    * shoulder / wrist issues → serve breakdown; larger impact on hard / grass
      where serve dominates rallies.
    * lower-body (knee, ankle, hip) → larger impact on clay (longer rallies,
      sliding) than on hard.
- Best-of-5 (Slams, Davis Cup) amplifies marginal-injury risk vs best-of-3
  (more retirement chances, more fatigue surfacing). For a marginal injury at
  a Slam, push one band stronger than at a 250.
- Coach changes mid-season are technical disruption, not availability — keep
  them out of the signed shift even if the notebook flags them.
- `lineup_confidence`: 'confirmed' once the player is on the entry list AND
  has practiced same-day; 'probable' when listed but warm-up issues reported;
  'uncertain' otherwise.
""".strip(),

    # ---------------- INJURY · SOCCER ----------------
    ("injury", "soccer"): """
Soccer-specific calibration:
- Predicted XI is load-bearing; the impact is position-conditioned, not just
  "star out / star in". Anchor magnitudes by role:
    * starting striker out:        −0.04 to −0.08
    * key creative midfielder out: −0.03 to −0.06
    * starting goalkeeper out:     −0.03 to −0.06 (wider band — backup keeper
      quality varies a lot)
    * starting key defender out:   −0.02 to −0.04
    * bench / fringe rotation:     −0.01 to −0.02 (do not size higher even
      when the absent player is a household name in another role)
- Suspensions and red-card bans land FLATLY at the same magnitude as a
  confirmed out for that position. Don't discount because the body is
  healthy — they're equally unavailable.
- Squad depth damps any one absence by ~30% on top-6 EPL / Bundesliga sides
  with deep benches; bottom-half teams with thin benches feel the absence at
  the upper end of the band.
- Fixture congestion (UCL midweek before a weekend league fixture, or 3
  matches in 7 days) drives ROTATION, not injury. If the notebook flags
  rotation risk without a confirmed absence, treat it as
  `lineup_confidence='probable'` or `'uncertain'` — don't double-count it as
  a signed shift on top of the fitness shift.
- "Doubtful" in the 24h pre-match presser has a ~60–70% out-rate; size the
  expected impact close to the role's "out" band, not as a coin flip.
""".strip(),
}


def render_reasoner_sport_hint(
    lens: LensName, event: PolymarketEvent
) -> str | None:
    """Return a sport-specific calibration hint to append to the reasoner's
    user message, or `None` when no specialization applies.

    Mirror of `render_sport_hint`: same `event.sport_type` detection, same
    user-message-only posture (NEVER the cached system block, which would
    bust per-event cache hits across the slate). Currently populated for the
    injury lens only; other (lens, sport) combinations fall through to None
    and the reasoner uses its generic prompt unchanged.
    """
    sport = event.sport_type
    if sport is None:
        return None
    body = REASONER_SPORT_HINTS.get((lens, sport))
    if body is None:
        return None
    return f"--- Sport-specific calibration ({sport}, lens={lens}) ---\n{body}"
