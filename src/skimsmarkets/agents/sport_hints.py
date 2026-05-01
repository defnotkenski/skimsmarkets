"""Sport-specific guidance injected into fetcher user messages.

The system prompts in `prompts.py` are sport-agnostic and stay cached
(`CacheControlEphemeralParam` requires fixed strings). Sport-specific
search-shaping guidance lives here and rides on the per-event user message,
so the cache hit on the system block is preserved.

Scope is intentionally narrow:
- Fetchers only. Reasoners structure what's already in the notebook, so the
  fetcher's specialization is what determines whether the right facts get
  captured at all. (Future: the injury reasoner has a sport-conditioned
  status→impact calibration that warrants its own specialization, but that
  is left for a follow-up.)
- Tennis and soccer only. These two have the most distinctive search patterns
  (surface splits + serve stats; xG + 3-way pricing + predicted XI) and
  enough Polymarket volume to validate. Other sports fall through to the
  generic prompt.
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
- Sources: tennisabstract.com (ELO + surface splits), ATP/WTA official, Infosys
  ATP stats, flashscore for recent results.
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
- Weather / venue: heat (hard court > clay), wind on outdoor courts (especially
  US Open day sessions), clay drying speed after rain, roof open/closed in
  night sessions.
""".strip(),

    ("market_context", "tennis"): """
Tennis-specific focus:
- Sportsbook coverage of tennis is thinner than NBA/NFL. Pinnacle and Bet365
  are sharpest; Betfair Exchange gives the truest market price. Most US books
  (DraftKings, FanDuel, BetMGM) lag and offer wider spreads.
- Tennis is a live-betting-dominant market — pre-match prices may not reflect
  late warm-up news or court conditions. Open-vs-current line movement in the
  last 2–4 hours pre-match is the sharp signal.
- Sources: oddsportal.com / betexplorer.com for line history; Betfair Exchange
  for the live equilibrium price.
- de-vig two-sided ML directly (no draws in tennis singles). Singles only —
  doubles markets have very different liquidity patterns.
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
- Weather: rain favors slower technical teams that don't rely on quick
  transitions; wind hurts long-ball teams; heat in summer-league fixtures.
""".strip(),

    ("market_context", "soccer"): """
Soccer-specific focus:
- THREE-WAY pricing (home / draw / away) — de-vig the FULL triplet, not just
  two sides. Surface fair P(home), P(draw), P(away) as separate computed
  numbers. Top-5 leagues have ~25–30% draw rates baked into market prices.
- Asian Handicap (AH) lines from Pinnacle and Asian books often carry SHARPER
  signal than European 1X2 lines. AH 0 (draw void) approximates a two-way
  market and is a useful sanity check on European pricing.
- Sportsbook coverage of soccer is mature and competitive. Pinnacle, Bet365,
  Betfair Exchange are sharpest. Avoid using US-facing books (DraftKings,
  FanDuel) as the consensus signal — they price on US recreational flow.
- Line movement is sharpest in the 24h pre-kickoff after team news drops.
  Open-vs-current movement in that window is the sharp action.
- Sources: oddsportal.com / betexplorer.com for line history, Betfair Exchange
  for the truest market, soccerway for fixture context.
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
