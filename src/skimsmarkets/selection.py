"""Pre-LLM event selection — pick the top-N matchups by *fundamental
imbalance* rather than by tipoff time.

Why this stage exists:
- Slates often arrive bigger than `MAX_SLATE_EVENTS`, and we cap before
  spending LLM tokens on lens chains and director synthesis.
- The historical cap was tipoff-sorted, which optimised for "soonest
  games" — fine when latency matters, wrong when the goal is "most
  defensible picks." Tipoff carries no signal about which event will
  produce a confident, well-aligned lens read.
- The slate judge scores `defensibility_score` on reasoning coherence +
  lens alignment + UW agreement. Empirically the events that score
  highest are the ones with **clear quality differentials** — lopsided
  matchups where specialists agree on direction. Coin-flip matchups
  where fundamentals are balanced can't produce confident reads no
  matter how many tokens the lenses get.

What "imbalance" means per sport, with cheap signals available pre-LLM:

- **Tennis**: rank-points ratio between the two players, sourced from
  the cached MatchStat rankings index. The provider warms the index
  once at startup (5 HTTP calls per tour) and `lookup_player_rank` is
  O(1) thereafter, so pre-cap scoring costs ~zero per event regardless
  of slate size. Points (not position) is the load-bearing field —
  ATP/WTA points spread is non-linear in rank, so points capture skill
  gap better. Sinner (14k pts) vs Alcaraz (13k pts) reads as nearly
  even (ratio 1.1), Sinner vs a rank-300 player reads as a blowout
  (ratio ~30x).

- **Team sports** (soccer, NBA, NFL, MLB, NHL, etc.): win-pct delta
  between the two sides, parsed from the `team_record` string
  ("28-6", "10-3", "12-2-3") gamma exposes on every market. Free —
  already on the bulk gamma `/events` payload at cap time.

- **Sports with no record** (futures-style, niche events): score 0 and
  fall through to the tipoff tiebreaker. We don't try to guess
  fundamental imbalance for sports we have no data on.

Cross-sport scaling: tennis points-ratio uses log10 normalisation
clipped to [0, 1] (a 10× points ratio caps the score at 1.0); team
win-pct delta is naturally [0, 1]. Both share a unit so a
mixed-sport slate sorts coherently.

Tipoff is the explicit tiebreaker for events sharing the same
imbalance score (typically: events without stat-based signal all
score 0.0). That preserves the "soonest first" intuition for the
fallback while letting genuine high-imbalance events override it.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime

from skimsmarkets.polymarket.models import PolymarketEvent
from skimsmarkets.tennis.identity import tennis_match_identity
from skimsmarkets.tennis.provider import TennisStatsProvider

log = logging.getLogger(__name__)

# Tennis imbalance normaliser. A points ratio of `_TENNIS_POINTS_RATIO_CAP`
# saturates the score at 1.0 — beyond that, the matchup is already as
# lopsided as our scoring needs to know. 10× chosen because most ATP/WTA
# tour-level matchups land in 1× to 5× ratio territory; only true
# qualifier-vs-top-10 matchups exceed 10×, and they all deserve the
# ceiling.
_TENNIS_POINTS_RATIO_CAP = 10.0


def _parse_team_record(record: str) -> tuple[int, int, int] | None:
    """Parse `"W-L"` or `"W-L-T"` into `(wins, losses, ties)`.

    Returns None for unparseable input (futures-style empty records,
    non-numeric content, wrong field count). NHL-style "W-L-OTL" looks
    like a 3-tuple but the third number is overtime losses, not ties —
    which is a *minor* over-counting in win-pct (treating OTL as a tie
    bumps up the win-pct slightly). We don't try to disambiguate by
    sport because the small mis-weighting is dwarfed by the real
    imbalance signal between e.g. a 35-15 team and a 12-38 team.
    """
    parts = record.split("-")
    if len(parts) < 2 or len(parts) > 3:
        return None
    try:
        nums = [int(p.strip()) for p in parts]
    except ValueError:
        return None
    if any(n < 0 for n in nums):
        return None
    if len(nums) == 2:
        return (nums[0], nums[1], 0)
    return (nums[0], nums[1], nums[2])


def _win_pct_from_record(record: str) -> float | None:
    """Convert a `team_record` string to a `[0, 1]` win percentage.

    Ties / draws / OTLs count as half-wins (standard sports
    convention). Returns None when the record has zero games (e.g.
    pre-season "0-0") so the caller doesn't see misleading 0.5 win-pct
    on no-sample teams.
    """
    parsed = _parse_team_record(record)
    if parsed is None:
        return None
    wins, losses, ties = parsed
    games = wins + losses + ties
    if games <= 0:
        return None
    return (wins + 0.5 * ties) / games


def _team_record_imbalance(event: PolymarketEvent) -> float | None:
    """Win-pct spread across all markets in the event.

    Two market shapes need different handling and a single iteration
    works for both:
      - **Binary head-to-heads** (MLB / NBA / NFL / UFC etc., the
        `_parse_h2h_question` shape): one YES market + a synthesized
        NO clone via `inverted_no_side`. The clone's `team_record`
        carries the *opposite* team's record (passed in as `no_record=`
        at synthesis time), so iterating both YES and NO surfaces
        gives the two distinct team records we need.
      - **3-way multi-outcome** (soccer with home/draw/away): each
        outcome is its own YES market with its own `team_record`;
        no NO clones synthesized, so iteration just reads each YES
        market once.

    `max - min` captures the gap between the strongest and weakest
    sides. For 3-way soccer with a clear favourite + clear underdog,
    that's the biggest imbalance regardless of where the draw row
    sits. Duplicate records (if any) collapse harmlessly because both
    `max` and `min` are idempotent.

    Returns None when fewer than two markets carry parseable records —
    typically futures, niche events, or H2H markets where gamma
    omitted one team's record entry.
    """
    win_pcts: list[float] = []
    for m in event.markets:
        if m.team_record is None:
            continue
        wp = _win_pct_from_record(m.team_record)
        if wp is not None:
            win_pcts.append(wp)
    if len(win_pcts) < 2:
        return None
    return max(win_pcts) - min(win_pcts)


def _tennis_imbalance(
    event: PolymarketEvent, provider: TennisStatsProvider
) -> float | None:
    """Log-points-ratio imbalance for a tennis event.

    Decision tree mirrors the enrichment gate (`tennis_match_identity`):
      - Sport must be tennis with an ATP/WTA slug prefix.
      - Both players must look like singles names.
      - Both must resolve in the warm rankings index with non-None
        `(position, points)`.
    Any miss returns None and the caller falls back to other signals.

    Score: `log10(max_points / min_points) / log10(cap)` clipped to
    [0, 1]. Points ratio is the right scale because ATP/WTA points
    are roughly proportional to "tour-level wins weighted by tier" —
    a 2× points ratio is meaningfully lopsided regardless of which
    region of the rankings the players sit in (Sinner 14k vs Alcaraz
    13k is *not* lopsided despite being ranks 1 vs 2; Player-100
    1500pts vs Player-200 750pts *is* lopsided despite the players
    being merely "two journeymen").
    """
    identity = tennis_match_identity(event)
    if identity is None:
        return None
    a_hit = provider.lookup_player_rank(identity.tour, identity.player_a)
    b_hit = provider.lookup_player_rank(identity.tour, identity.player_b)
    if a_hit is None or b_hit is None:
        return None
    _, points_a = a_hit
    _, points_b = b_hit
    if points_a <= 0 or points_b <= 0:
        return None
    ratio = max(points_a, points_b) / min(points_a, points_b)
    return min(1.0, math.log10(ratio) / math.log10(_TENNIS_POINTS_RATIO_CAP))


def imbalance_score(
    event: PolymarketEvent, tennis_provider: TennisStatsProvider
) -> float:
    """Composite per-event imbalance in `[0, 1]` (higher = more lopsided).

    Sport detection cascade: tennis first (cheapest to compute, highest
    confidence in signal because we have explicit ranking points), then
    team-record-based for everything else, then 0 as fallback. The
    cascade short-circuits at the first sport that produces a non-None
    signal — we don't try to combine tennis + team-record on the same
    event because they're never both populated.

    `tennis_provider` is the same provider used for full enrichment;
    the rank lookup against its warm index is free (no HTTP). The stub
    provider's `lookup_player_rank` always returns None, so under the
    no-key configuration tennis events fall through to score 0 and
    the tipoff tiebreaker decides.
    """
    s = _tennis_imbalance(event, tennis_provider)
    if s is not None:
        return s
    s = _team_record_imbalance(event)
    if s is not None:
        return s
    return 0.0


_FAR_FUTURE = datetime.max.replace(tzinfo=UTC)


def _earliest_tipoff(event: PolymarketEvent) -> datetime:
    """Earliest `game_start_time` across the event's markets.

    Mirrors the previous tipoff-sort key in `fetch_gamma_slate`. Events
    without any populated game-start time sort last so they don't
    displace tradable events at the head when used as the tipoff
    tiebreaker.
    """
    starts = [t for m in event.markets if (t := m.game_start_time) is not None]
    return min(starts) if starts else _FAR_FUTURE


async def select_top_events(
    events: list[PolymarketEvent],
    *,
    max_events: int,
    tennis_provider: TennisStatsProvider,
) -> list[PolymarketEvent]:
    """Apply imbalance scoring and cap the slate to `max_events`.

    No-op fast path: if the slate already fits under the cap, we skip
    the entire scoring pass and return events sorted by tipoff (the
    pipeline's ambient ordering — preserved so downstream stages that
    log "first event" / "last event" stay deterministic).

    When the slate exceeds the cap:
      1. Pre-warm the tennis rankings index for every tour represented
         in the slate (idempotent; ~5 HTTP calls per tour, one-time
         per process). Non-tennis-only slates skip this step.
      2. Score every event by `imbalance_score`.
      3. Sort by `(score desc, tipoff asc)` so high-imbalance events
         lead and tipoff is the deterministic tiebreaker among
         equally-scored events.
      4. Slice to `max_events`.

    The tipoff tiebreaker matters most for events scoring 0.0 (sports
    with no stat-based signal): they all share the bottom of the
    ranking and tipoff order picks among them, preserving the
    "soonest first" intuition for the fallback layer.
    """
    if max_events <= 0 or len(events) <= max_events:
        return events

    # Warm tennis index for every tour present in the slate. Stub
    # provider no-ops; no key configured = no warmup cost.
    tennis_tours: set[str] = set()
    for ev in events:
        ident = tennis_match_identity(ev)
        if ident is not None:
            tennis_tours.add(ident.tour)
    if tennis_tours:
        await tennis_provider.warm_for_selection(tennis_tours)

    scored = [
        (ev, imbalance_score(ev, tennis_provider), _earliest_tipoff(ev))
        for ev in events
    ]
    # Sort key: descending score, ascending tipoff. Negate score so
    # tuple-sort lands the right direction without `reverse=True`
    # (which would flip the tipoff direction too).
    scored.sort(key=lambda triple: (-triple[1], triple[2]))
    selected = [ev for ev, _, _ in scored[:max_events]]
    cut = len(events) - max_events
    top_score = scored[0][1] if scored else 0.0
    cut_score = scored[max_events - 1][1] if max_events <= len(scored) else 0.0
    log.info(
        "selected %d/%d events by imbalance (top=%.2f, cut@=%.2f); dropped %d",
        max_events, len(events), top_score, cut_score, cut,
    )
    return selected
