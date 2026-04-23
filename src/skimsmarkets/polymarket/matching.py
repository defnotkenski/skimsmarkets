"""Cross-venue event matcher.

Given a KalshiEvent and a pool of PolymarketEvent candidates (pre-filtered by
league), pick the Polymarket event that most likely refers to the same game and
produce a per-side mapping: Kalshi's `yes_sub_title` → the slug of the
Polymarket market representing the same side.

Design posture: conservative. A false positive (wrong pairing) silently
corrupts downstream edges — the director sees a consensus that isn't there, and
sizing picks an entry price for a different game. A false negative just drops
to Kalshi-only for that event, which is today's behavior. So the matcher
returns `None` whenever the signal is ambiguous, and the pipeline carries on.

The alias map is seeded sparsely on purpose: grow it from real unmatched-event
logs rather than guessing in advance.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Sequence

from skimsmarkets.kalshi.models import KalshiEvent
from skimsmarkets.polymarket.models import PolymarketEvent, PolymarketMarket

log = logging.getLogger(__name__)

# Seed city/nickname aliases. Canonical key → set of forms (all lowercase,
# punctuation-stripped) that should collapse to the same token set. This list
# is intentionally small — extend from real run logs, not from guesses.
_CITY_NICKNAME_MAP: dict[str, set[str]] = {
    "los angeles lakers": {"la lakers", "lakers"},
    "los angeles clippers": {"la clippers", "clippers"},
    "new york yankees": {"ny yankees", "yankees"},
    "new york mets": {"ny mets", "mets"},
    "new york knicks": {"ny knicks", "knicks"},
    "new york giants": {"ny giants", "giants"},
    "new york jets": {"ny jets", "jets"},
}

_PUNCT_RX = re.compile(r"[^\w\s]")


def _normalize(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return " ".join(_PUNCT_RX.sub(" ", name.lower()).split())


def _canonical_tokens(name: str) -> frozenset[str]:
    """Collapse aliases to their canonical form's token set.

    If `name` (normalized) matches any alias in `_CITY_NICKNAME_MAP`, return
    the tokens of the canonical key. Otherwise return the normalized tokens.
    """
    n = _normalize(name)
    if not n:
        return frozenset()
    for canonical, aliases in _CITY_NICKNAME_MAP.items():
        if n == canonical or n in aliases:
            return frozenset(canonical.split())
    return frozenset(n.split())


def _overlap(a: str, b: str) -> float:
    """Jaccard-ish overlap between two side-label token sets in [0, 1]."""
    ta, tb = _canonical_tokens(a), _canonical_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


@dataclass(frozen=True)
class SideMatch:
    """A single Kalshi side paired with its Polymarket counterpart slug.

    `is_no_side` distinguishes the two halves of a head-to-head Polymarket
    market that share a slug: one Kalshi side maps to the YES direction, the
    other to NO (with prices inverted). Without this flag, BBO fan-out can't
    tell which direction to use when two Kalshi sides resolve to the same slug.
    """

    kalshi_yes_sub_title: str
    polymarket_market_slug: str
    is_no_side: bool
    # Raw score that produced this pairing; exposed for debug logging only.
    score: float


@dataclass
class EventMatch:
    polymarket_event: PolymarketEvent
    side_map: dict[str, SideMatch] = field(default_factory=dict)


def _candidate_yes_labels(pm: PolymarketMarket) -> list[str]:
    """Produce the label pool for matching against a Kalshi yes_sub_title.

    Combines the primary display label (`yes_sub_title`, typically the mascot)
    with every known alias (`team_aliases`: name/safeName/abbreviation/alias)
    so Kalshi's city-style side labels ('Cleveland') can hit a Polymarket
    market that lists the mascot ('Cavaliers'). Falls back to the market's
    raw title when no team data is present (e.g. series-winner futures).
    """
    labels: list[str] = []
    if pm.yes_sub_title:
        labels.append(pm.yes_sub_title)
    for alias in pm.team_aliases:
        if alias not in labels:
            labels.append(alias)
    if pm.title and pm.title not in labels:
        labels.append(pm.title)
    return labels


def _time_delta_hours(
    kalshi_event: KalshiEvent,
    pm: PolymarketEvent,
) -> float | None:
    """Hours between the Kalshi settlement time and the Polymarket game-start time.

    Returns None when either side has no times. Uses Polymarket's
    `game_start_time` rather than `expected_expiration_time` because the
    latter is a ~2-week settlement window that makes all same-game matches
    look identical while wildly inflating next-day-game false positives
    against same-team opponents.

    For correctly-paired games this delta is typically 2–5 hours (Kalshi's
    expected_expiration sits shortly after game end, Polymarket's game start
    is at tipoff); a wrong-day pairing produces a 24h+ delta.
    """
    k_times = [m.expected_expiration_time for m in kalshi_event.markets
               if m.expected_expiration_time is not None]
    p_times = [m.game_start_time for m in pm.markets
               if m.game_start_time is not None]
    if not k_times or not p_times:
        return None
    k = min(k_times)
    p = min(p_times)
    return abs((k - p).total_seconds()) / 3600.0


def _time_bonus(
    kalshi_event: KalshiEvent,
    pm: PolymarketEvent,
    tolerance: timedelta,
) -> float:
    """Small additive bonus when settlement times look close. Zero if missing.

    Capped at 0.1 so name overlap remains the dominant signal; the ambiguity
    tiebreaker in `match_event` uses the raw delta for finer-grained ranking.
    """
    delta_hours = _time_delta_hours(kalshi_event, pm)
    if delta_hours is None:
        return 0.0
    tolerance_hours = tolerance.total_seconds() / 3600.0
    if delta_hours <= tolerance_hours:
        return 0.1 * (1.0 - delta_hours / tolerance_hours)
    return 0.0


# Polymarket `sportsMarketType` values that represent a direct "who wins"
# binary comparable to a Kalshi head-to-head side. Spreads / totals / futures
# are excluded: they share team labels with the moneyline market inside the
# same Polymarket event, and without this filter the matcher happily paired a
# Kalshi moneyline side to a "Team A covers +8.5" spread (same team_aliases,
# wildly different implied probability). Unknown / missing type is allowed
# through — keeps older records that predate this field still-matchable.
_HEAD_TO_HEAD_MARKET_TYPES: frozenset[str] = frozenset({
    "moneyline",
    "drawable_outcome",
})


def _build_side_map(
    kalshi_event: KalshiEvent,
    pm: PolymarketEvent,
    *,
    min_side_overlap: float,
) -> dict[str, SideMatch]:
    """Pair each Kalshi yes_sub_title with its best Polymarket side by label overlap.

    A Polymarket head-to-head market appears twice in `pm.markets`: once as
    the YES direction and once as the NO direction (same slug, is_no_side
    toggled). The used-key is `(slug, is_no_side)` so one slug can pair
    against two Kalshi sides — YES to one team, NO to the other — which is
    the normal case for NBA/NHL/MLB moneylines.

    Candidates are filtered to moneyline / drawable_outcome market types so a
    Kalshi moneyline can't accidentally pair with a spread market that shares
    the same team names. Unknown/missing types are allowed as a back-compat
    fallback.
    """
    side_map: dict[str, SideMatch] = {}
    used: set[tuple[str, bool]] = set()
    for k_market in kalshi_event.markets:
        if not k_market.yes_sub_title:
            continue
        best: tuple[float, PolymarketMarket | None] = (0.0, None)
        for pm_market in pm.markets:
            if (
                pm_market.sports_market_type is not None
                and pm_market.sports_market_type not in _HEAD_TO_HEAD_MARKET_TYPES
            ):
                continue
            key = (pm_market.slug, pm_market.is_no_side)
            if key in used:
                continue
            candidates = _candidate_yes_labels(pm_market)
            if not candidates:
                continue
            score = max(_overlap(k_market.yes_sub_title, c) for c in candidates)
            if score > best[0]:
                best = (score, pm_market)
        if best[1] is not None and best[0] >= min_side_overlap:
            side_map[k_market.yes_sub_title] = SideMatch(
                kalshi_yes_sub_title=k_market.yes_sub_title,
                polymarket_market_slug=best[1].slug,
                is_no_side=best[1].is_no_side,
                score=best[0],
            )
            used.add((best[1].slug, best[1].is_no_side))
    return side_map


def match_event(
    kalshi_event: KalshiEvent,
    candidates: Sequence[PolymarketEvent],
    *,
    settlement_tolerance_hours: float = 24.0,
    min_side_overlap: float = 0.5,
) -> EventMatch | None:
    """Pick the best Polymarket event for a KalshiEvent, then map sides.

    Returns None when the signal is ambiguous. Requires at least one side to
    match above threshold; for multi-market events, requires ≥2 sides matched
    (or a single dominant side with overlap ≥0.8) to avoid pairing off a
    single-team coincidence in an otherwise-different event.
    """
    if not candidates:
        return None

    tolerance = timedelta(hours=settlement_tolerance_hours)
    # Hard time-proximity filter. A Polymarket event whose game-start is more
    # than a week off the Kalshi event is almost certainly a different thing
    # (season futures, wrong game, etc.). When time data is missing on either
    # side, we let the candidate through and fall back to name-only ranking.
    max_game_delta_hours = 24.0 * 7.0
    scored: list[tuple[float, PolymarketEvent, dict[str, SideMatch]]] = []
    for pm in candidates:
        side_map = _build_side_map(kalshi_event, pm, min_side_overlap=min_side_overlap)
        if not side_map:
            continue
        delta = _time_delta_hours(kalshi_event, pm)
        if delta is not None and delta > max_game_delta_hours:
            log.debug(
                "kalshi event %s: candidate %s dropped — game-start delta %.1fh > %.0fh cap",
                kalshi_event.event_ticker, pm.slug, delta, max_game_delta_hours,
            )
            continue
        # Coverage-weighted: sum side scores (not mean) so a candidate that
        # matches more sides beats one that matches a single side exactly.
        # Without this, a wrong game that shares one team name outranks the
        # right game that partially matches both teams.
        sum_side_score = sum(m.score for m in side_map.values())
        score = sum_side_score + _time_bonus(kalshi_event, pm, tolerance)
        scored.append((score, pm, side_map))

    if not scored:
        return None

    scored.sort(key=lambda s: s[0], reverse=True)
    top_score, top_event, top_side_map = scored[0]

    kalshi_side_count = sum(1 for m in kalshi_event.markets if m.yes_sub_title)
    # Multi-market Kalshi event (head-to-head game with a side per team): must
    # pair ≥2 Polymarket sides. A single-side hit on a 2-team event is a classic
    # false positive — one team name coincidentally matches a different game on
    # the same league. No single-side shortcut here; the cost of a wrong edge
    # is higher than the cost of dropping to Kalshi-only. Single-market Kalshi
    # events (futures, prop bets) are the only place a single-side match is OK.
    if kalshi_side_count >= 2 and len(top_side_map) < 2:
        log.debug(
            "kalshi event %s: top polymarket candidate %s only matched %d of "
            "%d sides — rejecting as ambiguous (likely cross-game name collision)",
            kalshi_event.event_ticker,
            top_event.slug,
            len(top_side_map),
            kalshi_side_count,
        )
        return None

    # Tie-break within 0.10 by settlement-time proximity. A team name shared
    # across multiple MLS games (e.g. "San Diego FC" on both 4/22 and 4/25) is
    # common enough that name overlap alone can't disambiguate — but the
    # Kalshi event's settlement timestamp pins which game we mean. We reject
    # only if names tie AND times don't disambiguate either.
    if len(scored) >= 2 and (top_score - scored[1][0]) < 0.10:
        tied = [s for s in scored if top_score - s[0] < 0.10]
        # Rank tied candidates by time proximity; the one with smallest delta wins.
        ranked_by_time: list[tuple[float | None, tuple[float, PolymarketEvent, dict[str, SideMatch]]]] = [
            (_time_delta_hours(kalshi_event, s[1]), s) for s in tied
        ]
        ranked_by_time.sort(
            key=lambda pair: (pair[0] if pair[0] is not None else float("inf"))
        )
        best_delta = ranked_by_time[0][0]
        runner_up_delta = ranked_by_time[1][0] if len(ranked_by_time) > 1 else None
        if (
            best_delta is not None
            and runner_up_delta is not None
            and (runner_up_delta - best_delta) >= 12.0
        ):
            # 12h gap is enough to call one game distinct from the other.
            _, (top_score, top_event, top_side_map) = ranked_by_time[0]
            log.debug(
                "kalshi event %s: broke score tie by time proximity — chose %s (Δ=%.1fh) over %s (Δ=%.1fh)",
                kalshi_event.event_ticker, top_event.slug, best_delta,
                ranked_by_time[1][1][1].slug, runner_up_delta,
            )
        else:
            log.debug(
                "kalshi event %s: top %d polymarket candidates ambiguous "
                "(scores %s, time deltas %s) — rejecting",
                kalshi_event.event_ticker, len(tied),
                [f"{s[0]:.2f}" for s in tied],
                [f"{d:.1f}h" if d is not None else "?" for d, _ in ranked_by_time],
            )
            return None

    return EventMatch(polymarket_event=top_event, side_map=top_side_map)
