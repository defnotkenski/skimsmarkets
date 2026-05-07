"""Sport gate for tennis stats enrichment.

Single helper that returns a `TennisMatchIdentity` for ATP/WTA singles
head-to-heads and `None` for anything else. Centralises the "is this a
tennis match worth enriching?" decision in one place so the pipeline
stage and any future caller (CLI debugging, tests) ask the same
question the same way.

Intentional restrictions for v1:
- ATP / WTA singles only. Doubles markets often arrive with `/`-joined
  player pairs (e.g. `"Bopanna / Ebden vs Salisbury / Ram"`) which break
  the simple-name `_parse_h2h_question` contract.
- Single-tour matchups only. ATP-vs-WTA novelty markets exist on
  Polymarket; they don't share rankings tables and the vendor lookup
  would need cross-tour identity resolution we've explicitly deferred.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from skimsmarkets.polymarket.models import PolymarketEvent, _parse_h2h_question

# Slug prefixes gamma uses for ATP / WTA events. Matched on a leading
# segment with a trailing dash so partial collisions (`atpx-...`) can't
# slip through. Anchored on the slug because `event.sport_type` from
# `_gamma_sport_from_tags` collapses to the umbrella `"tennis"` and
# loses the tour distinction we need to call the right vendor endpoint.
_ATP_SLUG_PREFIX = "atp-"
_WTA_SLUG_PREFIX = "wta-"

# Doubles-team separators we drop on. Most vendors don't model doubles
# pairs as singletons; we punt rather than try to look up four players.
_DOUBLES_HINT_TOKENS: tuple[str, ...] = ("/", "&", " and ")

# Slug-prefix → surface mapping for the major tournaments. Polymarket
# tennis volume concentrates here (4 Slams + ATP/WTA Masters/1000
# swing), so a hardcoded table covers most events without needing a
# vendor lookup. Lower-tier 250 / 500 events miss this map and fall
# through to the modal-recent-surface fallback at the call site —
# graceful degrade, never abort scoring on an unrecognised slug.
#
# Keys match the slug remainder AFTER the `atp-` / `wta-` tour
# prefix is stripped. Year suffixes (e.g. `-2026-...`) follow
# naturally because the matcher uses startswith. Multiple aliases
# per tournament cover the spelling variants Polymarket has used
# historically.
_SLUG_SURFACE_MAP: dict[str, str] = {
    "australian-open": "hard",
    "aus-open": "hard",
    "french-open": "clay",
    "roland-garros": "clay",
    "wimbledon": "grass",
    "us-open": "hard",
    "indian-wells": "hard",
    "miami-open": "hard",
    "miami": "hard",
    "monte-carlo": "clay",
    "madrid-open": "clay",
    "madrid": "clay",
    "italian-open": "clay",
    "rome-open": "clay",
    "rome": "clay",
    "canadian-open": "hard",
    "canada-open": "hard",
    "national-bank-open": "hard",
    "cincinnati-open": "hard",
    "cincinnati": "hard",
    "shanghai-masters": "hard",
    "shanghai": "hard",
    "paris-masters": "hard",
    "paris-rolex": "hard",
}

# Slug-prefix → tournament tier. Same key vocabulary as
# `_SLUG_SURFACE_MAP` so the two cascades read off the same matched
# slug remainder. Tiers come from `_RANK_ID_TO_TIER` in
# `tennis/matchstat.py` so downstream consumers (recent-match rows,
# the multiplier table in `selection.py`) share one vocabulary.
_SLUG_TIER_MAP: dict[str, str] = {
    "australian-open": "grand_slam",
    "aus-open": "grand_slam",
    "french-open": "grand_slam",
    "roland-garros": "grand_slam",
    "wimbledon": "grand_slam",
    "us-open": "grand_slam",
    "indian-wells": "masters",
    "miami-open": "masters",
    "miami": "masters",
    "monte-carlo": "masters",
    "madrid-open": "masters",
    "madrid": "masters",
    "italian-open": "masters",
    "rome-open": "masters",
    "rome": "masters",
    "canadian-open": "masters",
    "canada-open": "masters",
    "national-bank-open": "masters",
    "cincinnati-open": "masters",
    "cincinnati": "masters",
    "shanghai-masters": "masters",
    "shanghai": "masters",
    "paris-masters": "masters",
    "paris-rolex": "masters",
}


class TennisMatchIdentity(BaseModel):
    """The two players plus tour, ready for the vendor lookup.

    `player_a` matches the favorite (Polymarket YES side after gamma's
    head-to-head expansion); `player_b` is the underdog. The contract
    matches `team_a_name` / `team_b_name` plumbing in the rest of the
    pipeline so the reasoner can line the vendor's data up against the
    event context labels without renaming.
    """

    model_config = ConfigDict(extra="ignore")

    player_a: str
    player_b: str
    tour: Literal["atp", "wta"]
    # Best-effort tournament hint pulled from the prefix of the gamma
    # question (the bit before the first `:`). The vendor often needs a
    # tournament context to disambiguate H2H by event; we surface it
    # here and let the provider decide whether to use it.
    tournament_hint: str | None = None


def _looks_like_doubles(name: str) -> bool:
    """True when the parsed side label looks like a doubles pair.

    Probed only after `_parse_h2h_question` has already split the
    `vs`-separated halves, so the input is one *side*, not the full
    question. We bail on any common pair separator regardless of order.
    """
    return any(tok in name for tok in _DOUBLES_HINT_TOKENS)


def _tour_from_slug(slug: str) -> Literal["atp", "wta"] | None:
    if slug.startswith(_ATP_SLUG_PREFIX):
        return "atp"
    if slug.startswith(_WTA_SLUG_PREFIX):
        return "wta"
    return None


def _slug_remainder(slug: str) -> str:
    """Strip the `atp-` / `wta-` prefix; return `""` for non-tour slugs.

    Used by both `_slug_surface` and `_slug_tier` so they match against
    the same canonical post-prefix string. Empty string when the slug
    isn't tour-prefixed at all — both downstream parsers treat that as
    "no match".
    """
    if slug.startswith(_ATP_SLUG_PREFIX):
        return slug[len(_ATP_SLUG_PREFIX):]
    if slug.startswith(_WTA_SLUG_PREFIX):
        return slug[len(_WTA_SLUG_PREFIX):]
    return ""


def _slug_surface(slug: str) -> str | None:
    """Hardcoded slug-prefix → surface lookup for the major tournaments.

    Returns one of the `_COURT_ID_TO_SURFACE` values
    ("hard"/"clay"/"grass") for the 4 Slams + ATP/WTA Masters/1000
    swing, or None when the slug doesn't match any known major.
    Selection-stage callers fall back to a modal-recent-surface
    inference when this returns None — see
    `_resolve_event_surface` in `selection.py`.

    Match logic: strip the tour prefix, then check whether the
    remainder STARTS WITH any key in `_SLUG_SURFACE_MAP`. `startswith`
    (rather than equality) handles year/round suffixes like
    `-2026-sinner-vs-alcaraz` that follow the tournament name in
    real Polymarket slugs.

    Multiple aliases collapse to the same canonical surface — Rome
    is `italian-open` on some payloads, `rome-open` on others, etc.
    Iteration order in `_SLUG_SURFACE_MAP` doesn't matter because
    the keys don't overlap (no key is a prefix of another for the
    populated tournaments).
    """
    remainder = _slug_remainder(slug)
    if not remainder:
        return None
    for key, surface in _SLUG_SURFACE_MAP.items():
        if remainder.startswith(key):
            return surface
    return None


def _slug_tier(slug: str) -> str | None:
    """Hardcoded slug-prefix → tournament tier lookup.

    Returns one of `"grand_slam"` / `"masters"` for the 4 Slams +
    ATP/WTA Masters/1000 swing, None for everything else (250s,
    500s, qualifiers, futures, exhibitions). Companion to
    `_slug_surface` — same key vocabulary, same `startswith` matcher.
    Selection-stage callers fall back to the favorite's most recent
    cached match tier when this returns None — see
    `_resolve_event_tier` in `selection.py`.
    """
    remainder = _slug_remainder(slug)
    if not remainder:
        return None
    for key, tier in _SLUG_TIER_MAP.items():
        if remainder.startswith(key):
            return tier
    return None


def _favorite_market_index(event: PolymarketEvent) -> int | None:
    """Index of the YES-side market with the higher implied probability.

    Mirrors the favourite-pick logic in `agents/fetchers/base.py:pick_team_a_market`
    but returns an index so the caller can read both the favorite's
    `yes_sub_title` (for player_a) and the paired NO clone's
    `yes_sub_title` (for player_b) without re-walking the list.

    Returns None when the event has no two-sided H2H — futures, 3-way
    soccer, or thinly-quoted markets with one side stripped upstream.
    """
    candidates = [
        (i, m)
        for i, m in enumerate(event.markets)
        if m.yes_sub_title and m.yes_implied_probability is not None
    ]
    if len(candidates) < 2:
        return None
    candidates.sort(key=lambda pair: pair[1].yes_implied_probability or -1.0, reverse=True)
    return candidates[0][0]


def tennis_match_identity(event: PolymarketEvent) -> TennisMatchIdentity | None:
    """Return identity for a single ATP/WTA singles match, or None.

    Decision tree:
      1. Sport must be tennis. Both `event.sport_type == "tennis"` (the
         tag-derived signal) and an `atp-` / `wta-` slug prefix must
         agree. Either alone is too brittle: futures bundles tag as
         tennis but aren't matches; non-tour exhibitions occasionally
         carry the prefix without the tag.
      2. The slug prefix decides the tour.
      3. The two side labels come from `_parse_h2h_question` on the
         favorite market's title; the YES / NO `yes_sub_title` strings
         feed `player_a` / `player_b` (favorite first, mirroring
         `team_a_name` plumbing).
      4. Both names must look like singles players — anything resembling
         a doubles pair short-circuits to None.
    """
    if event.sport_type != "tennis":
        return None
    tour = _tour_from_slug(event.slug or "")
    if tour is None:
        return None

    fav_idx = _favorite_market_index(event)
    if fav_idx is None:
        return None
    fav_market = event.markets[fav_idx]
    other = next(
        (m for i, m in enumerate(event.markets) if i != fav_idx and m.yes_sub_title),
        None,
    )
    if other is None:
        return None

    player_a = (fav_market.yes_sub_title or "").strip()
    player_b = (other.yes_sub_title or "").strip()
    if not player_a or not player_b:
        return None
    if _looks_like_doubles(player_a) or _looks_like_doubles(player_b):
        return None

    # Reuse `_parse_h2h_question` on the question title to harvest the
    # tournament prefix when present. The names from `marketSides` are
    # already authoritative — we only want the prefix here.
    tournament_hint: str | None = None
    title = fav_market.title or ""
    if ":" in title:
        head = title.split(":", 1)[0].strip()
        # Sanity-check that the body really was an h2h shape; otherwise
        # the prefix might just be a leading colon in a different format.
        if _parse_h2h_question(title) is not None and head:
            tournament_hint = head

    return TennisMatchIdentity(
        player_a=player_a,
        player_b=player_b,
        tour=tour,
        tournament_hint=tournament_hint,
    )
