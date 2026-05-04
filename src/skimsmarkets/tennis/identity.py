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
