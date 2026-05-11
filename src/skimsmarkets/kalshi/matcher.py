"""Tennis-specific matcher: PredictionRow → KalshiMarket.

Kalshi tennis match events title themselves `"{LastName} vs {LastName}"`
with no tournament prefix; Polymarket gives us full names (first +
last, sometimes diacritics, sometimes hyphenated givens). The matcher
normalises both sides through `_normalize_name` (lowercase, strip
diacritics, collapse hyphens → spaces), extracts the last whitespace-
separated token as the surname, and finds the Kalshi event whose
normalised title contains both surnames.

Within a matched event, the YES market for the predicted winner is
the one whose `yes_sub_title` contains the winner's surname. We
verify uniqueness — two players with the same surname in the same
match would be ambiguous; report it and skip rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from skimsmarkets.kalshi.models import KalshiEvent, KalshiMarket
from skimsmarkets.polymarket.models import _parse_h2h_question
from skimsmarkets.retro.models import PredictionRow
from skimsmarkets.tennis.matchstat import _normalize_name


def last_token(full_name: str) -> str:
    """Return the last whitespace-separated token of a normalised name.

    `_normalize_name` collapses hyphens to spaces, so `"En-Shuo Liang"`
    normalises to `"en shuo liang"` and this returns `"liang"`. For
    `"Frances Tiafoe"` → `"tiafoe"`. Empty input → `""` (caller
    treats as "no match possible").
    """
    tokens = _normalize_name(full_name).split()
    return tokens[-1] if tokens else ""


def extract_match_players(row: PredictionRow) -> tuple[str, str] | None:
    """Return `(predicted_winner_name, opponent_name)` as full normalised names.

    Three sources, in priority order:
      1. `tennis_stats.player_a.name` / `player_b.name` — typed,
         already-cleaned vendor names. Most reliable.
      2. `event_title` parsed via `_parse_h2h_question` — Polymarket's
         "Tournament: Player A vs. Player B" pattern. Works when
         tennis_stats is missing or on non-tennis rows that still
         carry an h2h title.
      3. Fall through to None → caller treats as "no match possible".

    The predicted-winner orientation comes from `row.predicted_winner`:
    whichever of the two players has the matching surname is the
    winner; the other is the opponent.
    """
    a_name: str | None = None
    b_name: str | None = None
    if row.tennis_stats is not None:
        a_name = row.tennis_stats.player_a.name
        b_name = row.tennis_stats.player_b.name
    if (a_name is None or b_name is None) and row.event_title:
        parsed = _parse_h2h_question(row.event_title)
        if parsed is not None:
            a_name, b_name = parsed
    if not a_name or not b_name:
        return None

    winner_norm = _normalize_name(row.predicted_winner)
    a_norm = _normalize_name(a_name)
    b_norm = _normalize_name(b_name)
    if winner_norm == a_norm:
        return a_name, b_name
    if winner_norm == b_norm:
        return b_name, a_name
    # The director sometimes names the winner with different
    # punctuation than the title or vendor (rare but observed on
    # hyphen vs space). Fall back to surname-only match.
    winner_last = last_token(row.predicted_winner)
    if winner_last and winner_last == last_token(a_name):
        return a_name, b_name
    if winner_last and winner_last == last_token(b_name):
        return b_name, a_name
    return None


@dataclass(frozen=True)
class MatchOutcome:
    """Result of matching one prediction row to a Kalshi market.

    `kind` discriminates the cases the trader handles:
      - `"matched"`: `market` is populated, place the trade.
      - `"no_kalshi_match"`: no event whose title carries both surnames.
      - `"ambiguous_match"`: more than one event matches (rare; would
        need date disambiguation we don't currently carry on the row).
      - `"no_yes_market"`: event found but no YES market for the winner.
      - `"market_closed"`: market exists but isn't active / has no ask.
      - `"unparseable_players"`: couldn't extract both player names
        from the prediction row.
    """

    kind: Literal[
        "matched",
        "no_kalshi_match",
        "ambiguous_match",
        "no_yes_market",
        "market_closed",
        "unparseable_players",
    ]
    market: KalshiMarket | None = None
    event_ticker: str | None = None


def find_kalshi_match(
    row: PredictionRow, events: list[KalshiEvent],
) -> MatchOutcome:
    """Locate the Kalshi market for `row.predicted_winner`, or report why not.

    Two-step: surname-pair match against event titles → resolve YES
    market by winner surname within the matched event. Pure function;
    no IO. Caller is responsible for having pre-fetched the events
    list (typically via `KalshiClient.list_events`).
    """
    players = extract_match_players(row)
    if players is None:
        return MatchOutcome(kind="unparseable_players")
    winner_full, opp_full = players
    winner_last = last_token(winner_full)
    opp_last = last_token(opp_full)
    if not winner_last or not opp_last:
        return MatchOutcome(kind="unparseable_players")

    matches: list[KalshiEvent] = []
    for ev in events:
        if not ev.title:
            continue
        # Pure-winner-format guard: winner events use "{Last} vs {Last}"
        # with no extra punctuation. Composite-market events on the
        # same surface (e.g. `KXATPEXACTMATCH` exact-score predictions)
        # use "First Last vs First Last: <Market Type>". The colon is
        # the reliable distinguisher — reject those here so they can't
        # cause spurious ambiguity even if discovery accidentally
        # includes their series.
        if ":" in ev.title:
            continue
        title_norm = _normalize_name(ev.title)
        # Substring on whole-word boundaries via split — guards against
        # false positives like "tia" matching inside "tiafoe" when the
        # opp's surname is short.
        title_tokens = set(title_norm.split())
        if winner_last in title_tokens and opp_last in title_tokens:
            matches.append(ev)
    if not matches:
        return MatchOutcome(kind="no_kalshi_match")
    if len(matches) > 1:
        return MatchOutcome(kind="ambiguous_match")
    matched = matches[0]

    yes_markets: list[KalshiMarket] = []
    for mkt in matched.markets:
        if not mkt.yes_sub_title:
            continue
        if winner_last in _normalize_name(mkt.yes_sub_title).split():
            yes_markets.append(mkt)
    if not yes_markets:
        return MatchOutcome(
            kind="no_yes_market", event_ticker=matched.event_ticker,
        )
    if len(yes_markets) > 1:
        # Two YES markets matching the same surname inside one event
        # would mean Kalshi grouped two players sharing a surname (rare
        # but possible — e.g. Williams sisters). Treat as ambiguous.
        return MatchOutcome(
            kind="ambiguous_match", event_ticker=matched.event_ticker,
        )
    market = yes_markets[0]
    if (market.status or "").lower() != "active" or market.yes_ask_dollars is None:
        return MatchOutcome(
            kind="market_closed",
            market=market,
            event_ticker=matched.event_ticker,
        )
    return MatchOutcome(
        kind="matched",
        market=market,
        event_ticker=matched.event_ticker,
    )
