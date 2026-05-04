"""Pydantic shapes for tennis player-stats enrichment.

Same posture as `unusual_whales/models.py:UnusualWhalesContext`:
- Structurally compact — only fields the statistics specialist can
  actually consume in a prompt; we don't mirror the vendor's full payload.
- `has_actionable_signal()` lets the renderer skip cleanly when the
  fetched context is too thin to be worth the prompt-token cost.
- All numeric fields are `| None` so a partial fetch (vendor missing a
  player record) is representable rather than triggering validation
  errors that would drop the whole context.

Field naming uses no source prefix because there is one tennis-stats
source per event — unlike the polymarket models where `gamma_*` /
`clob_*` distinguish two providers feeding the same record.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


def _coerce_date(v: Any) -> Any:
    """Tolerant date parser for vendor payloads.

    Tennis APIs ship dates in mixed shapes (ISO strings, epoch seconds,
    or already-parsed `date` objects). Mirrors the `_coerce_dt` helper
    in `unusual_whales/models.py` but for `date` rather than `datetime`.
    """
    if v is None or v == "":
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        # Accept ISO date or ISO datetime; trim time portion for the
        # latter so callers can pass either shape.
        try:
            return date.fromisoformat(s[:10])
        except ValueError:
            return None
    return None


class TennisPlayerStats(BaseModel):
    """Per-player snapshot the statistics specialist consumes verbatim.

    Every numeric field is optional because vendors differ in what they
    expose: rapid-API-style indexes have ranking + form but no surface
    splits, while tennisabstract-style sources have rich surface splits
    but no live ranking points. The reasoner is told to use whatever is
    populated and ignore absent fields.

    Career serve / return / break-point percentages come from a single
    extra vendor call (`/player/match-stats/{id}`). They're CAREER
    aggregates — not surface-conditioned, not last-N-match — but the
    relative ordering between two players (e.g. 70% vs 62% first-serve
    win) is itself a load-bearing matchup signal even at the career
    timeframe. Surface-conditioned form lives in `surface_win_loss`.
    """

    model_config = ConfigDict(extra="ignore")

    # Echoed verbatim from `_parse_h2h_question` so the reasoner can match
    # the player back to `team_a_name` / `team_b_name` in the event
    # context without a second normalization pass.
    name: str
    # Vendor-specific canonical id, kept for retro grading ("did the API
    # actually find this player?") rather than for the prompt — providers
    # often need the id internally to fetch H2H. None when the lookup
    # was fuzzy-name only.
    api_player_id: str | None = None
    rank_singles: int | None = None
    rank_points: int | None = None
    # Career-high singles ranking. Useful context: a current rank=80
    # player with best_rank=4 (descending veteran or comeback) reads
    # very differently from a rank=80 player with best_rank=78 (stable
    # journeyman). Free to populate — comes on the same profile call as
    # current rank.
    best_rank_singles: int | None = None
    # `(wins, losses)` over a vendor-defined window — typically YTD or
    # trailing 52 weeks. Provider docs the window in its rendering tail.
    ytd_win_loss: tuple[int, int] | None = None
    # Surface keys: `"hard"`, `"clay"`, `"grass"`, `"carpet"`. Values are
    # `(wins, losses)`. Surface-conditioned form is more predictive than
    # YTD aggregates per the existing tennis-statistics sport hint.
    surface_win_loss: dict[str, tuple[int, int]] | None = None
    # Compact "WWLWWLWWWL" string — last N matches in chronological order
    # (oldest → newest). N is vendor-defined; we don't try to enforce.
    last_10_form: str | None = None
    last_match_date: date | None = None
    # Career serve metrics. All four percentages live in [0.0, 1.0] —
    # the renderer multiplies by 100 for display. The vendor ships raw
    # counters (numerator + denominator) which the provider divides
    # before persisting; we keep ratios rather than counts so the prompt
    # is human-readable without further math.
    first_serve_in_pct: float | None = None
    first_serve_win_pct: float | None = None
    second_serve_win_pct: float | None = None
    # Break-point save % (when serving) and break-point conversion %
    # (when returning). The two together are the canonical "is this
    # player clutch?" pair the tennis sport hint already flags as
    # decisive in tight matches.
    break_point_save_pct: float | None = None
    break_point_convert_pct: float | None = None
    # Current-year W/L vs elite competition and at the biggest events.
    # Sourced from `/player/perf-breakdown` (the vendor's year × tier
    # matrix). These three rows are the load-bearing slice of an
    # otherwise-massive payload — together they answer "does this player
    # show up against good opponents and at the big stages?". Other
    # rows (`top1`, `top5`, futures, challengers, mainTour) are
    # available on the same response but compete for prompt space; we
    # pick the three with the cleanest signal-per-token.
    record_vs_top_10: tuple[int, int] | None = None
    record_at_grand_slam: tuple[int, int] | None = None
    record_at_masters: tuple[int, int] | None = None

    @field_validator("last_match_date", mode="before")
    @classmethod
    def _d(cls, v: Any) -> Any:
        return _coerce_date(v)


class TennisHeadToHead(BaseModel):
    """Head-to-head history between the two players in this match.

    The matchup-specific clutch records (decider, tiebreak, comeback)
    come from a separate `/h2h/stats` vendor call and are aggregated
    across ALL prior meetings between these two players. They're
    materially different from career averages — a player who's 67% in
    deciders overall might be 33% in deciders specifically against this
    opponent. The reasoner is told to weight these matchup-conditioned
    numbers above career averages when both are present.
    """

    model_config = ConfigDict(extra="ignore")

    a_wins: int = 0
    b_wins: int = 0
    last_meeting: date | None = None
    last_meeting_winner: str | None = None
    last_meeting_surface: str | None = None
    # Round name of the last meeting (e.g. "Final", "1/2", "1/4") — pass-
    # through from the vendor. Adds context that "they last played in a
    # Slam final" reads differently from "they last played a R128
    # qualifier match." Free to populate; comes on the same h2h/matches
    # call we already make.
    last_meeting_round: str | None = None
    # Score of the last meeting (e.g. "6-4 6-2", "7-6(5) 6-3 4-6 6-2").
    # Tells the reasoner whether the match was tight or dominant — a
    # straight-sets win means a different "form" signal than a
    # five-setter the same player won.
    last_meeting_result: str | None = None
    # Matchup-conditioned clutch records. Each is `(wins, total)` so the
    # reasoner sees the sample size (a 1-1 deciding-set record carries
    # less weight than 5-2). All four can be None when the matchup has
    # no prior meetings or when /h2h/stats was unavailable. The "a" /
    # "b" suffix matches the player_a / player_b convention used
    # everywhere else in the schema.
    decider_record_a: tuple[int, int] | None = None
    decider_record_b: tuple[int, int] | None = None
    tiebreak_record_a: tuple[int, int] | None = None
    tiebreak_record_b: tuple[int, int] | None = None
    # "Comeback rate when losing the first set" specifically against this
    # opponent. Captured for both players so the reasoner can see e.g.
    # "Alcaraz comes back from set 1 down 50% of the time vs Djokovic"
    # alongside the symmetric figure. Useful in best-of-5 contexts.
    first_set_lost_match_won_pct_a: float | None = None
    first_set_lost_match_won_pct_b: float | None = None

    @field_validator("last_meeting", mode="before")
    @classmethod
    def _d(cls, v: Any) -> Any:
        return _coerce_date(v)


class TennisStatsContext(BaseModel):
    """Compact per-match tennis stats blob attached to a `PolymarketEvent`.

    Built by the active `TennisStatsProvider` for events where
    `tennis_match_identity` returns a non-None identity. Lives on
    `event.tennis_stats` (None when the gate failed, the vendor is down,
    or the API returned no usable record).
    """

    model_config = ConfigDict(extra="ignore")

    # Vendor name (e.g. `"stub"`, `"rapid_api_tennis"`). Persisted to the
    # JSONL row so retro grading can group hit-rate by vendor the same
    # way `fetcher_provider` does for the search providers.
    provider: str
    fetched_at: datetime
    # Tournament + surface come from the vendor when available; fall back
    # to None and let the reasoner read the surface out of the event
    # context's existing tennis sport hint when the vendor doesn't say.
    surface: str | None = None
    tournament: str | None = None
    player_a: TennisPlayerStats
    player_b: TennisPlayerStats
    head_to_head: TennisHeadToHead | None = None

    def has_actionable_signal(self) -> bool:
        """True iff the fetched context has anything worth prompting on.

        Drops contexts that came back populated-but-empty — every numeric
        field None, no H2H, just two echoed names. Letting those through
        would cost prompt tokens for no information.

        Threshold: at least one of (a) any ranking populated on either
        player, (b) any surface or YTD split, (c) any non-zero H2H. UW's
        `has_actionable_signal` filter applies the same posture for flow
        signals.
        """
        for player in (self.player_a, self.player_b):
            if player.rank_singles is not None:
                return True
            if player.ytd_win_loss is not None:
                return True
            if player.surface_win_loss:
                return True
            if player.last_10_form:
                return True
        h2h = self.head_to_head
        if h2h is not None and (h2h.a_wins > 0 or h2h.b_wins > 0):
            return True
        return False
