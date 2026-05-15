"""Pluggable tennis stats provider — Protocol + stub + factory.

Mirrors the shape of `agents/fetchers/base.py:FetcherProvider` and
`agents/fetchers/factory.py:build_provider`. The Protocol exists from
day one so the pipeline wiring, models, prompt threading, JSONL
persistence, and CLI plumbing can land before the concrete vendor
adapter is picked. The real adapter then drops in next to the stub
without touching anything outside this file + the factory.

Why a stub instead of just `None`:
- Keeps the pipeline's `enrich_tennis_stats` stage wired regardless of
  whether the user has an API key. Disabled = the stub runs, returns
  None per event, the pipeline behaves as before. No `if cfg.enabled`
  branches sprinkled through the orchestrator.
- Lets the smoke-test path verify the gate (sport detection, doubles
  filter) without hitting a network. Asserting "stub returns None for
  every event" is enough to cover the non-tennis/no-key paths.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol, Self

from skimsmarkets import config as cfg
from datetime import date, datetime

from skimsmarkets.tennis.identity import TennisMatchIdentity
from skimsmarkets.tennis.models import (
    PerMatchStats,
    TennisHeadToHead,
    TennisStatsContext,
)

log = logging.getLogger(__name__)


# Vendor score tokens look like "6-4", "7-6(3)" — the parenthetical is the
# tiebreak mini-score and is irrelevant for the games-total we compute.
_SCORE_TOKEN = re.compile(r"^(\d+)-(\d+)(?:\(\d+\))?$")
# Same shape as _SCORE_TOKEN but capturing the parenthetical so per-set
# parsing can flag tiebreaks. Stays separate so the games-total path
# above can stay tight.
_SCORE_TOKEN_WITH_TB = re.compile(r"^(\d+)-(\d+)(?:\((\d+)\))?$")
# Substrings that mark a score string as an aborted match. We search
# case-folded; the vendor mixes "ret.", "RET", "W/O", "Walkover" across
# different rows.
_ABORTED_MARKERS: tuple[str, ...] = ("ret", "w/o", "walkover", "def.")


def _parse_set_scores(score: str | None) -> tuple[int, int] | None:
    """Parse a vendor score string into total games per side `(p1, p2)`.

    Vendor format: space-separated sets, each `<a>-<b>` optionally
    suffixed with `(n)` for a tiebreak mini-score. Examples: `"6-4 6-3"`,
    `"7-6(3) 6-4"`, `"4-6 6-2 6-3"`. A trailing `ret.` / `RET` / `W/O` /
    `Walkover` / `def.` indicates an aborted match where the score line
    doesn't represent a completed match's shape — return None so the
    caller skips the row entirely from variance computation rather than
    treating the partial scoreline as ground truth.

    Empty / unparseable input → None. Sums across all parseable set
    tokens; tokens that don't match the regex are skipped silently
    (defensive — vendor occasionally ships oddities like a single dash
    placeholder for unfinished matches).
    """
    if not score:
        return None
    lower = score.lower()
    if any(marker in lower for marker in _ABORTED_MARKERS):
        return None
    p1_total = 0
    p2_total = 0
    found_any = False
    for tok in score.split():
        m = _SCORE_TOKEN.match(tok)
        if m is None:
            continue
        p1_total += int(m.group(1))
        p2_total += int(m.group(2))
        found_any = True
    if not found_any:
        return None
    return p1_total, p2_total


def _match_completeness(p1_games: int, p2_games: int) -> float | None:
    """Winner's share of total games, in `[0.5, 1.0]`.

    Symmetric in p1/p2 — we don't care which side won, only HOW
    completely the match resolved. A 6-0 6-0 dismantling and a 0-6 0-6
    dismantling both score 1.0; a 7-6 6-7 7-6 nailbiter scores ~0.51.
    None when no games were played (degenerate; shouldn't reach here
    because `_parse_set_scores` returns None for aborted matches).

    Used by the selector's variance estimator: a player whose recent
    matches all resolve at similar completeness levels is "predictable
    in shape" regardless of whether they win or lose; a player whose
    recent matches mix blowouts with nailbiters has higher contingency
    risk in any given matchup.
    """
    total = p1_games + p2_games
    if total <= 0:
        return None
    return max(p1_games, p2_games) / total


@dataclass(frozen=True)
class ScoreDetails:
    """Per-set facts derivable from a single score string.

    Used by both the live provider (career-aggregate clutch on
    `TennisPlayerStats`) and the GBT feature builder (career-aggregate
    clutch features). Same parsing semantics on both paths so a player's
    career tiebreak rate matches across the lens render and the model
    feature.

    `final_set_margin` is the absolute games gap in the LAST set. For a
    match decided by retirement / walkover the parser returns None
    upstream — this dataclass is only constructed for completed matches
    with at least one parseable set token.
    """

    n_sets: int
    went_to_decider: bool
    n_tiebreaks_played: int
    # Match-winner-relative tiebreak count. Aggregators rotate this onto
    # the subject's perspective via `subject_is_winner`: subject_tbs_won
    # = winner_tiebreaks_won if subject_is_winner else
    # (n_tiebreaks_played - winner_tiebreaks_won).
    winner_tiebreaks_won: int
    winner_won_set_one: bool
    final_set_margin: int
    is_close_match: bool


def parse_score_details(
    score: str | None, best_of: int | None, winner_side: int | None
) -> ScoreDetails | None:
    """Extract per-set facts from a vendor score string.

    Returns None on aborted / malformed scores (re-uses the same
    `_ABORTED_MARKERS` filter `_parse_set_scores` does). Returns None
    when `winner_side` isn't 1 or 2 — without a labelled winner we
    can't tell whether set 1 was a comeback or a closeout.

    `went_to_decider` defaults to False when `best_of` is missing,
    which is conservative: a match marked as a decider only when we
    KNOW it went the distance feeds cleaner training signal than one
    that guesses based on set count alone (a 3-set match could be
    bo3 final-set OR bo5 mid-match — though the latter is rare in
    completed-match data).

    `is_close_match`: final-set margin ≤2 OR final set was a tiebreak.
    Captures the matches where small-edge skill (clutch, BP-save under
    pressure) flips the outcome.
    """
    if not score or winner_side not in (1, 2):
        return None
    lower = score.lower()
    if any(marker in lower for marker in _ABORTED_MARKERS):
        return None
    sets_p1: list[int] = []
    sets_p2: list[int] = []
    tiebreak_set_indices: list[int] = []
    for tok in score.split():
        m = _SCORE_TOKEN_WITH_TB.match(tok)
        if m is None:
            continue
        sets_p1.append(int(m.group(1)))
        sets_p2.append(int(m.group(2)))
        if m.group(3) is not None:
            tiebreak_set_indices.append(len(sets_p1) - 1)
    if not sets_p1:
        return None
    n_sets = len(sets_p1)
    set1_winner_side = 1 if sets_p1[0] > sets_p2[0] else 2
    winner_won_set_one = set1_winner_side == winner_side
    final_p1, final_p2 = sets_p1[-1], sets_p2[-1]
    final_set_margin = abs(final_p1 - final_p2)
    final_set_was_tb = max(final_p1, final_p2) == 7 and min(final_p1, final_p2) == 6
    is_close_match = final_set_margin <= 2 or final_set_was_tb
    went_to_decider = best_of is not None and n_sets == best_of
    # Tiebreak-set winner is whichever side has more games in that set —
    # tiebreak sets always end 7-6, so the side with 7 won the TB.
    winner_tiebreaks_won = sum(
        1
        for i in tiebreak_set_indices
        if (sets_p1[i] > sets_p2[i] and winner_side == 1)
        or (sets_p2[i] > sets_p1[i] and winner_side == 2)
    )
    return ScoreDetails(
        n_sets=n_sets,
        went_to_decider=went_to_decider,
        n_tiebreaks_played=len(tiebreak_set_indices),
        winner_tiebreaks_won=winner_tiebreaks_won,
        winner_won_set_one=winner_won_set_one,
        final_set_margin=final_set_margin,
        is_close_match=is_close_match,
    )


@dataclass(frozen=True)
class MatchStatsFixture:
    """One scheduled fixture from `/tennis/v2/{tour}/fixtures/{date}`,
    parsed and indexed by surname-pair for the slate-overlay stage.

    Carries everything Phase 1 of the ITF data-recovery effort needs:
      - `date`: per-match scheduled tipoff (None for early-round
        matches whose schedule hasn't been confirmed yet).
      - `player_*_id`: vendor player IDs — load-bearing for ITF
        because most ITF futures players sit outside the top-500
        ATP/WTA rankings index. Seeding these IDs into the
        rankings cache (as a side effect of `fetch_fixtures_for_date`)
        unlocks every per-player MatchStats endpoint
        (`/player/profile`, `/player/past-matches`, `/h2h`, etc.)
        for ITF events.
      - `tournament_name`: specific name (e.g. `"W75 Trnava"`) —
        more precise than gamma's slug-derived hint (which collapses
        all ITF events under a single sport tag).
      - `surface`: court surface (`"Hard"` / `"Clay"` / `"Grass"` /
        `"Carpet"`) — currently None for ITF events because the
        slug-prefix surface map doesn't index ITF tournaments.
      - `round_name`: tournament round (e.g. `"First"`,
        `"Quarter-Final"`).

    Sourced from the fixtures endpoint with
    `include=tournament,tournament.court,round`. All optional
    fields fall back to None when the vendor payload omits them.
    """

    date: datetime | None
    player_a_id: int | None
    player_b_id: int | None
    player_a_name: str | None
    player_b_name: str | None
    tournament_name: str | None
    surface: str | None
    round_name: str | None


@dataclass(frozen=True)
class MatchHistoryRow:
    """One past-match row, parsed once for selector consumption.

    Internal-only — lives in the provider's in-memory cache, never
    persisted to the JSONL row. Pydantic isn't earning its keep here
    (no validation against external payload, no serialization), so a
    frozen dataclass keeps it light. Two consumers read these rows:

    1. `lookup_player_match_history` (pre-LLM selection) — reads
       `match_completeness` and `first_serve_win_pct` to compute a
       per-player consistency score.
    2. `_player_recent_matches` (full enrichment, post-cap) — converts
       to `TennisRecentMatch` for the prompt block, slicing to the
       first N rows.

    `match_completeness` and `first_serve_win_pct` are pre-computed at
    parse time so the selector's `_player_consistency_score` doesn't
    re-parse score strings on every call. None values represent rows
    where the underlying data is unusable (aborted matches, missing
    stat blocks); consumers filter these out at aggregation time.
    """

    date: date | None
    opponent_name: str
    won: bool
    raw_score: str | None
    surface: str | None
    round: str | None
    tournament_name: str | None
    tournament_tier: str | None
    match_completeness: float | None
    first_serve_win_pct: float | None
    # `best_of` is needed by `parse_score_details` to label deciders
    # correctly. Lifted from the row's top-level `best_of` (vendor ships
    # 3 or 5; missing on a few older rows).
    best_of: int | None
    # 1 or 2, indicating which side (relative to the row's p1/p2)
    # won the match. Needed so the clutch aggregator can call
    # `parse_score_details` with the score's native side convention. The
    # subject's `won` is independent (depends on subject-is-p1) and is
    # already carried.
    winner_side: int | None
    # Subject's BP-saved / BP-faced for THIS match. Derived from the
    # OPPONENT'S box score (subject's BPs faced == opponent's BP-convert
    # opportunities; subject's BPs saved == opponent's BP-convert
    # opportunities − opponent's BPs converted). Both None when the row
    # lacks a stat block (live-suspended matches and very old rows).
    bp_saved: int | None
    bp_faced: int | None


class TennisStatsProvider(Protocol):
    """Async-context-managed vendor adapter.

    `name` is persisted into `TennisStatsContext.provider` and the JSONL
    row so retro grading can group hit-rate by vendor (mirroring how
    `FetcherProvider.name` rides on every notebook).

    `fetch` returns `None` when the vendor has no record for this match
    or the call fails — the pipeline degrades silently per match. The
    Protocol does NOT define a separate `enabled` property because the
    factory is the only place that decides which provider to instantiate;
    once a provider exists, the orchestrator just calls `fetch` on
    every tennis match identity it finds.
    """

    name: str

    async def fetch(
        self, identity: TennisMatchIdentity
    ) -> TennisStatsContext | None: ...

    async def warm_for_selection(self, tours: Iterable[str]) -> None:
        """Pre-warm any caches needed for fast synchronous rank lookups.

        Called by the pre-LLM selection stage when it needs to score
        events by player-rank delta. Real adapters paginate their
        rankings indexes here so subsequent `lookup_player_rank` calls
        are O(1) dict hits with no HTTP. Stub providers no-op.

        Idempotent — repeat calls for the same tour are free.
        """
        ...

    def lookup_player_rank(
        self, tour: str, name: str
    ) -> tuple[int, int] | None:
        """Synchronous rank lookup against any pre-warmed index.

        Returns `(rank_position, rank_points)` when both are known, or
        None when the player isn't in the index (or the index isn't
        warmed). Selection scoring uses both fields: `points` gives a
        better skill-gap proxy than `position` (the ATP/WTA points
        spread is non-linear in rank), and falls back to `position`
        when points are absent.
        """
        ...

    async def warm_form_for_selection(
        self, identities: Iterable[TennisMatchIdentity]
    ) -> None:
        """Pre-fetch player profile (form + best_rank) for each player
        named in `identities`. Selection scoring uses
        `lookup_player_form` afterwards as a synchronous cache hit.

        Idempotent — players already cached skip re-fetching. Real
        adapters fan out one HTTP per unique (tour, player) pair under
        their own concurrency cap; stubs no-op.
        """
        ...

    def lookup_player_form(
        self, tour: str, name: str
    ) -> tuple[str, int | None] | None:
        """Synchronous lookup of `(last_10_form_string, best_rank)`
        from the warmed profile cache.

        Returns None when the form data isn't available — either the
        player wasn't in the rankings index, `warm_form_for_selection`
        wasn't called for their identity, or the vendor returned no
        recent matches. Selection callers treat None as "no form
        signal" and skip the alignment adjustment.
        """
        ...

    async def warm_match_history_for_selection(
        self, identities: Iterable[TennisMatchIdentity]
    ) -> None:
        """Pre-fetch per-match history (with stat blocks) for every player
        named in `identities`. Selection scoring uses
        `lookup_player_match_history` afterwards as a synchronous cache
        hit to compute a consistency / variance metric.

        Companion to `warm_form_for_selection`: same dedup-by-unique-
        player pattern, separate HTTP because the per-match-stat payload
        is heavier than the profile (`form` + bio) data and we don't
        want to bloat every profile fetch for callers that only need
        form. The cached payload is also reused by `_player_recent_matches`
        in the post-cap enrichment path, so the warmup HTTP is not
        wasted on events that survive selection.

        Idempotent — players already cached skip re-fetching.
        """
        ...

    def lookup_player_match_history(
        self, tour: str, name: str
    ) -> list[MatchHistoryRow] | None:
        """Synchronous lookup of per-match history rows from the warmed
        cache.

        Returns None when:
          - the player isn't in the rankings index for this tour
          - `warm_match_history_for_selection` hasn't been called yet
          - the vendor returned no recent matches for this player
        Callers (selection scoring) treat None as "no consistency
        signal" and skip the consistency adjustment, falling back to
        base + form.

        Rows are returned in the vendor's order (newest first); callers
        that care about chronology can resort by `row.date`.
        """
        ...

    async def warm_surface_summary_for_selection(
        self, identities: Iterable[TennisMatchIdentity]
    ) -> None:
        """Pre-fetch per-player surface summary (YTD record + per-surface
        win/loss splits) for every player named in `identities`.
        Selection scoring reads the cache via
        `lookup_player_surface_record` to compute a surface-specialism
        adjustment.

        Companion to `warm_form_for_selection` and
        `warm_match_history_for_selection`. One HTTP per unique
        player. The cached payload is reused by
        `_player_surface_year_record` in the post-cap enrichment path
        so the warmup HTTP isn't wasted on events that survive
        selection. Idempotent.
        """
        ...

    def lookup_player_surface_record(
        self, tour: str, name: str
    ) -> tuple[
        tuple[int, int] | None, dict[str, tuple[int, int]] | None
    ] | None:
        """Synchronous lookup of `(ytd_record, surface_dict)` from the
        warmed surface cache.

        `ytd_record` is `(wins, losses)` aggregated across surfaces
        for the most recent year. `surface_dict` maps a surface key
        ("hard"/"clay"/"grass"/"carpet") to its `(wins, losses)`.
        Returns None when the player isn't in the rankings index or
        the warmup hasn't run for them. The outer return is None for
        cache miss; the inner pair components are independently None
        when the vendor returned an empty section.
        """
        ...

    def lookup_player_profile_extras(
        self, tour: str, name: str
    ) -> tuple[int | None, int | None] | None:
        """Synchronous lookup of `(age_years, best_rank)` from the
        warmed profile cache.

        Returns None when the player isn't in the rankings index or
        `warm_form_for_selection` hasn't been called for their
        identity. Either component is independently None when the
        vendor's profile response was missing the underlying field
        (no birthdate, no career-high rank).

        Used by selection scoring's age + career-trajectory
        adjustment. The values come from the SAME cache slot
        `lookup_player_form` reads — no extra HTTP, just a different
        view of the warmed profile.
        """
        ...

    async def warm_h2h_for_selection(
        self, identities: Iterable[TennisMatchIdentity]
    ) -> None:
        """Pre-fetch head-to-head context per matchup for every identity
        in the slate. Selection scoring reads the cache via `lookup_h2h`
        to compute an H2H sample-size + surface-conditioned bonus.

        Same idempotent dedup-and-cache pattern as the four other
        warmups, but keyed by **matchup pair** rather than by individual
        player — one HTTP per unique (tour, player_a_id, player_b_id)
        triple, ordered by identity convention so subsequent runtime
        fetches in `enrich_tennis_stats` post-cap re-use the cached
        payload for free on cap-survivor events.

        Pre-condition: rankings index warm. Identities where either
        player doesn't resolve are silently skipped (the H2H endpoint
        requires both vendor IDs).
        """
        ...

    def lookup_h2h(
        self, tour: str, name_a: str, name_b: str
    ) -> TennisHeadToHead | None:
        """Synchronous H2H lookup against the warmed cache.

        Identity-ordered: callers MUST pass `name_a` and `name_b` in
        the same order as the original `TennisMatchIdentity` (i.e. the
        order used at warmup time). Reordering bypasses the cache key.

        Returns None when:
          - either player isn't in the rankings index
          - `warm_h2h_for_selection` hasn't been called for this matchup
          - the warmup ran but the vendor returned no usable H2H rows
            (no prior meetings AND no matchup-conditioned stats)
        Selection callers treat None as "no H2H signal" — no penalty,
        no bonus.
        """
        ...

    async def fetch_post_match_stats(
        self,
        tour: str,
        player_id: int,
        on_date: date,
        opponent_name: str,
    ) -> PerMatchStats | None:
        """Pull per-match box-score stats for a single completed match.

        Used by the retro / self-improvement layer (NOT the live pipeline)
        to fetch the actual first-serve %, BP convert %, etc. that a
        player produced on a specific date — for comparison against the
        career baseline that was on `TennisPlayerStats` at prediction
        time. Match-row identification: `on_date` plus opponent name
        (case-insensitive, diacritic-stripped). Returns None when the
        vendor has no row for that date or the row's stats block is
        empty (live-suspended / walkover).
        """
        ...

    async def fetch_fixtures_for_date(
        self, *, tour: str, date_iso: str
    ) -> dict[frozenset[str], MatchStatsFixture]:
        """Pull scheduled fixtures + tournament metadata for one
        (tour, date) from the vendor.

        Returns a dict mapping `frozenset({surname_a, surname_b})` →
        `MatchStatsFixture` carrying tipoff, player IDs, tournament
        name, surface, and round. Used by the slate-build path
        (`pipeline.overlay_matchstats_tipoffs`) to:
          1. Overlay per-match tipoff precision onto Kalshi's
             session-bucketed `occurrence_datetime`.
          2. Seed the rankings index with fixture-derived player IDs
             so ITF players outside the top-500 ATP/WTA rankings
             still resolve through `_resolve(tour, name)` and unlock
             the per-player stats endpoints.

        Surnames are normalised via
        `tennis/matchstat.py:_normalize_name` last-token so the index
        matches the surname pair `pipeline._event_surname_pair_candidates` reads
        off the gamma event slug.

        `date_iso` is `YYYY-MM-DD`. Empty dict on any failure or when
        the vendor has no fixtures for that date — the overlay
        degrades silently per missing match. Concrete adapters MAY
        also seed their internal rankings index as a side effect.
        """
        ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


class StubTennisStatsProvider:
    """No-op provider. Always returns `None`.

    Used when no vendor is configured — the pipeline still gets a
    well-typed object to call `fetch` on, every event ends up with
    `tennis_stats=None`, and the rest of the pipeline behaves
    identically to a run with this enrichment disabled. The first real
    adapter (when an API key is added) lives in a sibling file and the
    factory picks it over this one.
    """

    name = "stub"

    async def fetch(
        self, identity: TennisMatchIdentity
    ) -> TennisStatsContext | None:
        log.debug(
            "tennis stub: skipping %s vs %s (%s)",
            identity.player_a,
            identity.player_b,
            identity.tour,
        )
        return None

    async def warm_for_selection(self, tours: Iterable[str]) -> None:
        # No backing index — selection scoring will see None for every
        # `lookup_player_rank` and fall back to other signals (team
        # records, tipoff). Iterate `tours` only for signature symmetry.
        for _ in tours:
            pass

    def lookup_player_rank(
        self, tour: str, name: str
    ) -> tuple[int, int] | None:
        return None

    async def warm_form_for_selection(
        self, identities: Iterable[TennisMatchIdentity]
    ) -> None:
        # No backing cache — selection scoring will see None for every
        # `lookup_player_form` and fall back to the points-only base
        # score. Iterate for signature symmetry.
        for _ in identities:
            pass

    def lookup_player_form(
        self, tour: str, name: str
    ) -> tuple[str, int | None] | None:
        return None

    async def warm_match_history_for_selection(
        self, identities: Iterable[TennisMatchIdentity]
    ) -> None:
        # No backing cache — selection scoring will see None for every
        # `lookup_player_match_history` and fall back to base + form.
        for _ in identities:
            pass

    def lookup_player_match_history(
        self, tour: str, name: str
    ) -> list[MatchHistoryRow] | None:
        return None

    async def warm_surface_summary_for_selection(
        self, identities: Iterable[TennisMatchIdentity]
    ) -> None:
        # No backing cache — selection scoring will see None for every
        # `lookup_player_surface_record` and skip the surface tier.
        for _ in identities:
            pass

    def lookup_player_surface_record(
        self, tour: str, name: str
    ) -> tuple[
        tuple[int, int] | None, dict[str, tuple[int, int]] | None
    ] | None:
        return None

    def lookup_player_profile_extras(
        self, tour: str, name: str
    ) -> tuple[int | None, int | None] | None:
        return None

    async def warm_h2h_for_selection(
        self, identities: Iterable[TennisMatchIdentity]
    ) -> None:
        # No backing cache — selection scoring will see None for every
        # `lookup_h2h` and skip the H2H tier.
        for _ in identities:
            pass

    def lookup_h2h(
        self, tour: str, name_a: str, name_b: str
    ) -> TennisHeadToHead | None:
        return None

    async def fetch_post_match_stats(
        self,
        tour: str,
        player_id: int,
        on_date: date,
        opponent_name: str,
    ) -> PerMatchStats | None:
        return None

    async def fetch_fixtures_for_date(
        self, *, tour: str, date_iso: str
    ) -> dict[frozenset[str], MatchStatsFixture]:
        # No backing vendor — the overlay stage falls through to the
        # existing Kalshi `occurrence_datetime` for every market.
        return {}

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


def build_tennis_provider(config: cfg.Config) -> TennisStatsProvider:
    """Pick a `TennisStatsProvider` based on config.

    Decision tree:
      1. `--no-tennis-stats` (`config.tennis_stats_disabled=True`) → stub,
         even when a key is present. Useful for token-cost A/B compares
         and for forcing pipelines through the stub path during
         debugging.
      2. No `tennis_stats_api_key` set → stub. Mirrors UW posture: an
         unset key silently disables the enrichment, never raises.
      3. Otherwise → MatchStat adapter (`tennisapidoc.matchstat.com`,
         RapidAPI-hosted). Adding a second concrete provider later
         means a third branch here, picking by an explicit
         `tennis_stats_provider` config field.

    The real-provider import is local to this branch so a stub-only
    run doesn't pay the import cost (same pattern as the fetcher
    factory).
    """
    if getattr(config, "tennis_stats_disabled", False):
        return StubTennisStatsProvider()
    api_key = getattr(config, "tennis_stats_api_key", None)
    if not api_key:
        return StubTennisStatsProvider()
    from skimsmarkets.tennis.matchstat import MatchStatTennisProvider

    return MatchStatTennisProvider(api_key=api_key)
