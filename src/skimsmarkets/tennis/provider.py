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
from collections.abc import Iterable
from types import TracebackType
from typing import Protocol, Self

from skimsmarkets import config as cfg
from skimsmarkets.tennis.identity import TennisMatchIdentity
from skimsmarkets.tennis.models import TennisStatsContext

log = logging.getLogger(__name__)


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
