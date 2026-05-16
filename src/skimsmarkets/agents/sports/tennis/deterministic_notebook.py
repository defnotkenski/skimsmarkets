"""Fetcher-bypass deterministic notebook builders for tennis lenses.

When the structured `tennis_stats` block on an event is rich enough
that the fetcher would be pure augmentation (the AUTHORITATIVE fields
the fetcher prompt enumerates are all populated), this module builds
a placeholder `LensNotebook` and skips the fetcher LLM call entirely.
The reasoner — now equipped with `code_execution` + `web_search`
tools (see `agents/reasoners.py`) — runs against the structured data
rendered into its user message via `render_extras`, and reaches for
its own tools only when a gap remains (loss color, recency commentary,
small-sample CI validation).

The notebook doesn't need to re-render the structured data because
`render_extras` already produces the per-lens rendered block as a
user-message append. So the notebook is a thin placeholder whose
`research_notes` explains the bypass posture and `coverage='rich'`
signals the reasoner not to downgrade confidence.

Returns `None` when data isn't rich — the pipeline falls through to
the existing fetcher → reasoner path in that case.

Gated by `cfg.FETCHER_BYPASS_ON_RICH_DATA` (default False) inside the
pipeline — these builders are always wired on the LensSpec; the flag
controls whether they're actually consulted.
"""

from __future__ import annotations

from skimsmarkets.agents.schemas import LensNotebook
from skimsmarkets.polymarket.models import PolymarketEvent
from skimsmarkets.tennis.models import TennisPlayerStats, TennisStatsContext


# ---------------------------------------------------------------------------
# Coverage detectors — encode the "rich enough" threshold per lens. Both
# require the load-bearing fields the fetcher prompt lists as
# AUTHORITATIVE to be populated for BOTH players. Surface coverage is
# checked against the match's surface specifically so a player without
# clay history on a clay match is treated as thin even if their hard
# split is complete.
# ---------------------------------------------------------------------------


def _form_surface_player_is_rich(
    p: TennisPlayerStats, surface: str | None
) -> bool:
    """Per-player rich-coverage check for form_and_surface.

    Required: rank, YTD W-L, last-10 form, all 3 career serve %, and
    surface W-L for THIS surface. Career return rates and tier records
    are nice-to-have but not gating (the fetcher prompt doesn't list
    them as universally present).
    """
    if p.rank_singles is None:
        return False
    if p.ytd_win_loss is None:
        return False
    if not p.last_10_form:
        return False
    if p.first_serve_in_pct is None:
        return False
    if p.first_serve_win_pct is None:
        return False
    if p.second_serve_win_pct is None:
        return False
    # Surface coverage for THIS surface — a player can be rich overall
    # but thin on the match's specific surface, in which case the
    # fetcher's web-search-for-surface-trajectory is still load-bearing.
    if surface and (
        p.surface_win_loss is None
        or surface not in p.surface_win_loss
    ):
        return False
    return True


def _matchup_clutch_player_is_rich(p: TennisPlayerStats) -> bool:
    """Per-player rich-coverage check for matchup_and_clutch.

    Required: handedness (`plays`), career BP-save + BP-convert, and
    at least one career-aggregate clutch record (decider, tiebreak,
    comeback, or close-match) populated. H2H is checked separately at
    the context level — it's a per-pair signal, not per-player.
    """
    if p.plays is None:
        return False
    if p.break_point_save_pct is None:
        return False
    if p.break_point_convert_pct is None:
        return False
    has_any_clutch = (
        p.career_decider_record is not None
        or p.career_tiebreak_record is not None
        or p.career_comeback_record is not None
        or p.career_close_match_record is not None
    )
    if not has_any_clutch:
        return False
    return True


def _form_surface_is_rich(ts: TennisStatsContext) -> bool:
    return (
        _form_surface_player_is_rich(ts.player_a, ts.surface)
        and _form_surface_player_is_rich(ts.player_b, ts.surface)
    )


def _matchup_clutch_is_rich(ts: TennisStatsContext) -> bool:
    """Matchup-rich requires both players' clutch fields AND a usable
    H2H signal — either prior meetings (the fetcher's primary signal)
    OR a confirmed first-time-meeting state (head_to_head present but
    with zero counts is itself informative; head_to_head=None is the
    thin case where the fetcher needs to look up whether they've ever
    played).
    """
    if not (
        _matchup_clutch_player_is_rich(ts.player_a)
        and _matchup_clutch_player_is_rich(ts.player_b)
    ):
        return False
    if ts.head_to_head is None:
        return False
    return True


# ---------------------------------------------------------------------------
# Team-name resolution — the LensNotebook schema requires both names.
# Mirrors the algo_lens approach (favorite-pick from market implied
# probabilities) so the deterministic path uses the same convention the
# fetcher would have populated.
# ---------------------------------------------------------------------------


def _resolve_team_names(event: PolymarketEvent) -> tuple[str, str] | None:
    candidates = [
        (m.yes_implied_probability or -1.0, m)
        for m in event.markets
        if m.yes_sub_title
    ]
    if len(candidates) < 2:
        return None
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    team_a = candidates[0][1].yes_sub_title
    team_b = next(
        (m.yes_sub_title for _, m in candidates[1:] if m.yes_sub_title),
        None,
    )
    if team_a is None or team_b is None:
        return None
    return team_a, team_b


# ---------------------------------------------------------------------------
# Notebook builders — return a placeholder LensNotebook when rich,
# None otherwise (caller falls through to fetcher).
# ---------------------------------------------------------------------------


_FORM_SURFACE_BYPASS_NOTES = """
Fetcher bypassed: structured tennis_stats block carries the authoritative
form/surface signals — rank, YTD W-L, surface-conditioned W-L on this
match's surface, last-10 form, career serve metrics. The rendered block
is in your user message extras (look for `--- Tennis form & surface
(vendor: ...) ---`); lift numbers from there directly.

Tool guidance for this bypass call:
- web_search: ONLY when the form string flags a recent loss whose quality
  matters (was a 6-3 6-2 a tight scrap or a bagel?) or when a layoff /
  training-block disruption is suspected from days_since_last_match.
  Skip web_search for numerical data the block already covers.
- code_execution: validate a baseline candidate (e.g. compute
  surface-conditioned Log5 from surface_win_loss, recency-weighted
  baseline from recent_matches) when the primary baseline is thin-sample.
- Default: NO tool calls. The structured data is sufficient for the
  baseline + form/surface shifts the schema requires.
""".strip()


_MATCHUP_CLUTCH_BYPASS_NOTES = """
Fetcher bypassed: structured tennis_stats block carries the authoritative
matchup/clutch signals — H2H counts (total + per-surface), handedness,
career BP-save / BP-convert percentages, career-aggregate clutch records
(deciders, tiebreaks, comeback, close-match), and any in-matchup
records that exist. The rendered block is in your user message extras
(look for `--- Tennis matchup & clutch (vendor: ...) ---`); lift numbers
from there directly.

Tool guidance for this bypass call:
- web_search: ONLY for tactical commentary the block can't carry — beat-
  reporter takes on past meetings, style-fit notes (does the lefty's
  forehand cross over to opponent's one-handed backhand?), choke-history
  threads. Skip web_search for H2H counts the block already lists.
- code_execution: compute a binomial CI on a small-sample H2H or
  in-matchup decider rate when the denominator is <10 and the prose
  needs to caveat. Compute career vs in-matchup deltas to surface
  matchup-specific regression.
- Default: NO tool calls. The structured matchup block is sufficient
  for the h2h_signed_shift + clutch_signed_shift the schema requires.
""".strip()


def build_form_surface_deterministic_notebook(
    event: PolymarketEvent,
) -> LensNotebook | None:
    """Build a placeholder `LensNotebook` for tennis_form_and_surface
    when the structured block is rich. Returns None to fall through to
    the LLM fetcher path otherwise.

    The reasoner consumes this notebook plus the rendered structured
    block (via `_render_tennis_form_extras` → `render_extras`), with
    `web_search` + `code_execution` available for the rare cases where
    a gap remains.
    """
    if event.tennis_stats is None:
        return None
    if not _form_surface_is_rich(event.tennis_stats):
        return None
    names = _resolve_team_names(event)
    if names is None:
        return None
    team_a, team_b = names
    return LensNotebook(
        lens="tennis_form_and_surface",
        team_a_name=team_a,
        team_b_name=team_b,
        research_notes=_FORM_SURFACE_BYPASS_NOTES,
        citations=[],
        computed_numbers=[],
        coverage="rich",
    )


def build_matchup_clutch_deterministic_notebook(
    event: PolymarketEvent,
) -> LensNotebook | None:
    """Build a placeholder `LensNotebook` for tennis_matchup_and_clutch
    when the structured block is rich. Returns None to fall through to
    the LLM fetcher path otherwise.
    """
    if event.tennis_stats is None:
        return None
    if not _matchup_clutch_is_rich(event.tennis_stats):
        return None
    names = _resolve_team_names(event)
    if names is None:
        return None
    team_a, team_b = names
    return LensNotebook(
        lens="tennis_matchup_and_clutch",
        team_a_name=team_a,
        team_b_name=team_b,
        research_notes=_MATCHUP_CLUTCH_BYPASS_NOTES,
        citations=[],
        computed_numbers=[],
        coverage="rich",
    )


def is_tennis_event_rich_coverage(event: PolymarketEvent) -> bool:
    """True iff the event's structured `tennis_stats` block clears the
    rich-coverage bar for BOTH tennis lenses (form_surface AND
    matchup_clutch). Used by the optional `--require-rich-stats` slate
    filter to drop tennis events with thin MatchStats coverage before
    the LLM stage.

    Non-tennis events return True unconditionally — the filter is
    tennis-specific by design (other sports have different vendor
    coverage shapes). Tennis events with `tennis_stats=None` (vendor
    miss, ITF coverage gap, cold-start) return False.

    Combines both lens checks rather than just the form lens because
    "rich coverage" for the whole event means BOTH lenses can bypass
    the fetcher; if only form is rich, matchup_clutch still pays the
    fetcher LLM cost which weakens the filter's "quality + cost" pitch.
    """
    if event.sport_type != "tennis":
        return True
    if event.tennis_stats is None:
        return False
    return (
        _form_surface_is_rich(event.tennis_stats)
        and _matchup_clutch_is_rich(event.tennis_stats)
    )


__all__ = [
    "build_form_surface_deterministic_notebook",
    "build_matchup_clutch_deterministic_notebook",
    "is_tennis_event_rich_coverage",
]
