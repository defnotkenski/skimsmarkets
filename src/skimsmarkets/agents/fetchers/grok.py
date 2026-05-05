"""Grok fetcher provider — xAI SDK with `agent_count=4` multi-agent ensemble.

Each lens runs as one xAI chat with `web_search` + `x_search` +
`code_execution` tools available; `agent_count=4` fans the search out
across 4 independent trajectories that the SDK merges into one structured
response. The 4× search-path diversity is what we get from xAI's native
ensemble — Gemini has no equivalent primitive, so the parallel
`GeminiProvider` runs single-pass and the A/B is "Grok-with-its-native-
ensemble vs Gemini-single-pass" by design.

Per-sport-lens-set refactor: prompts are pre-built per `(sport, lens)` at
construction by iterating `SPORT_LENS_SETS`. `_TOOLS_BY_LENS` is the
provider-owned, FLAT dict of per-lens tool prose — keyed by full lens
name (not by `(sport, lens)`) because lens names are unique across the
registry. Adding a new sport adds N entries to `_TOOLS_BY_LENS` (one
per new lens) without touching existing entries.
"""

from __future__ import annotations

import logging
from typing import Literal

from xai_sdk import AsyncClient as XAIAsyncClient
from xai_sdk.chat import system, user
from xai_sdk.tools import code_execution, web_search, x_search

from skimsmarkets.agents.fetchers.base import (
    assert_lens_match,
    build_lens_prompts_for_set,
    render_user_message_for_lens,
)
from skimsmarkets.agents.schemas import LensNotebook
from skimsmarkets.agents.sports import SPORT_LENS_SETS
from skimsmarkets.agents.sports.base import LensSet
from skimsmarkets.polymarket.models import PolymarketEvent

log = logging.getLogger(__name__)

GROK_MODEL = "grok-4.20-multi-agent-0309"
# xAI's `agent_count` is typed `Literal[4, 16]` — the SDK only accepts those
# two ensemble sizes. Annotate explicitly so static checkers narrow correctly
# at the `chat.create(...)` call site.
GROK_AGENT_COUNT: Literal[4] = 4


# Generic notebook tail — output rules + tool list naming the xAI tools by
# their actual SDK names (`web_search`, `x_search`, `code_execution`). Every
# lens prompt closes with this block.
NOTEBOOK_TAIL_GROK = """
You are a FETCHER, not a reasoner. Your job is evidence capture — not judgment.
Do NOT output a probability, a signed shift, a directional verdict, or a single
"team_a will probably win" sentence. The downstream reasoner does that.

Tools available — use whichever fit what you're trying to learn, and chain several calls if
the first doesn't answer the question:
- web_search: URL-citable facts — stats pages, official injury reports, press coverage,
  weather, venue.
- x_search: breaking news, beat-reporter leaks, team/player accounts, public sentiment —
  usually the fastest channel for anything <24h old.
- code_execution: run Python when numbers need computing — converting ratings to
  probabilities, weighting recent-form vs season baselines, computing weather-impact
  adjustments, log5 / Poisson / surface-conditioned baselines. Don't eyeball math you
  could compute. Surface every numeric derivation in `computed_numbers` so the
  reasoner can use it as-is.

You are expected to actually call these tools — not recite what you already know.

Output rules — return ONLY valid JSON matching the LensNotebook schema:
- `lens` must equal the lens you've been assigned.
- `team_a_name` / `team_b_name` are copied verbatim from the user message.
- `research_notes` is free-form prose (multi-paragraph, sectioned as you like).
  Bullet what you found and — important — what's MISSING. No probability, no
  signed shift, no "team_a wins because…" sentence.
- `citations`: every URL you actually retrieved via search. Never fabricate URLs.
  `claim` is a one-line summary; `retrieved_value` is the concrete fact (a stat,
  a status, a line) lifted from the page.
- `computed_numbers`: every number you derived via code_execution. `method` is a
  one-line note on the math.
- `coverage`: 'thin' when primary sources were unavailable, 'rich' when you found
  multiple high-quality sources, 'adequate' otherwise. The reasoner downgrades
  confidence to 'low' on a thin notebook.

Live games: when the event context's `Game state` line shows `LIVE` (with period,
elapsed time, and score), prioritise capturing the in-play state and recent in-game
developments — pre-game baselines decay quickly once the ball is in the air. Note in
`research_notes` that you adjusted research focus for live state.
""".strip()


# ----- DEFAULT lens set (legacy three-lens trio) tool prose -----

_TOOLS_DEFAULT_STATISTICS = """
What each tool can give you here:
- web_search: stats pages (basketball-reference, fangraphs, fbref, pro-football-reference,
  or sport equivalents), recent game logs, home/away splits, rating systems.
- x_search: recent roster or line changes that might invalidate a statistical baseline.
- code_execution: derive candidate team_a-win baselines via log5, rating-differential, or
  recent-N-games weighting and surface them in `computed_numbers` (label them clearly so
  the reasoner can pick the most defensible one). Compute league base rates (e.g.
  home-team win%) for the reasoner to anchor against. Don't pick a single final number —
  the reasoner will weigh candidates.
""".strip()

_TOOLS_DEFAULT_INJURY = """
What each tool can give you here:
- x_search: beat reporters (e.g. Shams, Woj, Schefter, Rapoport, Passan, or sport
  equivalents) and official team accounts — injury and lineup news usually breaks here
  faster than anywhere else.
- web_search: official team injury reports, ESPN injury index, The Athletic. For combat
  sports / tennis, weigh-ins, withdrawals, training-camp reporting.
- code_execution: when a star is out, compute the on/off win-rate split, win-share delta,
  BPM-with/without, or sport-equivalent impact number and surface it in `computed_numbers`
  (e.g. label='lakers_with_lebron_winrate', value=0.62, method='regular-season W/L when
  active vs out, n=…'). The reasoner will combine these into the signed shift.
""".strip()

_TOOLS_DEFAULT_NARRATIVE = """
What each tool can give you here:
- x_search: public sentiment, reporter takes, team and player accounts, fan-base mood,
  locker-room chatter. Pull recent posts from beat reporters and team handles, not just
  generic search.
- web_search: beat-reporter features, team press conferences, coaching interviews,
  managerial-change reporting, derby/cup-final coverage.
- code_execution: ground a narrative claim in a number when you can (e.g. post-firing
  coaching-bump win% in the league, trade-deadline record splits) and put it in
  `computed_numbers`.
""".strip()


# ----- TENNIS lens set tool prose -----

_TOOLS_TENNIS_FORM_AND_SURFACE = """
What each tool can give you here:
- web_search: tennisabstract.com (Elo + surface splits + recent-match logs), ATP/WTA
  official, Infosys ATP stats, flashscore for recent results, tennis.com features. The
  structured tennis-stats block in your user message has YTD W-L, surface_win_loss,
  career serve/return %, tier records, last_10_form, and recent_matches — copy those
  verbatim into research_notes / computed_numbers without re-searching for them.
- x_search: training-block reports, recent-loss color (was that 6-3 6-2 a clean win or
  did the favorite drop a tight set?), surface-trajectory commentary, equipment changes
  (string-tension shifts that affect form on faster surfaces).
- code_execution: surface-conditioned Elo (from tennisabstract or recent matches),
  recent-form-weighted baseline (last 10 / last 20 weighted toward this surface),
  candidate baselines surfaced as `team_a_baseline_surface_elo`,
  `team_a_baseline_recent_8_weighted`, etc. Don't pre-bake H2H, conditions, or stakes
  effects into the baseline — those are owned by the other tennis lenses.
""".strip()

_TOOLS_TENNIS_MATCHUP_AND_CLUTCH = """
What each tool can give you here:
- web_search: tennisabstract.com H2H pages (lifetime + per-surface), ATP/WTA official
  H2H pages, tennis press recent-meeting recaps. Game-style fit commentary
  (lefty-vs-OHB, big-server-vs-returner) on tennis.com / Tennis Channel features.
- x_search: tactical commentary on past meetings, beat-reporter takes on style
  matchups, choke-history threads, deep-Slam-run history.
- code_execution: H2H-conditioned win-rate (with binomial CIs when N is small),
  decider-record fit comparison (in-matchup deciders vs career deciders),
  comeback-rate and closeout-rate comparison, BP-save vs BP-convert delta. Label
  numbers like `decider_record_alcaraz_vs_djokovic`, `comeback_pct_alcaraz_in_matchup`.
- DO NOT push for SURFACE effect here — the form lens owns surface. Surface-conditioned
  H2H informs your reasoning qualitatively only.
""".strip()

_TOOLS_TENNIS_CONDITIONS_AND_CONTEXT = """
What each tool can give you here:
- web_search: weather forecasts (weather.com, accuweather, Météo-France for European
  events) for the match window; venue-specific surface speed (CPI / surface-pace index
  references in tennis-abstract write-ups); ball brand and altitude refs in tournament
  press; tournament draw / schedule for fatigue load; ATP/WTA official withdrawal
  notices. For stakes / motivation, Tennis Channel + tennis.com features.
- x_search: same-day press conferences, training-camp reports, late warm-up issues,
  beat reporters (José Morgado, Christopher Clarey, Ben Rothenberg) and player social
  media (Twitter / Instagram stories often leak warm-up issues), coaching-change
  reporting, narrative threads.
- code_execution: tour-baseline retirement rate (~3–5% of matches) for flagging
  withdrawal-risk-elevated players in the trailing 90d. Cumulative tournament-load
  estimates (sets × minutes per set, rough fatigue-index math). Label numbers like
  `team_a_fatigue_index_minutes_played`, `weather_serve_drag_team_a`.
""".strip()


# Flat dict — keys are full lens names. Unique across the registry, so a
# new sport just adds N keys.
_TOOLS_BY_LENS: dict[str, str] = {
    # Default lens set
    "statistics": _TOOLS_DEFAULT_STATISTICS,
    "injury": _TOOLS_DEFAULT_INJURY,
    "narrative": _TOOLS_DEFAULT_NARRATIVE,
    # Tennis lens set
    "tennis_form_and_surface": _TOOLS_TENNIS_FORM_AND_SURFACE,
    "tennis_matchup_and_clutch": _TOOLS_TENNIS_MATCHUP_AND_CLUTCH,
    "tennis_conditions_and_context": _TOOLS_TENNIS_CONDITIONS_AND_CONTEXT,
}


class GrokProvider:
    """xAI/Grok implementation of `FetcherProvider`.

    Holds the `XAIAsyncClient` for the run and pre-builds per-(sport, lens)
    system prompts at construction by iterating `SPORT_LENS_SETS`. Each
    `fetch` call resolves the cached prompt by `(sport, lens_name)` and
    only does the per-event message rendering. Tools (`web_search`,
    `x_search`, `code_execution`) are passed fresh per call — the SDK
    expects a list, not a singleton.
    """

    name = "grok"
    model = GROK_MODEL

    def __init__(self, api_key: str) -> None:
        self._xai = XAIAsyncClient(api_key=api_key)
        # Pre-build per-(sport, lens) system prompts at construction so
        # each fetch call only does the per-event message rendering.
        # Keyed by (sport, lens_name) because lens names ARE unique
        # across the registry but lookup by tuple is cheap and explicit
        # at the call site (`provider._lens_prompts[(sport, lens)]`).
        self._lens_prompts: dict[tuple[str, str], str] = {}
        for sport, lens_set in SPORT_LENS_SETS.items():
            for prompt_name, prompt in build_lens_prompts_for_set(
                lens_set, _TOOLS_BY_LENS, NOTEBOOK_TAIL_GROK
            ).items():
                self._lens_prompts[(sport, prompt_name)] = prompt

    async def fetch(
        self,
        event: PolymarketEvent,
        lens: str,
        *,
        lens_set: LensSet,
    ) -> LensNotebook:
        chat = self._xai.chat.create(
            model=GROK_MODEL,
            agent_count=GROK_AGENT_COUNT,
            messages=[system(self._lens_prompts[(lens_set.sport, lens)])],
            tools=[web_search(), x_search(), code_execution()],
        )
        spec = lens_set.lens_specs_by_name[lens]
        user_msg = render_user_message_for_lens(event, spec)
        chat.append(user(user_msg))
        response, parsed = await chat.parse(LensNotebook)
        assert_lens_match(parsed, lens, event.id)
        log.debug(
            "fetcher=grok sport=%s lens=%s event=%s coverage=%s citations=%d "
            "computed=%d tokens in/out=%s/%s",
            lens_set.sport,
            lens,
            event.id,
            parsed.coverage,
            len(parsed.citations),
            len(parsed.computed_numbers),
            getattr(response.usage, "prompt_tokens", None),
            getattr(response.usage, "completion_tokens", None),
        )
        return parsed

    async def aclose(self) -> None:
        await self._xai.close()
