"""Grok fetcher provider — xAI SDK with `agent_count=4` multi-agent ensemble.

Each lens runs as one xAI chat with `web_search` + `x_search` +
`code_execution` tools available; `agent_count=4` fans the search out
across 4 independent trajectories that the SDK merges into one structured
response. The 4× search-path diversity is what we get from xAI's native
ensemble — Gemini has no equivalent primitive, so the parallel
`GeminiProvider` runs single-pass and the A/B is "Grok-with-its-native-
ensemble vs Gemini-single-pass" by design.
"""

from __future__ import annotations

import logging
from typing import Literal

from xai_sdk import AsyncClient as XAIAsyncClient
from xai_sdk.chat import system, user
from xai_sdk.tools import code_execution, web_search, x_search

from skimsmarkets.agents.fetchers.base import (
    assert_lens_match,
    build_lens_prompts,
    render_context,
    render_lens_extras,
)
from skimsmarkets.agents.schemas import LensName, LensNotebook
from skimsmarkets.agents.sport_hints import render_sport_hint
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
  sportsbook odds, weather, venue.
- x_search: breaking news, beat-reporter leaks, team/player accounts, public sentiment —
  usually the fastest channel for anything <24h old.
- code_execution: run Python when numbers need computing — de-vigging sportsbook odds,
  converting ratings to probabilities, weighting recent-form vs season baselines.
  Don't eyeball math you could compute. Surface every numeric derivation in
  `computed_numbers` so the reasoner can use it as-is.

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


# Per-lens "What each tool can give you here" blocks. Naming the xAI tools
# explicitly so Grok's adaptive search loop knows exactly which primitive
# to reach for at each decision point.
TOOLS_SECTION_STATISTICS = """
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


TOOLS_SECTION_INJURY = """
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


TOOLS_SECTION_NARRATIVE = """
What each tool can give you here:
- x_search: public sentiment, reporter takes, team and player accounts, fan-base mood,
  locker-room chatter. Pull recent posts from beat reporters and team handles, not just
  generic search.
- web_search: beat-reporter features, team press conferences, coaching interviews, and
  (for outdoor sports) weather and venue pages.
- code_execution: ground a narrative claim in a number when you can (e.g. post-firing
  coaching-bump win% in the league, trade-deadline record splits) and put it in
  `computed_numbers`.
""".strip()


TOOLS_SECTION_MARKET_CONTEXT = """
What each tool can give you here:
- web_search: current moneyline / outright odds from DraftKings, FanDuel, BetMGM, and
  especially Pinnacle (the sharpest book). Check open-vs-current for line movement.
- x_search: sharp-money commentary, betting-Twitter line-movement reporting, steam-move
  alerts.
- code_execution: de-vig the two-sided sportsbook odds into fair probabilities before
  comparing — raw American moneylines include vig and will systematically mislead a
  direct comparison.
""".strip()


_TOOLS_BY_LENS: dict[LensName, str] = {
    "statistics": TOOLS_SECTION_STATISTICS,
    "injury": TOOLS_SECTION_INJURY,
    "narrative": TOOLS_SECTION_NARRATIVE,
    "market_context": TOOLS_SECTION_MARKET_CONTEXT,
}


class GrokProvider:
    """xAI/Grok implementation of `FetcherProvider`.

    Holds the `XAIAsyncClient` for the run and pre-builds the four
    lens-specific system prompts at construction so each `fetch` call
    only does the per-event message rendering. Tools (`web_search`,
    `x_search`, `code_execution`) are passed fresh per call — the SDK
    expects a list, not a singleton.
    """

    name = "grok"
    model = GROK_MODEL

    def __init__(self, api_key: str) -> None:
        self._xai = XAIAsyncClient(api_key=api_key)
        self._lens_prompts = build_lens_prompts(_TOOLS_BY_LENS, NOTEBOOK_TAIL_GROK)

    async def fetch(
        self, event: PolymarketEvent, lens: LensName
    ) -> LensNotebook:
        chat = self._xai.chat.create(
            model=GROK_MODEL,
            agent_count=GROK_AGENT_COUNT,
            messages=[system(self._lens_prompts[lens])],
            tools=[web_search(), x_search(), code_execution()],
        )
        # Sport-specific guidance rides on the user message (NOT the cached
        # system block) so the cached system prompt stays warm across all
        # events. Returns None for sports we don't specialize, in which
        # case we send the bare context.
        user_msg = render_context(event)
        if (sport_hint := render_sport_hint(lens, event)) is not None:
            user_msg += "\n\n" + sport_hint
        # Lens-specific extras (currently: tennis player stats for the
        # statistics lens). Same posture as `render_sport_hint` — appended
        # to the per-event user message, never to the cached system block.
        if (extras := render_lens_extras(lens, event)) is not None:
            user_msg += "\n\n" + extras
        chat.append(user(user_msg))
        response, parsed = await chat.parse(LensNotebook)
        assert_lens_match(parsed, lens, event.id)
        log.debug(
            "fetcher=grok lens=%s event=%s coverage=%s citations=%d computed=%d "
            "tokens in/out=%s/%s",
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
