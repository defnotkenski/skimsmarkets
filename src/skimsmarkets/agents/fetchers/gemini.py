"""Gemini fetcher provider — Google Gen AI SDK with `gemini-3.1-pro-preview`
single-pass.

Each lens runs as one Gemini `generate_content` call with `google_search`
grounding + `code_execution` available. Single-pass per lens (no `agent_count`
equivalent in Gemini's API) — the A/B against `GrokProvider` is intentionally
"Gemini single-pass vs Grok native ensemble" so each model competes in its
native mode rather than one being shoehorned into the other's loop.

Twitter/X gap: Gemini has no native X search (xAI's `x_search` is a
provider-specific primitive). The per-lens tools sections below tell Gemini
to fall back to `google_search` with `site:x.com` / `site:twitter.com` for
beat-reporter posts. Coverage will likely be thinner on the social-data
lenses (narrative, injury) — that's part of what the A/B measures, not a
bug to paper over.

Structured output: per Gemini docs, the Gemini 3 series (`gemini-3.1-pro-
preview`, `gemini-3-flash-preview`) supports `response_json_schema`
combined with `google_search` + `code_execution` tools simultaneously —
the historical schema-vs-grounding conflict was a Gemini 2.x limitation.
We pass `LensNotebook.model_json_schema()` so the schema is enforced
server-side, matching the Grok provider's `chat.parse(LensNotebook)`
posture (CLAUDE.md "structured output everywhere"). The code-fence
stripper on the parse path stays as defence-in-depth.
"""

from __future__ import annotations

import logging
import re

from google import genai
from google.genai import types as genai_types

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

GEMINI_MODEL = "gemini-3.1-pro-preview"
# Output ceiling for the structured JSON response (does NOT include thinking
# tokens — those are billed separately on Gemini 3.x). Matches the
# Claude reasoner / director ceiling (16k) so a wordy LensNotebook with
# rich `research_notes` + a dozen citations + computed_numbers fits with
# headroom. Default Gemini caps are model-dependent and have truncated
# real notebooks mid-JSON in practice — set this explicitly.
GEMINI_MAX_OUTPUT_TOKENS = 16_000

# Gemini-3.x occasionally returns finish_reason=STOP with truncated JSON,
# or empty text when grounding tools fire in an unhappy path. Both are
# transient sampling bugs — a re-call almost always clears them. One retry
# (2 attempts total) catches the common case without burning unbounded
# tokens on a genuinely broken event. Genuine API errors (auth, 429
# RESOURCE_EXHAUSTED) raise `google.genai.errors.ClientError` which is NOT
# a RuntimeError and bubbles past the retry loop unchanged.
_PARSE_RETRY_ATTEMPTS = 2


# Generic notebook tail — output rules + tool list naming Gemini's actual
# tools. The Twitter/X workaround is mentioned where x_search would have
# appeared in the Grok variant.
NOTEBOOK_TAIL_GEMINI = """
You are a FETCHER, not a reasoner. Your job is evidence capture — not judgment.
Do NOT output a probability, a signed shift, a directional verdict, or a single
"team_a will probably win" sentence. The downstream reasoner does that.

Tools available — use whichever fit what you're trying to learn, and chain several calls if
the first doesn't answer the question:
- google_search: URL-citable facts — stats pages, official injury reports, press coverage,
  weather, venue. For Twitter/X content (beat-reporter posts, breaking news, public
  sentiment), use google_search with `site:x.com` or `site:twitter.com` plus reporter
  handles. Recent posts may be incompletely indexed; flag thin social-media coverage
  in `coverage` when injury/narrative reporting depends on it.
- code_execution: run Python when numbers need computing — converting ratings to
  probabilities, weighting recent-form vs season baselines, computing weather-impact
  adjustments, log5 / Poisson / surface-conditioned baselines. Don't eyeball math you
  could compute. Surface every numeric derivation in `computed_numbers` so the
  reasoner can use it as-is.

You are expected to actually call these tools — not recite what you already know.

Paraphrase, do NOT quote verbatim. When summarizing what you found in
`research_notes` or `citations.claim`, restate findings in your own words rather
than copying sentences from search results, tweets, or news articles. Numbers,
stat lines, player names, scores, and dates should be transcribed accurately —
those are facts, not prose — but the surrounding sentences must be original.
Near-verbatim copying of grounded source text triggers content-similarity
filters and will cause the model to return an empty response, dropping the
event from the slate. This applies especially to social-media content
(beat-reporter posts via `site:x.com`) where source phrasing is short and
distinctive: extract the fact, then write your own sentence about it.

Output rules — return ONLY valid JSON matching the LensNotebook schema, with no
prose before or after the JSON object and no markdown code fences:
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


# Per-lens "What each tool can give you here" blocks. The xAI `x_search`
# bullet is replaced with a `site:x.com` workaround; keep the social-data
# lenses (narrative, injury) honest about the coverage trade-off.
TOOLS_SECTION_STATISTICS = """
What each tool can give you here:
- google_search: stats pages (basketball-reference, fangraphs, fbref, pro-football-reference,
  or sport equivalents), recent game logs, home/away splits, rating systems. For roster /
  line changes that might invalidate a baseline, query `site:x.com` plus the team or
  reporter handle.
- code_execution: derive candidate team_a-win baselines via log5, rating-differential, or
  recent-N-games weighting and surface them in `computed_numbers` (label them clearly so
  the reasoner can pick the most defensible one). Compute league base rates (e.g.
  home-team win%) for the reasoner to anchor against. Don't pick a single final number —
  the reasoner will weigh candidates.
""".strip()


TOOLS_SECTION_INJURY = """
What each tool can give you here:
- google_search: official team injury reports, ESPN injury index, The Athletic. For beat
  reporters (Shams, Woj, Schefter, Rapoport, Passan, or sport equivalents) and team
  accounts where injury news usually breaks first, query `site:x.com` plus the reporter
  handle (e.g. `site:x.com @ShamsCharania Lakers injury`). Recent posts may be incompletely
  indexed — when injury reporting is sparse, set `coverage='thin'` and surface the gap in
  `research_notes`. For combat sports / tennis, weigh-ins, withdrawals, training-camp
  reporting.
- code_execution: when a star is out, compute the on/off win-rate split, win-share delta,
  BPM-with/without, or sport-equivalent impact number and surface it in `computed_numbers`
  (e.g. label='lakers_with_lebron_winrate', value=0.62, method='regular-season W/L when
  active vs out, n=…'). The reasoner will combine these into the signed shift.
""".strip()


TOOLS_SECTION_NARRATIVE = """
What each tool can give you here:
- google_search: beat-reporter features, team press conferences, coaching interviews,
  managerial-change reporting, derby/cup-final coverage. For public sentiment and
  locker-room chatter, query `site:x.com` plus team / player handles or relevant
  beat-reporter accounts (e.g. `site:x.com @MarcSpears Celtics locker room`). Twitter
  indexing on Google is incomplete and lags — when sentiment data is sparse, set
  `coverage='thin'` rather than overstating what you found.
- code_execution: ground a narrative claim in a number when you can (e.g. post-firing
  coaching-bump win% in the league, trade-deadline record splits) and put it in
  `computed_numbers`.
""".strip()


_TOOLS_BY_LENS: dict[LensName, str] = {
    "statistics": TOOLS_SECTION_STATISTICS,
    "injury": TOOLS_SECTION_INJURY,
    "narrative": TOOLS_SECTION_NARRATIVE,
}


# Gemini occasionally wraps JSON in ```json … ``` even with
# response_mime_type=application/json when grounding tools fired. This
# unwraps both fenced and unfenced responses without affecting clean ones.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    match = _FENCE_RE.match(text)
    return match.group(1) if match else text.strip()


class GeminiProvider:
    """Google Gen AI / Gemini implementation of `FetcherProvider`.

    Holds the `genai.Client` for the run and pre-builds the three
    lens-specific system prompts at construction. Each `fetch` call sends
    the system prompt via `system_instruction`, the rendered event context
    as the user content, and the two tools (`google_search`,
    `code_execution`) on the per-call config.

    Single-pass by design — Gemini has no `agent_count` analogue. See the
    module docstring for the A/B framing.
    """

    name = "gemini"
    model = GEMINI_MODEL

    def __init__(self, api_key: str) -> None:
        self._client = genai.Client(api_key=api_key)
        self._lens_prompts = build_lens_prompts(_TOOLS_BY_LENS, NOTEBOOK_TAIL_GEMINI)

    async def fetch(self, event: PolymarketEvent, lens: LensName) -> LensNotebook:
        user_msg = render_context(event)
        if (sport_hint := render_sport_hint(lens, event)) is not None:
            user_msg += "\n\n" + sport_hint
        # Lens-specific extras — currently the tennis player-stats block
        # for the statistics lens, no-op for everything else. Mirrors the
        # Grok provider; both stay in lockstep on what each lens sees.
        if (extras := render_lens_extras(lens, event)) is not None:
            user_msg += "\n\n" + extras

        # Tools are passed per-call (the SDK's Tool wrappers are lightweight
        # config objects, not stateful primitives). google_search +
        # code_execution mirror the xAI pair; x_search has no Gemini
        # equivalent so the system prompt routes Twitter lookups through
        # `site:x.com` queries on google_search.
        config = genai_types.GenerateContentConfig(
            system_instruction=self._lens_prompts[lens],
            tools=[
                genai_types.Tool(google_search=genai_types.GoogleSearch()),
                genai_types.Tool(code_execution=genai_types.ToolCodeExecution()),
            ],
            # Server-side schema-constrained output. Gemini 3.x supports
            # `response_json_schema` alongside grounding + code-execution
            # tools; pass the JSON schema rather than the Pydantic class
            # because `response_json_schema` is the documented field for
            # the `from_json_schema()` path and is what the docs recommend
            # for Pydantic models in 2026.
            response_mime_type="application/json",
            response_json_schema=LensNotebook.model_json_schema(),
            # `thinking_level="high"` is the default for gemini-3.1-pro-preview
            # but we set it explicitly to mirror the Claude reasoner's
            # `effort="max"` posture — be explicit about cost/quality tradeoffs
            # rather than relying on defaults that may change between model
            # versions. The fetcher does adaptive search + de-vig math, both
            # of which benefit from deeper reasoning.
            thinking_config=genai_types.ThinkingConfig(
                thinking_level=genai_types.ThinkingLevel.HIGH
            ),
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
        )

        # The API call + parse is retried once on transient parse-class
        # failures (empty text, truncated JSON despite finish_reason=STOP).
        # Success breaks out; the second attempt's failure re-raises with
        # the same RuntimeError shape the pipeline expects for
        # ErrorRecord(stage="fetcher:<lens>"). `response` and `parsed` are
        # set on the success path before break.
        response = None
        parsed: LensNotebook | None = None
        for attempt in range(_PARSE_RETRY_ATTEMPTS):
            try:
                response = await self._client.aio.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=user_msg,
                    config=config,
                )

                raw = _strip_code_fence(response.text or "")
                # `finish_reason=MAX_TOKENS` is the canonical signal that the
                # response was truncated mid-output — surfacing it in the error
                # lets a future operator distinguish "schema/format problem" from
                # "bump the budget" without re-running.
                finish_reason = None
                candidates = getattr(response, "candidates", None) or []
                if candidates:
                    finish_reason = getattr(candidates[0], "finish_reason", None)
                if not raw:
                    raise RuntimeError(
                        f"Gemini returned empty response for lens={lens} "
                        f"event={event.id} finish_reason={finish_reason}"
                    )
                try:
                    parsed = LensNotebook.model_validate_json(raw)
                except Exception as e:
                    # Surface a slice of the raw response so debugging is fast —
                    # Gemini sometimes prepends a tool-use trace before the JSON
                    # which `_strip_code_fence` doesn't catch.
                    preview = raw[:400].replace("\n", " ")
                    raise RuntimeError(
                        f"Gemini response failed LensNotebook parse for lens={lens} "
                        f"event={event.id} finish_reason={finish_reason}: {e}. "
                        f"preview={preview!r}"
                    ) from e
                break
            except RuntimeError as e:
                if attempt + 1 < _PARSE_RETRY_ATTEMPTS:
                    log.warning(
                        "gemini parse retry lens=%s event=%s attempt=%d/%d: %s",
                        lens, event.id, attempt + 1, _PARSE_RETRY_ATTEMPTS, e,
                    )
                    continue
                raise

        assert parsed is not None and response is not None  # break-path invariant
        assert_lens_match(parsed, lens, event.id)

        # `usage_metadata` is the Gen AI SDK's token-count record. Field
        # names differ from xAI; pull defensively so a future SDK rename
        # doesn't break logging.
        usage = getattr(response, "usage_metadata", None)
        log.debug(
            "fetcher=gemini lens=%s event=%s coverage=%s citations=%d computed=%d "
            "tokens in/out=%s/%s",
            lens,
            event.id,
            parsed.coverage,
            len(parsed.citations),
            len(parsed.computed_numbers),
            getattr(usage, "prompt_token_count", None),
            getattr(usage, "candidates_token_count", None),
        )
        return parsed

    # noinspection PyMethodMayBeStatic
    async def aclose(self) -> None:
        # `genai.Client` has no public close — the underlying transport is
        # managed by the SDK and released on garbage collection. Defined
        # for Protocol parity with `GrokProvider.aclose` (which DOES need
        # `self` to close its xAI client).
        return None
