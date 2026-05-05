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
lenses (`tennis_conditions_and_context` for warm-up issues / late
withdrawals) — that's part of what the A/B measures, not a bug to paper
over.

Per-sport-lens-set refactor: prompts are pre-built per `(sport, lens)` at
construction by iterating `SPORT_LENS_SETS`. `_TOOLS_BY_LENS` is the
provider-owned, FLAT dict of per-lens tool prose — keyed by full lens
name (not by `(sport, lens)`) because lens names are unique across the
registry. Adding a new sport adds N entries to `_TOOLS_BY_LENS` (one
per new lens) without touching existing entries.

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
    build_lens_prompts_for_set,
    render_user_message_for_lens,
)
from skimsmarkets.agents.schemas import LensNotebook
from skimsmarkets.agents.sports import SPORT_LENS_SETS
from skimsmarkets.agents.sports.base import LensSet
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


# ----- TENNIS lens set tool prose -----

_TOOLS_TENNIS_FORM_AND_SURFACE = """
What each tool can give you here:
- google_search: tennisabstract.com (Elo + surface splits + recent-match logs),
  ATP/WTA official, Infosys ATP stats, flashscore for recent results, tennis.com
  features. The structured tennis-stats block in your user message has YTD W-L,
  surface_win_loss, career serve/return %, tier records, last_10_form, and
  recent_matches — copy those verbatim into research_notes / computed_numbers
  without re-searching for them. For training-block or recent-loss color, query
  `site:x.com` plus the player handle or the tour beat reporter.
- code_execution: surface-conditioned Elo (from tennisabstract or recent matches),
  recent-form-weighted baseline (last 10 / last 20 weighted toward this surface),
  candidate baselines surfaced as `team_a_baseline_surface_elo`,
  `team_a_baseline_recent_8_weighted`, etc. Don't pre-bake H2H, conditions, or
  stakes effects into the baseline — those are owned by the other tennis lenses.
""".strip()

_TOOLS_TENNIS_MATCHUP_AND_CLUTCH = """
What each tool can give you here:
- google_search: tennisabstract.com H2H pages (lifetime + per-surface), ATP/WTA
  official H2H pages, tennis press recent-meeting recaps. Game-style fit
  commentary on tennis.com / Tennis Channel features. For tactical commentary
  on past meetings or choke-history threads, query `site:x.com` plus the
  player handle or beat reporter.
- code_execution: H2H-conditioned win-rate (with binomial CIs when N is small),
  decider-record fit comparison (in-matchup deciders vs career deciders),
  comeback-rate and closeout-rate comparison, BP-save vs BP-convert delta. Label
  numbers like `decider_record_alcaraz_vs_djokovic`, `comeback_pct_alcaraz_in_matchup`.
- DO NOT push for SURFACE effect here — the form lens owns surface. Surface-conditioned
  H2H informs your reasoning qualitatively only.
""".strip()

_TOOLS_TENNIS_CONDITIONS_AND_CONTEXT = """
What each tool can give you here:
- google_search: weather forecasts (weather.com, accuweather, Météo-France for
  European events) for the match window; venue-specific surface speed (CPI /
  surface-pace index references in tennis-abstract write-ups); ball brand and
  altitude refs in tournament press; tournament draw / schedule for fatigue
  load; ATP/WTA official withdrawal notices. For stakes / motivation, Tennis
  Channel + tennis.com features. For same-day press conferences, training-camp
  reports, late warm-up issues, beat reporters (José Morgado, Christopher
  Clarey, Ben Rothenberg) and player social media, query `site:x.com` plus the
  reporter or player handle. Twitter indexing on Google is incomplete and lags —
  when same-day social coverage is sparse, set `coverage='thin'` rather than
  overstating what you found.
- code_execution: tour-baseline retirement rate (~3–5% of matches) for flagging
  withdrawal-risk-elevated players in the trailing 90d. Cumulative tournament-
  load estimates (sets × minutes per set, rough fatigue-index math). Label
  numbers like `team_a_fatigue_index_minutes_played`, `weather_serve_drag_team_a`.
""".strip()


_TOOLS_BY_LENS: dict[str, str] = {
    # Tennis lens set
    "tennis_form_and_surface": _TOOLS_TENNIS_FORM_AND_SURFACE,
    "tennis_matchup_and_clutch": _TOOLS_TENNIS_MATCHUP_AND_CLUTCH,
    "tennis_conditions_and_context": _TOOLS_TENNIS_CONDITIONS_AND_CONTEXT,
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

    Holds the `genai.Client` for the run and pre-builds per-(sport, lens)
    system prompts at construction by iterating `SPORT_LENS_SETS`. Each
    `fetch` call sends the system prompt via `system_instruction`, the
    rendered event context as the user content, and the two tools
    (`google_search`, `code_execution`) on the per-call config.

    Single-pass by design — Gemini has no `agent_count` analogue. See the
    module docstring for the A/B framing.
    """

    name = "gemini"
    model = GEMINI_MODEL

    def __init__(self, api_key: str) -> None:
        self._client = genai.Client(api_key=api_key)
        self._lens_prompts: dict[tuple[str, str], str] = {}
        for sport, lens_set in SPORT_LENS_SETS.items():
            for prompt_name, prompt in build_lens_prompts_for_set(
                lens_set, _TOOLS_BY_LENS, NOTEBOOK_TAIL_GEMINI
            ).items():
                self._lens_prompts[(sport, prompt_name)] = prompt

    async def fetch(
        self,
        event: PolymarketEvent,
        lens: str,
        *,
        lens_set: LensSet,
    ) -> LensNotebook:
        spec = lens_set.lens_specs_by_name[lens]
        user_msg = render_user_message_for_lens(event, spec)

        # Tools are passed per-call (the SDK's Tool wrappers are lightweight
        # config objects, not stateful primitives). google_search +
        # code_execution mirror the xAI pair; x_search has no Gemini
        # equivalent so the system prompt routes Twitter lookups through
        # `site:x.com` queries on google_search.
        config = genai_types.GenerateContentConfig(
            system_instruction=self._lens_prompts[(lens_set.sport, lens)],
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
            "fetcher=gemini sport=%s lens=%s event=%s coverage=%s citations=%d "
            "computed=%d tokens in/out=%s/%s",
            lens_set.sport,
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
