"""Per-lens Claude reasoners — Stage B of the two-stage agent chain.

Each reasoner takes the `LensNotebook` produced by its lens's fetcher
(Grok or Gemini, transparent to the reasoner) plus the same event context
the fetcher saw, and emits the typed report
(`StatisticsReport`, `InjuryReport`, `NarrativeReport`) that the director
consumes. Verdicts (probability, signed shift, motivation_edge) live here,
not in the notebook.

System prompts are cached per-lens via `CacheControlEphemeralParam` —
Anthropic caps active ephemeral breakpoints at 4 per request and we use 1
per request (one system block per call). If anything ever wants to add a
shared common prefix across reasoners, it must live in its OWN cached block
ABOVE the lens-specific block — never concatenated into the lens prompt, or
every lens forces a fresh cache write and the whole point of caching is
lost.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from anthropic import AsyncAnthropic
from anthropic.types import (
    CacheControlEphemeralParam,
    MessageParam,
    OutputConfigParam,
    TextBlockParam,
    ThinkingConfigAdaptiveParam,
)
from pydantic import BaseModel, ValidationError

from skimsmarkets.agents.director import (
    CLAUDE_MAX_OUTPUT_TOKENS,
    CLAUDE_MODEL,
    _PARSE_RETRY_ATTEMPTS,
)
from skimsmarkets.agents.fetchers import render_context, render_lens_extras
from skimsmarkets.agents.prompts import (
    INJURY_REASONER_SYSTEM,
    NARRATIVE_REASONER_SYSTEM,
    STATISTICS_REASONER_SYSTEM,
)
from skimsmarkets.agents.sport_hints import render_reasoner_sport_hint
from skimsmarkets.agents.schemas import (
    InjuryReport,
    LensName,
    LensNotebook,
    NarrativeReport,
    SpecialistReport,
    StatisticsReport,
)
from skimsmarkets.polymarket.models import PolymarketEvent

log = logging.getLogger(__name__)

ReasonerFn = Callable[
    [AsyncAnthropic, PolymarketEvent, LensNotebook],
    Awaitable[SpecialistReport],
]


async def _run_reasoner(
    anthropic: AsyncAnthropic,
    event: PolymarketEvent,
    notebook: LensNotebook,
    system_prompt: str,
    output_format: type[BaseModel],
    lens: LensName,
) -> BaseModel:
    """Shared body for the three lens reasoners.

    Mirrors `synthesize_prediction` in director.py: cached system block,
    adaptive thinking, max effort. The user message is the canonical event
    context (same string the fetcher saw) plus the notebook serialized as
    indented JSON. Reasoner prompts are explicit that
    `notebook.computed_numbers` are pre-derived and should be used as-is.
    """
    # Lens-specific extras (currently: tennis player stats for the
    # statistics lens). Threaded into BOTH the fetcher and the reasoner so
    # the reasoner can lift the vendor's numbers into its output the same
    # way it does notebook.computed_numbers — without us having to also
    # serialize them through the notebook itself.
    extras = render_lens_extras(lens, event)
    extras_block = f"\n\n{extras}" if extras else ""
    # Sport-specific calibration hint (currently: injury reasoner for tennis
    # and soccer). Same posture as the fetcher's sport_hint and the lens
    # extras above — appended to the per-event user message, never the
    # cached system block, so the slate-wide cache hit on the system prompt
    # is preserved. Returns None for sport/lens combinations we don't
    # specialize, leaving the reasoner on its generic prompt.
    sport_hint = render_reasoner_sport_hint(lens, event)
    sport_hint_block = f"\n\n{sport_hint}" if sport_hint else ""
    user_msg = (
        render_context(event)
        + extras_block
        + sport_hint_block
        + "\n\n--- LensNotebook ---\n"
        + notebook.model_dump_json(indent=2)
        + "\n\nReturn the typed report per the schema. "
        "Trust the event context for team_a_name / team_b_name; use "
        "notebook.computed_numbers as-is."
    )

    system_block = TextBlockParam(
        type="text",
        text=system_prompt,
        cache_control=CacheControlEphemeralParam(type="ephemeral"),
    )
    user_message = MessageParam(role="user", content=user_msg)

    # Retry once on parse-class failures (malformed JSON, no parsed output) —
    # same posture as the director. Genuine API errors raise their own
    # exception classes and bubble past unchanged.
    parsed = None
    report = None
    for attempt in range(_PARSE_RETRY_ATTEMPTS):
        try:
            parsed = await anthropic.messages.parse(
                model=CLAUDE_MODEL,
                max_tokens=CLAUDE_MAX_OUTPUT_TOKENS,
                system=[system_block],
                messages=[user_message],
                output_format=output_format,
                thinking=ThinkingConfigAdaptiveParam(type="adaptive"),
                output_config=OutputConfigParam(effort="max"),
            )
            report = parsed.parsed_output
            if report is None:
                raise RuntimeError(
                    f"Reasoner returned no parsed output for lens={lens} event={event.id}; "
                    f"stop_reason={parsed.stop_reason}"
                )
            break
        except (ValidationError, RuntimeError) as e:
            if attempt + 1 < _PARSE_RETRY_ATTEMPTS:
                log.warning(
                    "reasoner parse retry lens=%s event=%s attempt=%d/%d: %s",
                    lens, event.id, attempt + 1, _PARSE_RETRY_ATTEMPTS, e,
                )
                continue
            raise

    assert parsed is not None and report is not None  # break-path invariant
    log.debug(
        "reasoner=%s event=%s tokens in/out=%s/%s",
        lens,
        event.id,
        parsed.usage.input_tokens,
        parsed.usage.output_tokens,
    )
    return report


async def reason_statistics(
    anthropic: AsyncAnthropic, event: PolymarketEvent, notebook: LensNotebook
) -> StatisticsReport:
    out = await _run_reasoner(
        anthropic, event, notebook, STATISTICS_REASONER_SYSTEM, StatisticsReport,
        "statistics",
    )
    assert isinstance(out, StatisticsReport)
    return out


async def reason_injury(
    anthropic: AsyncAnthropic, event: PolymarketEvent, notebook: LensNotebook
) -> InjuryReport:
    out = await _run_reasoner(
        anthropic, event, notebook, INJURY_REASONER_SYSTEM, InjuryReport, "injury",
    )
    assert isinstance(out, InjuryReport)
    return out


async def reason_narrative(
    anthropic: AsyncAnthropic, event: PolymarketEvent, notebook: LensNotebook
) -> NarrativeReport:
    out = await _run_reasoner(
        anthropic, event, notebook, NARRATIVE_REASONER_SYSTEM, NarrativeReport,
        "narrative",
    )
    assert isinstance(out, NarrativeReport)
    return out


REASONERS: dict[str, ReasonerFn] = {
    "statistics": reason_statistics,
    "injury": reason_injury,
    "narrative": reason_narrative,
}
