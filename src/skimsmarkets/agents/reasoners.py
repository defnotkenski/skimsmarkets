"""Per-lens Claude reasoners — Stage B of the two-stage agent chain.

Each reasoner takes the `LensNotebook` produced by its lens's fetcher
(Grok or Gemini, transparent to the reasoner) plus the same event context
the fetcher saw, and emits the typed report
(`StatisticsReport`, `InjuryReport`, `NarrativeReport`, `MarketContextReport`)
that the director consumes. Verdicts (probability, signed shift,
motivation_edge, sharp_money_signal) live here, not in the notebook.

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
from pydantic import BaseModel

from skimsmarkets.agents.director import CLAUDE_MAX_OUTPUT_TOKENS, CLAUDE_MODEL
from skimsmarkets.agents.fetchers import render_context, render_lens_extras
from skimsmarkets.agents.prompts import (
    INJURY_REASONER_SYSTEM,
    MARKET_CONTEXT_REASONER_SYSTEM,
    NARRATIVE_REASONER_SYSTEM,
    STATISTICS_REASONER_SYSTEM,
)
from skimsmarkets.agents.schemas import (
    InjuryReport,
    LensName,
    LensNotebook,
    MarketContextReport,
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
    """Shared body for the four lens reasoners.

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
    user_msg = (
        render_context(event)
        + extras_block
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


async def reason_market_context(
    anthropic: AsyncAnthropic, event: PolymarketEvent, notebook: LensNotebook
) -> MarketContextReport:
    out = await _run_reasoner(
        anthropic, event, notebook, MARKET_CONTEXT_REASONER_SYSTEM, MarketContextReport,
        "market_context",
    )
    assert isinstance(out, MarketContextReport)
    return out


REASONERS: dict[str, ReasonerFn] = {
    "statistics": reason_statistics,
    "injury": reason_injury,
    "narrative": reason_narrative,
    "market_context": reason_market_context,
}
