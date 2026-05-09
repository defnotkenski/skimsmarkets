"""Generic per-lens Claude reasoner — Stage B of the two-stage agent chain.

Each reasoner takes the `LensNotebook` produced by its lens's fetcher
(provider-agnostic from the reasoner's POV) plus the same event context
the fetcher saw, and emits the typed report declared by the lens's
`LensSpec.report_schema`.

After the per-sport-lens-set refactor there is ONE generic
`run_reasoner` function — `lens_spec.report_schema` and
`lens_spec.reasoner_system` are the only inputs that vary per lens.
The legacy `REASONERS` dispatch dict (one ReasonerFn per hardcoded lens
name) is gone; iteration now goes through `lens_set.lenses`.

System prompts are cached per-lens via `CacheControlEphemeralParam` —
Anthropic caps active ephemeral breakpoints at 4 per request and we use 1
per request (one system block per call). Adding shared content across
reasoners would require its own cached block ABOVE the lens-specific
block — never concatenated, or every lens forces a fresh cache write
and the whole point of caching is lost.
"""

from __future__ import annotations

import logging

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
from skimsmarkets.agents.fetchers import render_context
from skimsmarkets.agents.schemas import LensNotebook, TokenUsage
from skimsmarkets.agents.sports.base import LensSpec
from skimsmarkets.polymarket.models import PolymarketEvent

log = logging.getLogger(__name__)


async def run_reasoner(
    anthropic: AsyncAnthropic,
    event: PolymarketEvent,
    notebook: LensNotebook,
    spec: LensSpec,
    *,
    token_sink: list[TokenUsage] | None = None,
) -> BaseModel:
    """Run one Claude reasoner for one event/lens.

    Mirrors `synthesize_prediction` in director.py: cached system block,
    adaptive thinking, max effort. The user message is the canonical event
    context (same string the fetcher saw), the per-lens user-message
    extras (`spec.render_extras`), the per-lens reasoner sport hint, and
    the notebook serialized as indented JSON.

    The output type is `spec.report_schema` — the caller is responsible
    for any concrete-type narrowing it needs (cross-sport pipeline code
    just stores `BaseModel` and lets the per-sport director path
    iso-cast).
    """
    extras = (
        spec.render_extras(event) if spec.render_extras is not None else None
    )
    extras_block = f"\n\n{extras}" if extras else ""

    sport_hint = spec.render_reasoner_hint()
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
        text=spec.reasoner_system,
        cache_control=CacheControlEphemeralParam(type="ephemeral"),
    )
    user_message = MessageParam(role="user", content=user_msg)

    # Retry once on parse-class failures (malformed JSON, no parsed output) —
    # same posture as the director. Genuine API errors raise their own
    # exception classes and bubble past unchanged.
    parsed = None
    report: BaseModel | None = None
    for attempt in range(_PARSE_RETRY_ATTEMPTS):
        try:
            parsed = await anthropic.messages.parse(
                model=CLAUDE_MODEL,
                max_tokens=CLAUDE_MAX_OUTPUT_TOKENS,
                system=[system_block],
                messages=[user_message],
                output_format=spec.report_schema,
                thinking=ThinkingConfigAdaptiveParam(type="adaptive"),
                output_config=OutputConfigParam(effort="max"),
            )
            report = parsed.parsed_output
            if report is None:
                raise RuntimeError(
                    f"Reasoner returned no parsed output for lens={spec.name} "
                    f"event={event.id}; stop_reason={parsed.stop_reason}"
                )
            break
        except (ValidationError, RuntimeError) as e:
            if attempt + 1 < _PARSE_RETRY_ATTEMPTS:
                log.warning(
                    "reasoner parse retry lens=%s event=%s attempt=%d/%d: %s",
                    spec.name, event.id, attempt + 1, _PARSE_RETRY_ATTEMPTS, e,
                )
                continue
            raise

    assert parsed is not None and report is not None  # break-path invariant
    if token_sink is not None:
        token_sink.append(TokenUsage(
            stage=f"reasoner:{spec.name}",
            provider="anthropic",
            model=CLAUDE_MODEL,
            input_tokens=parsed.usage.input_tokens,
            output_tokens=parsed.usage.output_tokens,
        ))
    log.debug(
        "reasoner=%s event=%s tokens in/out=%s/%s",
        spec.name,
        event.id,
        parsed.usage.input_tokens,
        parsed.usage.output_tokens,
    )
    return report
