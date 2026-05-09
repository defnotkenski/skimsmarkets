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

from skimsmarkets.agents.schemas import EventPrediction, MarketPrediction, TokenUsage
from skimsmarkets.agents.sport_hints import render_director_sport_hint
from skimsmarkets.agents.sports import DIRECTOR_SHARED_PREAMBLE
from skimsmarkets.agents.sports.base import LensSet
from skimsmarkets.polymarket.models import PolymarketEvent, PolymarketMarket
from skimsmarkets.tennis import (
    render_tennis_gbt_block,
    render_tennis_simulation_block,
)
from skimsmarkets.unusual_whales import render_uw_block

log = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-opus-4-7"
# max_tokens is required by the Messages API and must cover thinking + response
# combined. The director emits a small Pydantic object, so 16k gives adaptive
# thinking plenty of room without tripping the SDK's 10-minute non-streaming
# guardrail (which fires when max_tokens × effort implies a longer call).
CLAUDE_MAX_OUTPUT_TOKENS = 16_000

# Claude's structured-output path occasionally emits malformed JSON (observed:
# trailing comma after an empty `specialist_weights` dict) and the SDK's
# `messages.parse` raises `pydantic.ValidationError` directly. Treat as a
# transient sampling bug and retry once. Genuine API failures (auth,
# `anthropic.RateLimitError`, `anthropic.APIConnectionError`) raise their own
# exception classes that are NOT ValidationError and bubble past the loop
# unchanged. Same shape used in `agents/reasoners.py`.
_PARSE_RETRY_ATTEMPTS = 2


def _render_event_context_block(event: PolymarketEvent) -> str:
    lines = [
        f"Event: {event.id} — {event.title or '(untitled)'}",
        f"Series: {event.series_slug or '?'}",
        event.game_state_line(),
        f"Tradable sides ({len(event.markets)}):",
    ]
    # Pre-match prose context — gamma's `eventMetadata.context_description`
    # is an AI-generated paragraph (form, recent H2H, line motivation).
    # Lives in the per-event user message (NOT the cached system prompt)
    # so the prompt cache hit on DIRECTOR_SHARED_PREAMBLE + sport tail is
    # preserved.
    if event.context_description:
        lines.append(f"Pre-match context: {event.context_description}")
    for m in event.markets:
        implied = m.yes_implied_probability
        bid = f"${m.yes_bid_dollars:.3f}" if m.yes_bid_dollars is not None else "?"
        ask = f"${m.yes_ask_dollars:.3f}" if m.yes_ask_dollars is not None else "?"
        implied_str = f"{implied:.3f}" if implied is not None else "unknown"
        side_tag = " [NO side, inverted]" if m.is_no_side else ""
        # Mirror the specialist extras builder so the director sees the same
        # micro-structure signals. These are deterministic enrichments
        # (not LLM opinions) so the director should weigh them directly.
        # Order: state (only when not OPEN) → top-of-book size → full-book
        # $ → session range → 1d move → CLOB liquidity → competitive →
        # record. State and intraday range come from `markets.book`;
        # `liq`/`1d`/`comp` from the gamma piggyback; `record` from
        # events.list team payload.
        extras: list[str] = []
        if m.market_state and m.market_state != "MARKET_STATE_OPEN":
            extras.append(f"state={m.market_state.removeprefix('MARKET_STATE_')}")
        if m.yes_bid_size_top is not None or m.yes_ask_size_top is not None:
            bs = (
                f"{m.yes_bid_size_top:.0f}"
                if m.yes_bid_size_top is not None
                else "?"
            )
            asz = (
                f"{m.yes_ask_size_top:.0f}"
                if m.yes_ask_size_top is not None
                else "?"
            )
            extras.append(f"size={bs}/{asz}")
        if (
            m.yes_bid_book_dollars is not None
            or m.yes_ask_book_dollars is not None
        ):
            bb = (
                f"${m.yes_bid_book_dollars:,.0f}"
                if m.yes_bid_book_dollars is not None
                else "$?"
            )
            ab = (
                f"${m.yes_ask_book_dollars:,.0f}"
                if m.yes_ask_book_dollars is not None
                else "$?"
            )
            extras.append(f"book={bb}/{ab}")
        if (
            m.high_px_dollars is not None
            and m.low_px_dollars is not None
        ):
            extras.append(
                f"range={m.high_px_dollars - m.low_px_dollars:.3f}"
            )
        if m.gamma_liquidity_dollars is not None:
            extras.append(f"liq=${m.gamma_liquidity_dollars:,.0f}")
        if m.gamma_one_day_price_change is not None:
            extras.append(f"1d={m.gamma_one_day_price_change:+.3f}")
        if m.gamma_competitive is not None:
            extras.append(f"comp={m.gamma_competitive:.2f}")
        if m.team_record:
            extras.append(f"record={m.team_record}")
        # CLOB price-history extras (sparkline + recency scalars). `30m=`
        # is omitted from the director context to keep the per-market line
        # tight; specialists get it for the closer-in look.
        if m.clob_price_path_sparkline:
            extras.append(f"path={m.clob_price_path_sparkline}")
        if m.clob_price_change_4h is not None:
            extras.append(f"4h={m.clob_price_change_4h:+.3f}")
        if m.clob_price_change_1h is not None:
            extras.append(f"1h={m.clob_price_change_1h:+.3f}")
        extras_str = f" {' '.join(extras)}" if extras else ""
        lines.append(
            f"  - slug={m.slug}{side_tag} yes='{m.yes_sub_title or '(no label)'}' "
            f"bid/ask={bid}/{ask} implied={implied_str}{extras_str}"
        )
    # Unusual Whales flow signals reach the director as raw background data —
    # alongside bid/ask — rather than through any specialist's opinion. The
    # block is only appended when UW had coverage for this event's slug; its
    # absence is normal (most non-NBA/NFL/big-soccer events won't match).
    if event.uw_context is not None:
        lines.append("")
        lines.append(render_uw_block(event.uw_context))
    # Career-baseline Monte Carlo sim, also director-only — same posture as
    # UW. Lenses don't see this; their job is computing contextual deltas
    # (form, surface, H2H, conditions) ON TOP OF a long-run baseline, not
    # second-guessing the baseline itself. Block is present only on tennis
    # events where both players had populated career serve/return data;
    # absence is normal (non-tennis events, tennis events on stub/missing
    # vendor data).
    if event.tennis_simulation is not None:
        lines.append("")
        lines.append(render_tennis_simulation_block(event.tennis_simulation))
    # GBT third prior, also director-only — same posture as the sim.
    # Trained on point-in-time aggregated career rates + surface splits
    # + recent form + age + H2H. Director uses it as a finite-window
    # historical prior alongside the market and the iid sim.
    if event.tennis_gbt is not None:
        lines.append("")
        lines.append(render_tennis_gbt_block(event.tennis_gbt))
    return "\n".join(lines)


def _render_user_message(
    event: PolymarketEvent,
    reports: dict[str, BaseModel],
    lens_set: LensSet,
) -> str:
    """Compose the director's per-event user message.

    Reports are rendered IN THE ORDER the lens set declares so the
    director reads them in the same layout the sport-specific tail
    references. Each report block is named after its lens (e.g.
    `--- tennis_form_and_surface report ---`) so the director's prose
    can refer to lens names that match `specialist_weights` keys.
    """
    # Sport-specific contingency menu for sizing the `confidence` tier.
    # Rides on the per-event user message — NEVER the cached system block —
    # so the slate-wide cache hit is preserved.
    sport_hint = render_director_sport_hint(event)
    sport_hint_block = f"{sport_hint}\n\n" if sport_hint else ""

    report_blocks: list[str] = []
    for spec in lens_set.lenses:
        report = reports.get(spec.name)
        if report is None:
            # Defensive — `_run_lenses` should have produced one report
            # per declared lens or dropped the event before we got here.
            continue
        report_blocks.append(
            f"--- {spec.name} report ---\n{report.model_dump_json(indent=2)}"
        )
    reports_str = "\n\n".join(report_blocks)

    return (
        _render_event_context_block(event)
        + "\n\n"
        + sport_hint_block
        + reports_str
        + "\n\nReturn an EventPrediction per the schema. "
        "Set predicted_winner to the exact yes_sub_title string of the side you expect to win."
    )


def _find_market_for_winner(
    event: PolymarketEvent, winner_name: str
) -> PolymarketMarket | None:
    """Find the Polymarket side whose yes_sub_title matches the director's predicted winner."""
    target = winner_name.strip().lower()
    for m in event.markets:
        if m.yes_sub_title and m.yes_sub_title.strip().lower() == target:
            return m
    return None


def _project_to_market_prediction(
    event: PolymarketEvent,
    winner_market: PolymarketMarket,
    event_pred: EventPrediction,
    lens_set: LensSet,
) -> MarketPrediction:
    """Project the event-level prediction onto the winning side's Polymarket
    market so reporting has a single self-contained record.
    """
    return MarketPrediction(
        market_slug=winner_market.slug,
        event_id=event.id,
        event_title=event.title,
        sport_type=event.sport_type,
        lens_set_name=lens_set.sport,
        predicted_winner=event_pred.predicted_winner,
        predicted_yes_probability=event_pred.predicted_winner_probability,
        polymarket_implied_probability=winner_market.yes_implied_probability,
        confidence=event_pred.confidence,
        headline=event_pred.headline,
        reasoning=event_pred.reasoning,
        specialist_weights=event_pred.specialist_weights,
        disagreements_flagged=event_pred.disagreements_flagged,
        uw_flow_note=event_pred.uw_flow_note,
        retracted_shifts=event_pred.retracted_shifts,
    )


async def synthesize_prediction(
    anthropic: AsyncAnthropic,
    event: PolymarketEvent,
    reports: dict[str, BaseModel],
    lens_set: LensSet,
    *,
    token_sink: list[TokenUsage] | None = None,
) -> MarketPrediction:
    """Synthesize a sport's specialist reports into an event-level
    `EventPrediction`, then project onto the predicted winner's market.

    The director sends TWO cached system blocks: the sport-agnostic
    `DIRECTOR_SHARED_PREAMBLE` (cached once across the whole slate) and
    the sport-specific `lens_set.director_system_tail` (cached once per
    unique sport on the slate). Two ephemeral breakpoints — well within
    the Anthropic 4-cap; reasoners take 1, judge takes 1, no overlap.
    """
    user_msg = _render_user_message(event, reports, lens_set)

    # Order matters: the shared preamble caches once across all sports;
    # the sport-specific tail caches once per unique sport. Anthropic
    # processes cache breakpoints in order, so a fresh sport on the slate
    # produces a cache MISS on the tail block but a cache HIT on the
    # preamble — net cost of the new sport is just the tail's tokens.
    shared_block = TextBlockParam(
        type="text",
        text=DIRECTOR_SHARED_PREAMBLE,
        cache_control=CacheControlEphemeralParam(type="ephemeral"),
    )
    sport_tail_block = TextBlockParam(
        type="text",
        text=lens_set.director_system_tail,
        cache_control=CacheControlEphemeralParam(type="ephemeral"),
    )
    user_message = MessageParam(role="user", content=user_msg)

    # Retry once on parse-class failures (malformed JSON, no parsed output).
    # `parsed` and `event_pred` are set on the success path before break.
    parsed = None
    event_pred = None
    for attempt in range(_PARSE_RETRY_ATTEMPTS):
        try:
            parsed = await anthropic.messages.parse(
                model=CLAUDE_MODEL,
                max_tokens=CLAUDE_MAX_OUTPUT_TOKENS,
                system=[shared_block, sport_tail_block],
                messages=[user_message],
                output_format=EventPrediction,
                # Opus 4.7 only supports adaptive thinking. `effort` is NOT inside the
                # thinking dict — it's a sibling field under `output_config`. "max" lets
                # the model spend unconstrained reasoning budget per event.
                thinking=ThinkingConfigAdaptiveParam(type="adaptive"),
                output_config=OutputConfigParam(effort="max"),
            )
            event_pred = parsed.parsed_output
            if event_pred is None:
                raise RuntimeError(
                    f"Director returned no parsed output for event {event.id}; "
                    f"stop_reason={parsed.stop_reason}"
                )
            break
        except (ValidationError, RuntimeError) as e:
            if attempt + 1 < _PARSE_RETRY_ATTEMPTS:
                log.warning(
                    "director parse retry event=%s attempt=%d/%d: %s",
                    event.id, attempt + 1, _PARSE_RETRY_ATTEMPTS, e,
                )
                continue
            raise

    assert parsed is not None and event_pred is not None  # break-path invariant
    if token_sink is not None:
        token_sink.append(TokenUsage(
            stage="director",
            provider="anthropic",
            model=CLAUDE_MODEL,
            input_tokens=parsed.usage.input_tokens,
            output_tokens=parsed.usage.output_tokens,
        ))
    log.debug(
        "director event=%s sport=%s tokens in/out=%s/%s",
        event.id,
        lens_set.sport,
        parsed.usage.input_tokens,
        parsed.usage.output_tokens,
    )

    winner_market = _find_market_for_winner(event, event_pred.predicted_winner)
    if winner_market is None:
        raise RuntimeError(
            f"Director's predicted_winner={event_pred.predicted_winner!r} did not match "
            f"any yes_sub_title in event {event.id}. "
            f"Known sides: {[m.yes_sub_title for m in event.markets]}"
        )

    return _project_to_market_prediction(
        event, winner_market, event_pred, lens_set
    )
