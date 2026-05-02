"""Slate-level case-defensibility judge.

Runs ONCE per pipeline run, after all per-event directors finish. Reads
each event's `MarketPrediction` and emits a `SlateDefensibilityJudgment`
with one `DefensibilityAssessment` per event, scoring **case
defensibility** (not edge / not EV — see CLAUDE.md "confidence ranker,
not edge finder"). The judge's `defensibility_score` replaces
`predicted_yes_probability` as the leaderboard's primary sort key.

Naming intentionally measures the *absence* of risk (defensibility) so
"higher = better" matches the leaderboard's descending-sort direction
without inversion confusion.

Inputs are reasoning-only — the judge sees the director's *synthesis*
(`reasoning`, `confidence`, `disagreements_flagged`, `uw_flow_note`,
`specialist_weights`) plus the predicted/implied probability gap, but NOT
the upstream raw notebooks/specialist reports or market microstructure
(spread, book depth). That keeps the judge's job narrow: "is the case
the director made internally coherent and well-supported," not "is this a
good bet."

Failure posture: a judge call failure is recorded as one slate-level
`ErrorRecord` and the pipeline continues with
`result.defensibility_assessments` empty — the leaderboard falls back to
predicted-probability sort. Mirrors the silent-degrade enrichment-stage
posture documented in CLAUDE.md.
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

from skimsmarkets.agents.prompts import JUDGE_SYSTEM
from skimsmarkets.agents.schemas import MarketPrediction, SlateDefensibilityJudgment

log = logging.getLogger(__name__)

# Match the director's quality bar — the judge is reading the same kind of
# prose the director emitted. Sonnet would be cheaper but the judge's task
# (cross-event coherence assessment) benefits from the full Opus reasoning
# budget, especially when the slate hits the `MAX_SLATE_EVENTS` cap.
CLAUDE_MODEL = "claude-opus-4-7"
# 16k matches the director's ceiling. The judge emits a list of compact
# DefensibilityAssessment objects (~50 tokens each), so even a 50-event
# slate fits comfortably; the headroom is for adaptive thinking.
CLAUDE_MAX_OUTPUT_TOKENS = 16_000


def _render_event_block(p: MarketPrediction) -> str:
    """Compact, scannable block per event for the judge's user message.

    Reasoning-only inputs (per design): the director's synthesis, the
    confidence tier, and the predicted/implied probability gap. The raw
    upstream evidence (notebooks, specialist reports) and market
    microstructure are intentionally omitted — the judge's job is to score
    the *case*, not to second-guess the director's evidence.
    """
    title = p.event_title or p.event_id
    implied = (
        f"{p.polymarket_implied_probability:.3f}"
        if p.polymarket_implied_probability is not None
        else "unknown"
    )
    gap_pp = (
        (p.predicted_yes_probability - p.polymarket_implied_probability) * 100.0
        if p.polymarket_implied_probability is not None
        else None
    )
    gap_str = f"{gap_pp:+.1f}pp" if gap_pp is not None else "n/a (no implied)"
    weights_str = ", ".join(
        f"{k}={v:.2f}" for k, v in sorted(p.specialist_weights.items())
    )
    disagreements = (
        "; ".join(p.disagreements_flagged) if p.disagreements_flagged else "(none)"
    )
    uw = p.uw_flow_note if p.uw_flow_note else "(no UW coverage)"
    return (
        f"event_id: {p.event_id}\n"
        f"event_title: {title}\n"
        f"predicted_winner: {p.predicted_winner}\n"
        f"predicted_yes_probability: {p.predicted_yes_probability:.3f}\n"
        f"polymarket_implied_probability: {implied}\n"
        f"predicted_minus_implied: {gap_str}\n"
        f"confidence_tier: {p.confidence}\n"
        f"specialist_weights: {weights_str}\n"
        f"disagreements_flagged: {disagreements}\n"
        f"uw_flow_note: {uw}\n"
        f"headline: {p.headline}\n"
        f"reasoning: {p.reasoning}"
    )


def _render_user_message(predictions: list[MarketPrediction]) -> str:
    """Numbered list of compact per-event blocks. The trailing instruction is
    intentionally short — the rubric lives in the cached system prompt.
    """
    blocks = [
        f"--- Event {i + 1} of {len(predictions)} ---\n{_render_event_block(p)}"
        for i, p in enumerate(predictions)
    ]
    return (
        f"You are judging a slate of {len(predictions)} event(s). "
        "Score each by case defensibility per the system prompt's rubric.\n\n"
        + "\n\n".join(blocks)
        + "\n\nReturn ONE DefensibilityAssessment per event, keyed by "
        "`event_id`, in a SlateDefensibilityJudgment.assessments list."
    )


async def judge_slate(
    anthropic: AsyncAnthropic,
    predictions: list[MarketPrediction],
) -> SlateDefensibilityJudgment:
    """Single LLM call that scores every prediction in the slate.

    Caller is responsible for guarding the empty-slate case (we don't fire
    the call when there's nothing to judge) and for catching exceptions —
    the pipeline records a slate-level `ErrorRecord` and degrades to the
    predicted-probability sort on failure.
    """
    user_msg = _render_user_message(predictions)

    system_block = TextBlockParam(
        type="text",
        text=JUDGE_SYSTEM,
        cache_control=CacheControlEphemeralParam(type="ephemeral"),
    )
    user_message = MessageParam(role="user", content=user_msg)

    parsed = await anthropic.messages.parse(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_OUTPUT_TOKENS,
        system=[system_block],
        messages=[user_message],
        output_format=SlateDefensibilityJudgment,
        thinking=ThinkingConfigAdaptiveParam(type="adaptive"),
        output_config=OutputConfigParam(effort="max"),
    )
    judgment = parsed.parsed_output
    if judgment is None:
        raise RuntimeError(
            f"Judge returned no parsed output; stop_reason={parsed.stop_reason}"
        )
    log.info(
        "judge scored %d/%d events; tokens in/out=%s/%s",
        len(judgment.assessments),
        len(predictions),
        parsed.usage.input_tokens,
        parsed.usage.output_tokens,
    )
    return judgment
