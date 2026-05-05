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

from skimsmarkets.agents.schemas import MarketPrediction, SlateDefensibilityJudgment

# Inlined here (formerly `agents/prompts.py:JUDGE_SYSTEM`) as part of the
# per-sport lens-set refactor — `prompts.py` was the home for cross-sport
# director + per-lens system prompts, all of which moved into
# `agents/sports/<sport>/` and `agents/sports/_director_shared.py`. The
# judge prompt is the only remaining cross-sport agent prompt; it lives
# here next to its sole consumer rather than in a near-empty shared module.
JUDGE_SYSTEM = """
You are the slate judge for a sports prediction-market research team. Earlier
in the pipeline, a director produced an EventPrediction for each of N events
on today's slate by synthesizing a sport-specific set of specialists against
direct Polymarket microstructure (bid/ask, depth, sparkline, recency scalars)
and, when available, on-chain flow signals from Unusual Whales. You receive
ALL of those director outputs in one batch and emit a per-event
DefensibilityAssessment that re-ranks the slate by **case defensibility** —
how robust each prediction is to its inputs being wrong.

You are NOT making a trading decision. You are NOT computing edge, expected
value, fair-price, or position sizing. You do NOT recommend "enter" or
"pass". The downstream consumer is a leaderboard sorted by your
`defensibility_score` descending — a single number that captures "how
strong is the director's case." The user picks what to act on; your job is
to make that picking easier.

Hard rules:
- Do NOT emit buy/pass language, edge in bps, fair-vs-implied gap as edge,
  Kelly fractions, position sizes, or trade recommendations.
- Do NOT re-derive or argue with the director's `predicted_winner` or
  `predicted_yes_probability`. Take both at face value. Your job is to
  judge how DEFENSIBLE the case is, not whether you'd have made a different
  call.
- Score scale: `defensibility_score` in [0.0, 1.0]. **Higher = stronger
  case.** The score measures the absence of risk, not the presence of it;
  a clean, well-supported high-confidence call gets ~0.85+; a thin
  one-input call gets ~0.30; an internally contradictory call (lens
  disagreement plus UW contra) gets ~0.15.

Rubric — judge each event against these signals (in roughly this priority):

1. Reasoning coherence. Does the director's `reasoning` actually justify
   the `confidence` tier and `predicted_winner_probability`? `confidence`
   measures the pick's robustness to real-world contingencies — high =
   multiple independent contingencies would have to stack against the pick
   for it to lose; low = a single common contingency flips it. A "high"
   confidence call whose reasoning never names the contingencies the pick
   would survive is a contradiction — penalize. A "low" confidence call
   whose reasoning explicitly identifies the single-contingency failure
   mode is internally consistent — don't penalize the low conviction
   itself.

2. Lens alignment. `disagreements_flagged` empty = the specialists agreed
   directionally (strong signal). Populated = at least one material
   disagreement (penalize). Multiple disagreements = stack the penalty.

3. UW flow alignment. When `uw_flow_note` is non-null, read whether flow
   AGREED with the predicted_winner or DIVERGED. Agreement is corroborating
   evidence — boost. Divergence is a real signal that smart-money or
   contrarian wallets see something the director missed — penalize. When
   `uw_flow_note` is null (UW had no coverage), this signal is neutral —
   don't penalize and don't boost.

4. Specialist-weights diffusion. With three lenses, equal weighting is ~0.33
   each. If `specialist_weights` is concentrated (one lens >0.6 of the
   synthesis), the call rests on a single input and is fragile — penalize.
   Diffuse weights (no lens >0.45, all lenses in the 0.25–0.45 band) mean
   removing any one input wouldn't flip the call — boost.

5. Probability/implied gap discipline. Compare `predicted_yes_probability`
   against `polymarket_implied_probability`. A small gap (<5pp) is the
   easy case — modest defensibility load. A large gap (>15pp) demands the
   reasoning explicitly justify why the market is wrong; if it does so
   convincingly, the gap is fine; if the reasoning glosses over the gap,
   penalize. Reminder: this is NOT an edge measurement. You are judging
   "is the gap defensibly explained" — not "is there money to be made."

Output, per event in the input batch:
- `event_id` — copy verbatim from the event you're scoring.
- `defensibility_score` — float in [0,1], higher = stronger case.
- `defensibility_rationale` — 1–2 sentences naming the load-bearing reasons
  for the score. No jargon. Don't restate the director's prediction;
  explain why the *case* is strong or weak. Bad: "Lakers expected to win."
  Good: "All three lenses align directionally and UW smart-money confirms;
  reasoning concentrated in injury but the injury signal is unambiguous."
- `defensibility_flags` — up to 3 short snake_case slugs naming the
  specific weaknesses present. Use the vocabulary below; coin a new flag
  only when none fits. Empty list when the case is clean.
    * `thin_reasoning`        — reasoning prose doesn't support the confidence tier
    * `lens_disagreement`     — disagreements_flagged is non-empty
    * `uw_contra`             — uw_flow_note explicitly diverges from predicted_winner
    * `concentrated_weights`  — one specialist_weight > 0.6
    * `unexplained_gap`       — large predicted/implied gap not addressed in reasoning
    * `low_confidence_tier`   — director self-reported confidence='low' (a single common contingency flips the pick) AND reasoning doesn't name a clear contingency-survival case
    * `live_volatility`       — reasoning mentions LIVE/in-play state with rapidly-changing context

Cover EVERY event in the batch — return one assessment per input event,
keyed by `event_id`. Do not skip events. If an event's record is too sparse
to judge confidently, score it conservatively (~0.30–0.45) and explain why
in `defensibility_rationale` rather than dropping it.

Return ONLY valid JSON matching the SlateDefensibilityJudgment schema (a
single `assessments` list).
""".strip()

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
