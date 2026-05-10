"""Step 3 — batched LLM pattern analysis (one call per sport).

Takes a flat list of `EventFeatures` (wins + losses both, labeled),
compresses each into a compact CSV-shaped row in the user message,
and asks Claude Opus to find patterns OVERREPRESENTED IN LOSSES vs.
wins. Anti-hindsight discipline: the prompt forbids per-event
explanations and demands cross-batch patterns.

Cached system prompt (1 ephemeral breakpoint) carries the analysis
framing + lens-set vocabulary; the per-run feature table rides on
the user message so the cache hit is preserved across reruns.
Pydantic-constrained output via `messages.parse` — same shape as
the director call.

Failure semantics: any LLM failure logs a warning and returns an
empty `RetroFindings` for that sport. The CLI still prints whatever
calibrate produced — Step 3 is additive, not load-bearing.
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
from pydantic import ValidationError

from skimsmarkets.retro.models import EventFeatures, RetroFindings

log = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-opus-4-7"
CLAUDE_MAX_OUTPUT_TOKENS = 16_000

# Same retry posture as the director / reasoners: one in-place retry
# on parse-class failures (malformed JSON, empty parsed_output).
# Genuine API errors (auth, RateLimitError) bubble past unchanged.
_PARSE_RETRY_ATTEMPTS = 2

# Anti-hindsight system prompt. Cached once per analyze call; only the
# user message (the feature table) varies between sports. Lens vocabulary
# is the only sport-coupled bit and lives in the user message so the
# cache hits across sports as the system gets sharded later.
_SYSTEM_PROMPT = """\
You are a retrospective analysis specialist for a Polymarket sports
prediction system. You are given a CSV-shaped table of past events,
each labeled `won=true` or `won=false`. Your job is to find features
that are OVERREPRESENTED IN LOSSES vs WINS — patterns that, if the
system noticed them at prediction time, would have prevented losses.

DISCIPLINE — read carefully:

1. DO NOT explain individual losses. Per-event explanations after
   the fact are hindsight bias — anything plausible-sounding can be
   constructed for any single loss. Only report patterns that recur
   across MULTIPLE losses AND are clearly LESS frequent in wins.

2. DO NOT report base-rate facts about the slate. "Many losses were
   on clay" is meaningless if most of the slate is on clay. Only
   features whose distribution differs between wins and losses are
   signal.

3. DO NOT recommend changes you can't justify with the data shown.
   "The system overweights ranking" is a recommendation; "the system
   should weight ranking less" without the data to support it is not.

4. The `divergence_*` columns capture how a player's actual match
   performance differed from their pre-match career baseline
   (actual - baseline; positive = overperformed). Large negative
   divergence on the predicted side, paired with a loss, is the
   high-signal pattern: the player underperformed and we didn't see
   it coming. Look for whether ANY pre-match feature (surface,
   confidence tier, market favorite/underdog, defensibility score)
   correlates with these collapses.

5. When you attribute a recurring failure to a lens, name the lens
   exactly as it appears in the sport's registered LensSet
   (e.g. for tennis: tennis_form_and_surface,
   tennis_matchup_and_clutch, tennis_conditions_and_context). Do
   not invent lens names.

6. Recommendations are for human review by the operator, who will
   manually translate them into prompt edits to DIRECTOR_SPORT_HINTS,
   LensSpec.reasoner_sport_hint, or LensSet.director_system_tail.
   Recommendations should be concrete (a sentence or two each), not
   generic platitudes.

Output a `RetroFindings` object with:
- `recurring_patterns`: 3-7 short bullets, each naming one
  loss-overrepresented feature with its rough magnitude.
- `lens_underperformance`: dict mapping lens name → recurring failure
  mode. Empty when no clear lens attribution emerges from the data.
- `prompt_recommendations`: 2-5 concrete edits worth considering.
"""


def _row_to_csv(f: EventFeatures) -> str:
    """One feature row as a compact comma-separated string.

    Float fields are rounded to 3 decimals for token economy. None
    values render as empty (CSV-canonical), distinguishable from
    zero. Field order matches the header in `_render_user_message`.
    """

    def _f(v: float | None, digits: int = 3) -> str:
        if v is None:
            return ""
        return f"{v:.{digits}f}"

    def _s(v: object) -> str:
        if v is None:
            return ""
        s = str(v)
        # Defensive against commas / newlines inside string fields
        # (event titles can carry colons but rarely commas; quote
        # anyway to keep the table parseable as CSV).
        if "," in s or '"' in s or "\n" in s:
            s = '"' + s.replace('"', '""') + '"'
        return s

    return ",".join([
        _s(f.event_id),
        _s(f.event_title),
        _s(f.sport_type),
        _s(f.surface),
        _s(f.predicted_winner),
        _f(f.predicted_prob),
        _f(f.market_implied_prob),
        _s(f.confidence),
        _f(f.defensibility_score),
        _s(f.case_bucket),
        _s(f.market_favorite_pick),
        _s(f.negative_edge),
        _s(f.won),
        _f(f.baseline_first_serve_in_pct_a),
        _f(f.actual_first_serve_in_pct_a),
        _f(f.divergence_first_serve_in_a),
        _f(f.baseline_first_serve_win_pct_a),
        _f(f.actual_first_serve_win_pct_a),
        _f(f.divergence_first_serve_win_a),
        _f(f.baseline_second_serve_win_pct_a),
        _f(f.actual_second_serve_win_pct_a),
        _f(f.divergence_second_serve_win_a),
        _f(f.baseline_bp_convert_pct_a),
        _f(f.actual_bp_convert_pct_a),
        _f(f.divergence_bp_convert_a),
        _f(f.baseline_first_serve_in_pct_b),
        _f(f.actual_first_serve_in_pct_b),
        _f(f.divergence_first_serve_in_b),
        _f(f.baseline_first_serve_win_pct_b),
        _f(f.actual_first_serve_win_pct_b),
        _f(f.divergence_first_serve_win_b),
        _f(f.baseline_second_serve_win_pct_b),
        _f(f.actual_second_serve_win_pct_b),
        _f(f.divergence_second_serve_win_b),
        _f(f.baseline_bp_convert_pct_b),
        _f(f.actual_bp_convert_pct_b),
        _f(f.divergence_bp_convert_b),
    ])


_CSV_HEADER = (
    "event_id,event_title,sport_type,surface,predicted_winner,"
    "predicted_prob,market_implied_prob,confidence,defensibility_score,"
    "case_bucket,market_favorite_pick,negative_edge,won,"
    "baseline_first_serve_in_pct_a,actual_first_serve_in_pct_a,"
    "divergence_first_serve_in_a,"
    "baseline_first_serve_win_pct_a,actual_first_serve_win_pct_a,"
    "divergence_first_serve_win_a,"
    "baseline_second_serve_win_pct_a,actual_second_serve_win_pct_a,"
    "divergence_second_serve_win_a,"
    "baseline_bp_convert_pct_a,actual_bp_convert_pct_a,"
    "divergence_bp_convert_a,"
    "baseline_first_serve_in_pct_b,actual_first_serve_in_pct_b,"
    "divergence_first_serve_in_b,"
    "baseline_first_serve_win_pct_b,actual_first_serve_win_pct_b,"
    "divergence_first_serve_win_b,"
    "baseline_second_serve_win_pct_b,actual_second_serve_win_pct_b,"
    "divergence_second_serve_win_b,"
    "baseline_bp_convert_pct_b,actual_bp_convert_pct_b,"
    "divergence_bp_convert_b"
)


def _render_user_message(
    sport: str, feats: list[EventFeatures]
) -> str:
    n_wins = sum(1 for f in feats if f.won is True)
    n_losses = sum(1 for f in feats if f.won is False)
    lines = [
        f"Sport: {sport}",
        f"Total settled events: {len(feats)} ({n_wins} wins, {n_losses} losses)",
        "",
        "Event feature table (CSV; '_a' = predicted winner, '_b' = opponent;",
        "divergence = actual - baseline, positive = player overperformed):",
        "",
        _CSV_HEADER,
    ]
    for f in feats:
        lines.append(_row_to_csv(f))
    lines.extend([
        "",
        "Find patterns OVERREPRESENTED IN LOSSES vs WINS. Do not explain "
        "individual losses. Do not flag base-rate facts about the slate. "
        "Name lenses exactly when attributing failures.",
    ])
    return "\n".join(lines)


async def analyze_sport(
    anthropic: AsyncAnthropic,
    sport: str,
    feats: list[EventFeatures],
) -> RetroFindings:
    """One LLM call per sport. Filters to settled events with at least
    one of `won=True` and `won=False` — single-class data is useless
    for differential pattern-finding.
    """
    settled = [f for f in feats if f.settled and f.won is not None]
    n_wins = sum(1 for f in settled if f.won is True)
    n_losses = sum(1 for f in settled if f.won is False)
    if n_wins == 0 or n_losses == 0:
        log.info(
            "retro analyze: %s skipped — need both wins and losses "
            "(have %d/%d)", sport, n_wins, n_losses,
        )
        return RetroFindings(
            sport=sport,
            n_events=len(settled),
            n_wins=n_wins,
            n_losses=n_losses,
        )

    user_msg = _render_user_message(sport, settled)
    system_block = TextBlockParam(
        type="text",
        text=_SYSTEM_PROMPT,
        cache_control=CacheControlEphemeralParam(type="ephemeral"),
    )
    user_message = MessageParam(role="user", content=user_msg)

    parsed = None
    findings = None
    for attempt in range(_PARSE_RETRY_ATTEMPTS):
        try:
            parsed = await anthropic.messages.parse(
                model=CLAUDE_MODEL,
                max_tokens=CLAUDE_MAX_OUTPUT_TOKENS,
                system=[system_block],
                messages=[user_message],
                output_format=RetroFindings,
                thinking=ThinkingConfigAdaptiveParam(type="adaptive"),
                output_config=OutputConfigParam(effort="max"),
            )
            findings = parsed.parsed_output
            if findings is None:
                raise RuntimeError(
                    f"retro analyze: no parsed output for sport={sport!r}; "
                    f"stop_reason={parsed.stop_reason}"
                )
            break
        except (ValidationError, RuntimeError) as e:
            if attempt + 1 < _PARSE_RETRY_ATTEMPTS:
                log.warning(
                    "retro analyze parse retry sport=%s attempt=%d/%d: %s",
                    sport, attempt + 1, _PARSE_RETRY_ATTEMPTS, e,
                )
                continue
            log.warning(
                "retro analyze: sport=%s failed after %d attempts: %s",
                sport, _PARSE_RETRY_ATTEMPTS, e,
            )
            return RetroFindings(
                sport=sport,
                n_events=len(settled),
                n_wins=n_wins,
                n_losses=n_losses,
            )

    assert parsed is not None and findings is not None
    log.info(
        "retro analyze sport=%s tokens in/out=%s/%s",
        sport, parsed.usage.input_tokens, parsed.usage.output_tokens,
    )
    # Stamp the actual N values regardless of what the model wrote
    # — the model can hallucinate counts even when the table is
    # right in front of it.
    findings.sport = sport
    findings.n_events = len(settled)
    findings.n_wins = n_wins
    findings.n_losses = n_losses
    return findings


async def analyze_all_sports(
    anthropic: AsyncAnthropic,
    feats: list[EventFeatures],
    sports_filter: set[str] | None = None,
) -> dict[str, RetroFindings]:
    """Run `analyze_sport` for each sport found in the feature list.

    `sports_filter`: when provided, only sports in this set are
    analyzed (useful for `--sport tennis` flag). When None, every
    sport with both wins and losses gets a call. Sports without
    enough samples for differential analysis return an empty
    `RetroFindings` (no LLM cost paid).
    """
    by_sport: dict[str, list[EventFeatures]] = {}
    for f in feats:
        sport = f.sport_type or "unknown"
        if sports_filter is not None and sport not in sports_filter:
            continue
        by_sport.setdefault(sport, []).append(f)

    out: dict[str, RetroFindings] = {}
    # Sequential rather than gather() — one analyze call per sport is
    # already big and parallel execution would just contend on the
    # Anthropic rate limit. If sport count grows past ~5 we can
    # revisit.
    for sport in sorted(by_sport):
        out[sport] = await analyze_sport(anthropic, sport, by_sport[sport])
    return out
