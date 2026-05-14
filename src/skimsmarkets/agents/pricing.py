"""Per-model token pricing and cost computation.

The Anthropic Messages API returns raw token counts in `Usage`; there is
no dollar field in the response and no SDK-level cost endpoint. Cost has
to be computed downstream by multiplying token counts against published
per-model rates.

Rates are kept here (not in config) so a model bump (`opus-4-7` →
`opus-4-8`) is a one-line PR with the diff sitting next to the call
sites that emit `TokenUsage`. All rates are in USD per million tokens
(MTok); cost output is in USD.

`MODEL_RATES` covers Anthropic models currently in use by the ranker
pipeline + retro. Non-Anthropic providers (Grok, Gemini) are out of
scope for now — `cost_usd` returns None for unknown model IDs so the
caller can decide whether to skip or warn.

Cache pricing follows Anthropic's standard multipliers:
  - cache write (5m TTL): 1.25× base input rate
  - cache read (hit):     0.10× base input rate
We store the multiplied rates directly to keep the math at the call
site obvious; if Anthropic publishes a new tier (1h TTL is currently
2×) add a separate `cache_write_1h` rate rather than overloading the
existing field.

Extended thinking tokens are billed at the standard output rate and are
already included in `usage.output_tokens` — no separate accounting
needed. Batch API is a 50% discount applied at billing; if the pipeline
ever uses it, multiply the result by 0.5 at the call site (the raw
`Usage` object doesn't flag batch responses).
"""

from __future__ import annotations

from dataclasses import dataclass

from skimsmarkets.agents.schemas import TokenUsage


@dataclass(frozen=True)
class ModelRates:
    """Per-MTok USD rates for one model.

    All four buckets must be specified — if a model doesn't support
    prompt caching, set cache rates to the same as `input` (so the
    math stays correct if the SDK ever reports nonzero cache fields
    for that model).
    """

    input: float
    output: float
    cache_write_5m: float
    cache_read: float


# Anthropic models. Rates as of 2026-01 per
# https://www.anthropic.com/pricing — update when the model bumps or
# rates change. Cache multipliers: write_5m = 1.25× input, read = 0.10×.
MODEL_RATES: dict[str, ModelRates] = {
    "claude-opus-4-7": ModelRates(
        input=5.0,
        output=25.0,
        cache_write_5m=6.25,
        cache_read=0.50,
    ),
}


def cost_usd(usage: TokenUsage) -> float | None:
    """Compute the dollar cost of one LLM call from its `TokenUsage`.

    Returns None when the model isn't in `MODEL_RATES` (non-Anthropic
    providers in v1, or an Anthropic model we forgot to register).
    None tokens count as 0 — the SDK occasionally omits cache fields
    when no `cache_control` is on the request, and a missing bucket
    is unambiguously zero, not unknown.

    Returned value is rounded to 6 decimal places — fine-grained
    enough to distinguish single-call costs at low volumes without
    leaking float-precision noise into the JSONL.
    """
    rates = MODEL_RATES.get(usage.model)
    if rates is None:
        return None

    input_tokens = usage.input_tokens or 0
    output_tokens = usage.output_tokens or 0
    cache_write = usage.cache_creation_input_tokens or 0
    cache_read = usage.cache_read_input_tokens or 0

    cost = (
        input_tokens * rates.input
        + output_tokens * rates.output
        + cache_write * rates.cache_write_5m
        + cache_read * rates.cache_read
    ) / 1_000_000
    return round(cost, 6)
