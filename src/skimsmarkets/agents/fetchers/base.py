"""Provider-agnostic fetcher core — the Protocol every provider implements,
plus the user-message rendering and lens-dispatch helpers shared across
providers.

The fetcher's job is evidence capture, not judgment — it emits a
`LensNotebook` (free-form prose + citations + computed numbers) and never
commits to a probability or directional verdict. The verdict lives in the
downstream Claude reasoner (see `agents/reasoners.py`). Provider files
(`grok.py`, `gemini.py`) implement `FetcherProvider` by wiring their SDK
to per-(sport, lens) system prompts pre-built at construction by
iterating `SPORT_LENS_SETS`.

Per-sport-lens-set refactor: the legacy `LENS_PROMPT_BUILDERS` dict
(keyed by `LensName` Literal) is gone — each `LensSpec` now owns its
own `fetcher_system_builder`. The legacy `render_lens_extras` switch is
also gone — each `LensSpec.render_extras` callable produces the per-lens
user-message append.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Protocol

from skimsmarkets.agents.schemas import LensNotebook, TokenUsage
from skimsmarkets.agents.sports.base import LensSet, LensSpec
from skimsmarkets.polymarket.models import PolymarketEvent, PolymarketMarket

log = logging.getLogger(__name__)


def build_lens_prompts_for_set(
    lens_set: LensSet,
    tools_by_lens: dict[str, str],
    notebook_tail: str,
) -> dict[str, str]:
    """Pre-build per-lens system prompts for ONE sport's lens set.

    `tools_by_lens` maps the lens's name → that provider's per-lens
    "What each tool can give you here" prose. `notebook_tail` is the
    provider's generic tool list + output rules. Each lens's
    `fetcher_system_builder` builds the cached fetcher system prompt
    by composing the lens-specific preamble (which describes the
    lens's *job*) with these provider-specific bits (which name the
    provider's tools).

    Lens names are unique across the entire registry (tennis lens names
    don't collide with default lens names), so providers maintain a
    flat `_TOOLS_BY_LENS` dict keyed by full lens name and accumulate
    entries as new sports ship.
    """
    return {
        spec.name: spec.fetcher_system_builder(
            tools_by_lens[spec.name], notebook_tail
        )
        for spec in lens_set.lenses
    }


class FetcherProvider(Protocol):
    """A provider that runs the per-lens fetch stage.

    `name` and `model` are persisted to the per-run JSONL (top-level row
    metadata) so retrospective grading can group hit-rate by provider /
    model version. `fetch` runs one lens for one event in the named
    sport's lens set and returns the parsed `LensNotebook`; lens-mismatch
    validation happens inside the provider via `assert_lens_match` so
    prompt-mixup bugs fail loud at fetch time.

    `lens_set` is passed alongside `lens` so the provider can resolve
    the cached system prompt by `(sport, lens_name)` without re-doing
    dispatch — the pipeline already resolved it once via
    `agents.sports.resolve_lens_set` at the lens-dispatch stage.
    """

    name: str
    model: str

    async def fetch(
        self,
        event: PolymarketEvent,
        lens: str,
        *,
        lens_set: LensSet,
        token_sink: list[TokenUsage] | None = None,
    ) -> LensNotebook: ...

    async def aclose(self) -> None: ...


def assert_lens_match(parsed: LensNotebook, expected: str, event_id: str) -> None:
    """Fail loud when a fetcher returns the wrong `lens` discriminator.

    Catches prompt-mixup bugs at fetch time rather than letting them
    silently break downstream reasoner dispatch. Both providers call this
    immediately after parsing.
    """
    if parsed.lens != expected:
        raise RuntimeError(
            f"fetcher lens mismatch for event {event_id}: expected {expected!r}, "
            f"got {parsed.lens!r}"
        )


def pick_team_a_market(event: PolymarketEvent) -> PolymarketMarket | None:
    """team_a = the Polymarket favorite (highest yes_implied_probability) among the
    event's tradable sides.

    Returns None when no market has a label and a valid implied probability. Head-to-
    head events have two sides after Polymarket's NO-side expansion; 3-way soccer
    events have three. Either way, the favorite bubbles to the top of a simple sort.
    """
    scored = [
        (m.yes_implied_probability or -1.0, m) for m in event.markets if m.yes_sub_title
    ]
    if not scored:
        return None
    scored.sort(key=lambda s: s[0], reverse=True)
    top_prob, top_market = scored[0]
    if top_prob < 0:
        return None
    return top_market


def render_context(event: PolymarketEvent) -> str:
    """Event-level user message handed to every specialist.

    Names team_a and team_b (the first non-team_a side) using the exact
    yes_sub_title strings so specialists echo them back verbatim. The LLM
    stages are blind to the market price: no bid/ask, implied probability,
    or price history is rendered — only non-directional activity (volume,
    open interest, liquidity) and the team record. See CLAUDE.md's
    market-blindness invariant.
    """
    team_a_market = pick_team_a_market(event)
    team_b_market = next(
        (m for m in event.markets if m is not team_a_market and m.yes_sub_title),
        None,
    )

    team_a_name = team_a_market.yes_sub_title if team_a_market else "(unknown)"
    team_b_name = team_b_market.yes_sub_title if team_b_market else "(unknown)"

    market_lines: list[str] = []
    for m in event.markets:
        extras: list[str] = []
        # State first — when not OPEN it's load-bearing (don't trust the
        # price). When OPEN, omit to keep the line tight; absence implies
        # nominal tradability.
        if m.market_state and m.market_state != "MARKET_STATE_OPEN":
            # Strip the `MARKET_STATE_` prefix so the LLM sees a clean tag
            # like `SUSPENDED` / `HALTED` / `MATCH_AND_CLOSE_AUCTION`.
            extras.append(f"state={m.market_state.removeprefix('MARKET_STATE_')}")
        # Non-directional activity only — volume, open interest, and resting
        # liquidity carry no read on which side the market favours, so they
        # don't anchor the LLM's independent estimate. Everything price-
        # derived (bid/ask, implied, spread, book shape, price history) is
        # deliberately omitted; see CLAUDE.md's market-blindness invariant.
        if m.volume_dollars is not None:
            extras.append(f"vol=${m.volume_dollars:,.0f}")
        if m.open_interest_dollars is not None:
            # "oi" (open interest) — outstanding shares × price, NOT
            # order-book depth. Polymarket's real CLOB liquidity arrives via
            # `liq=$...` below when the gamma piggyback ran.
            extras.append(f"oi=${m.open_interest_dollars:,.0f}")
        if m.gamma_liquidity_dollars is not None:
            # `liq` is gamma's `liquidityClob` — the actual dollars sitting
            # on the order book, distinct from `oi` (open interest). Both
            # are surfaced so the LLM sees "how much is held" and "how much
            # can I trade right now" as separate signals.
            extras.append(f"liq=${m.gamma_liquidity_dollars:,.0f}")
        if m.team_record:
            # W/L record for this side's team (e.g. "28-6"). Saves the
            # specialist a web-search round trip for season form.
            extras.append(f"record={m.team_record}")
        # [NO side, inverted] flags head-to-head markets whose side is a
        # synthesized NO clone of the slug's YES book, not a directly-quoted
        # second market.
        side_tag = " [NO side, inverted]" if m.is_no_side else ""
        extras_str = f" {' '.join(extras)}" if extras else ""
        market_lines.append(
            f"  - slug={m.slug}{side_tag} yes='{m.yes_sub_title or '(no label)'}'"
            f"{extras_str}"
        )

    # Walrus-bind so the `is not None` check narrows `t` to `datetime` inside
    # the comprehension — without it, static checkers keep the element type as
    # `datetime | None` and flag `min(...)` / `.isoformat()` downstream.
    start_times = [
        t for m in event.markets if (t := m.game_start_time) is not None
    ]
    tipoff = min(start_times).isoformat() if start_times else "(unknown)"

    # Game-state line (PRE-MATCH / LIVE / ENDED) is rendered whenever Polymarket
    # provides it — making the state explicit beats having the LLM infer phase
    # from absent fields.
    state_line = event.game_state_line()

    return (
        f"Event: {event.id} — {event.title or '(no title)'}\n"
        f"Series: {event.series_slug or '(unknown)'}\n"
        f"Tipoff: {tipoff}\n\n"
        f"{state_line}\n\n"
        f"team_a_name = {team_a_name}   (the reference side — echo this string verbatim)\n"
        f"team_b_name = {team_b_name}\n\n"
        f"Tradable sides on Polymarket ({len(event.markets)}):\n"
        + "\n".join(market_lines)
        + "\n\n"
        + "Produce your report now, per the schema. "
        "Use the exact team_a_name / team_b_name strings above in your output."
    )


def render_user_message_for_lens(
    event: PolymarketEvent, spec: LensSpec
) -> str:
    """Compose the per-lens user message a fetcher actually sends.

    Order: cross-lens event context, then per-lens fetcher sport hint,
    then per-lens render_extras (e.g. tennis stats block on
    `tennis_form_and_surface`). All rides on the user message — never the
    cached system block.
    """
    user_msg = render_context(event)
    if (sport_hint := spec.render_fetcher_hint()) is not None:
        user_msg += "\n\n" + sport_hint
    if spec.render_extras is not None and (
        extras := spec.render_extras(event)
    ) is not None:
        user_msg += "\n\n" + extras
    return user_msg


# Type alias retained for readability at the protocol boundary; providers
# don't strictly need it but keeping it documents the per-lens fetch shape.
FetcherFn = Callable[[PolymarketEvent], Awaitable[LensNotebook]]
