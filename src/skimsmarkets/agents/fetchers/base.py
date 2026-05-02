"""Provider-agnostic fetcher core — the Protocol every provider implements,
plus the user-message rendering and lens-dispatch helpers shared across
providers.

The fetcher's job is evidence capture, not judgment — it emits a
`LensNotebook` (free-form prose + citations + computed numbers) and never
commits to a probability or directional verdict. The verdict lives in the
downstream Claude reasoner (see `agents/reasoners.py`). Provider files
(`grok.py`, `gemini.py`) implement `FetcherProvider` by wiring their SDK
to the prebuilt per-lens system prompts that `LENS_PROMPT_BUILDERS` exposes.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Protocol

from skimsmarkets.agents.prompts import (
    injury_notebook_system,
    market_context_notebook_system,
    narrative_notebook_system,
    statistics_notebook_system,
)
from skimsmarkets.agents.schemas import LensName, LensNotebook
from skimsmarkets.polymarket.models import PolymarketEvent, PolymarketMarket

log = logging.getLogger(__name__)


# Lens → (tools_section, notebook_tail) → system_prompt builder.
# Providers call `build_lens_prompts` with their own per-lens tool prose
# and shared notebook tail to pre-build the four lens-specific system
# prompts at construction time. Centralised here so both providers stay
# in sync on the lens-preamble bodies (which describe the lens's *job*,
# not the provider's tools).
LENS_PROMPT_BUILDERS: dict[LensName, Callable[[str, str], str]] = {
    "statistics": statistics_notebook_system,
    "injury": injury_notebook_system,
    "narrative": narrative_notebook_system,
    "market_context": market_context_notebook_system,
}


def build_lens_prompts(
    tools_by_lens: dict[LensName, str], notebook_tail: str
) -> dict[LensName, str]:
    """Pre-build the four lens system prompts for one provider.

    `tools_by_lens` is the provider's per-lens "What each tool can give
    you here" prose; `notebook_tail` is the provider's generic tool list
    + output rules. The lens-preamble bodies (the load-bearing prose
    describing each lens's job) come from `LENS_PROMPT_BUILDERS` so both
    providers stay in lockstep on what each lens actually does.

    Iterates `LENS_PROMPT_BUILDERS` keys directly (rather than `.items()`)
    so the loop variable retains its `LensName` literal type — PyCharm's
    type narrowing widens through tuple-unpacked comprehensions.
    """
    return {
        lens: LENS_PROMPT_BUILDERS[lens](tools_by_lens[lens], notebook_tail)
        for lens in LENS_PROMPT_BUILDERS
    }


class FetcherProvider(Protocol):
    """A provider that runs the per-lens fetch stage.

    `name` and `model` are persisted to the per-run JSONL (top-level row
    metadata) so retrospective grading can group hit-rate by provider /
    model version. `fetch` runs one lens for one event and returns the
    parsed `LensNotebook`; lens-mismatch validation happens inside the
    provider via `assert_lens_match` so prompt-mixup bugs fail loud at
    fetch time.
    """

    name: str
    model: str

    async def fetch(
        self, event: PolymarketEvent, lens: LensName
    ) -> LensNotebook: ...

    async def aclose(self) -> None: ...


def assert_lens_match(parsed: LensNotebook, expected: LensName, event_id: str) -> None:
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

    Names team_a (Polymarket favorite) and team_b (the first non-team_a side) using
    the exact yes_sub_title strings so specialists echo them back verbatim. Every
    tradable side is rendered with its bid/ask, implied, volume and liquidity so the
    market-context specialist sees the whole event board without another fetch.
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
        implied = m.yes_implied_probability
        bid = f"${m.yes_bid_dollars:.3f}" if m.yes_bid_dollars is not None else "?"
        ask = f"${m.yes_ask_dollars:.3f}" if m.yes_ask_dollars is not None else "?"
        implied_str = f"{implied:.3f}" if implied is not None else "unknown"
        extras: list[str] = []
        # State first — when not OPEN it's load-bearing (don't trust the
        # price). When OPEN, omit to keep the line tight; absence implies
        # nominal tradability.
        if m.market_state and m.market_state != "MARKET_STATE_OPEN":
            # Strip the `MARKET_STATE_` prefix so the LLM sees a clean tag
            # like `SUSPENDED` / `HALTED` / `MATCH_AND_CLOSE_AUCTION`.
            extras.append(f"state={m.market_state.removeprefix('MARKET_STATE_')}")
        if m.yes_bid_dollars is not None and m.yes_ask_dollars is not None:
            spread_bps = int(round((m.yes_ask_dollars - m.yes_bid_dollars) * 10000))
            extras.append(f"spread={spread_bps}bps")
        if m.yes_bid_size_top is not None or m.yes_ask_size_top is not None:
            # `size=B/A` is top-of-book contracts (the qty available at the
            # exact best bid / best ask). One-sided books and very thin
            # two-sided books are a strong low-confidence signal even when
            # both bid and ask are quoted. This replaces the older
            # `depth=` line — price-level counts (`yes_bid_depth`) are
            # weaker signal than actual contracts at top.
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
            # `book=$B/A` is total $ resting across the entire visible
            # ladder on each side. Together with `size` it tells "how
            # much at top" and "how much across all levels." Distinct
            # from gamma's `liq` (which is a single market-level number).
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
            m.open_px_dollars is not None
            and m.yes_bid_dollars is not None
            and m.yes_ask_dollars is not None
        ):
            # Session sentiment in one number: how much has the price moved
            # since the open. Positive = market bid this side up over the
            # session; near-zero = stable consensus. Drop on NO-side clones
            # (open_px is YES-trajectory).
            mid = (m.yes_bid_dollars + m.yes_ask_dollars) / 2.0
            extras.append(f"from_open={mid - m.open_px_dollars:+.3f}")
        if (
            m.high_px_dollars is not None
            and m.low_px_dollars is not None
        ):
            # Intraday range as a vol proxy. Wide range (e.g. `range=0.10`)
            # on a market priced near 0.50 means the price has been
            # contested today; narrow range means consensus.
            extras.append(
                f"range={m.high_px_dollars - m.low_px_dollars:.3f}"
            )
        if m.last_trade_qty is not None and m.last_trade_price_dollars is not None:
            # Size on the most recent print. A 5-share dust trade vs a
            # 500-share rip carry very different information about
            # last_trade_price. Drop on NO clone (directional).
            extras.append(f"last_qty={m.last_trade_qty:.0f}")
        if m.volume_dollars is not None:
            extras.append(f"vol=${m.volume_dollars:,.0f}")
        if m.open_interest_dollars is not None:
            # Labeled "oi" (open interest) — outstanding shares × price, NOT
            # order-book depth. Polymarket's real CLOB liquidity arrives via
            # `liq=$...` below when the gamma piggyback ran.
            extras.append(f"oi=${m.open_interest_dollars:,.0f}")
        if m.gamma_liquidity_dollars is not None:
            # `liq` is gamma's `liquidityClob` — the actual dollars sitting
            # on the order book, distinct from `oi` (open interest). Both
            # are surfaced so the LLM sees "how much is held" and "how much
            # can I trade right now" as separate signals.
            extras.append(f"liq=${m.gamma_liquidity_dollars:,.0f}")
        if m.gamma_one_day_price_change is not None:
            # Signed price move over the past 24h in dollars. Positive ≈
            # smart money pushed the line up; near-zero ≈ stable consensus.
            extras.append(f"1d={m.gamma_one_day_price_change:+.3f}")
        if m.gamma_competitive is not None:
            # Polymarket's own competitiveness score (0..1, higher = more
            # contested). Worth surfacing for the LLM to factor in.
            extras.append(f"comp={m.gamma_competitive:.2f}")
        if m.team_record:
            # W/L record for this side's team (e.g. "28-6"). Saves the
            # specialist a web-search round trip for season form.
            extras.append(f"record={m.team_record}")
        # CLOB price-history extras. `path=` is a 5-point sparkline of the
        # past ~24h (e.g. `0.520→0.554→0.601→0.612→0.620`) — captures the
        # *shape* of the move, distinct from the scalar `1d=` (gamma) and
        # the recency-windowed scalars below. Order: longest → shortest so
        # the LLM can read "session move was X, last 4h was Y, last hour
        # was Z" as a recency funnel.
        if m.clob_price_path_sparkline:
            extras.append(f"path={m.clob_price_path_sparkline}")
        if m.clob_price_change_4h is not None:
            extras.append(f"4h={m.clob_price_change_4h:+.3f}")
        if m.clob_price_change_1h is not None:
            extras.append(f"1h={m.clob_price_change_1h:+.3f}")
        if m.clob_price_change_30m is not None:
            extras.append(f"30m={m.clob_price_change_30m:+.3f}")
        # [NO side, inverted] flags head-to-head markets where this side's prices
        # were derived by inverting the slug's YES book. Keep readers aware that
        # the numbers came from that flip, not a directly-quoted second market.
        side_tag = " [NO side, inverted]" if m.is_no_side else ""
        extras_str = f" {' '.join(extras)}" if extras else ""
        market_lines.append(
            f"  - slug={m.slug}{side_tag} yes='{m.yes_sub_title or '(no label)'}' "
            f"bid/ask={bid}/{ask} implied={implied_str}{extras_str}"
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
        f"team_a_name = {team_a_name}   (the Polymarket favorite going into this event)\n"
        f"team_b_name = {team_b_name}\n\n"
        f"Tradable sides on Polymarket ({len(event.markets)}):\n"
        + "\n".join(market_lines)
        + "\n\n"
        + "Produce your report now, per the schema. "
        "Use the exact team_a_name / team_b_name strings above in your output."
    )


# Type alias retained for readability at the protocol boundary; providers
# don't strictly need it but keeping it documents the per-lens fetch shape.
FetcherFn = Callable[[PolymarketEvent], Awaitable[LensNotebook]]
