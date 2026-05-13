"""Per-field population audit across the Polymarket data path.

Runs the full data pipeline (slate → CLOB book + history → UW → tennis
stats + sim + GBT) on today's tennis slate, then reports the populate
rate for every event-level and market-level field. Stops before the
LLM stage so it doesn't need ANTHROPIC_API_KEY.

Same posture as the per-row audit that surfaced the Kalshi-era silent
holes (tennis_stats.age_years 0/24, .career_decider_record 0/24,
.surface 0/24). Run this after any data-path refactor to catch new
silent holes before they hide for weeks.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import Any

import httpx

from skimsmarkets import config as cfg
from skimsmarkets.pipeline import (
    apply_horizon_filter,
    enrich_tennis_gbt,
    enrich_tennis_stats,
    enrich_tennis_simulation,
    overlay_matchstats_tipoffs,
    resolve_uw_context,
)
from skimsmarkets.polymarket.enrichment import enrich_clob_book, enrich_price_history
from skimsmarkets.polymarket.models import PolymarketEvent, PolymarketMarket
from skimsmarkets.polymarket.slate import fetch_gamma_slate
from skimsmarkets.tennis.provider import build_tennis_provider
from skimsmarkets.unusual_whales import GammaTokenResolver, UnusualWhalesClient

logging.basicConfig(level=logging.WARNING)


def _is_populated(v: Any) -> bool:
    """Truthy enough to count as populated. False for None and empty
    collections; 0 and 0.0 count as populated (they're real values)."""
    if v is None:
        return False
    if isinstance(v, str | list | tuple | dict | set | frozenset):
        return len(v) > 0
    return True


def _populate_rate(values: Iterable[Any]) -> tuple[int, int, float]:
    values = list(values)
    n_total = len(values)
    n_populated = sum(1 for v in values if _is_populated(v))
    rate = n_populated / n_total if n_total else 0.0
    return n_populated, n_total, rate


def _report(label: str, values: Iterable[Any], threshold: float = 0.5) -> None:
    n, total, rate = _populate_rate(values)
    marker = "✗" if rate < threshold else ("•" if rate < 1.0 else "✓")
    print(f"  {marker} {label}: {n}/{total} ({rate * 100:.0f}%)")


def _event_fields_audit(events: list[PolymarketEvent]) -> None:
    print(f"\n=== PolymarketEvent fields ({len(events)} events) ===")
    for name in (
        "id", "slug", "title", "category", "series_slug",
        "active", "closed", "live", "ended", "score", "period", "elapsed",
        "main_spread_line", "main_total_line", "sport_type", "teams",
        "markets", "context_description",
        "uw_context", "tennis_stats", "tennis_simulation", "tennis_gbt",
    ):
        _report(name, [getattr(ev, name) for ev in events])


def _market_fields_audit(events: list[PolymarketEvent]) -> None:
    all_markets: list[PolymarketMarket] = [m for ev in events for m in ev.markets]
    yes_markets = [m for m in all_markets if not m.is_no_side]
    no_markets = [m for m in all_markets if m.is_no_side]
    print(
        f"\n=== PolymarketMarket fields "
        f"({len(all_markets)} markets total, {len(yes_markets)} YES, "
        f"{len(no_markets)} NO clones) ==="
    )
    field_names = (
        "slug", "id", "title", "yes_sub_title", "team_aliases",
        "sports_market_type", "is_no_side",
        # Prices + book (from gamma listing + CLOB enrichment)
        "yes_bid_dollars", "yes_ask_dollars",
        "yes_bid_depth", "yes_ask_depth",
        "yes_bid_size_top", "yes_ask_size_top",
        "yes_bid_book_dollars", "yes_ask_book_dollars",
        # Intraday + volume
        "notional_traded_dollars", "high_px_dollars", "low_px_dollars",
        "open_px_dollars", "close_px_dollars",
        "last_trade_price_dollars", "last_trade_qty", "market_state",
        "volume_dollars", "open_interest_dollars", "liquidity_dollars",
        # Team metadata
        "team_record", "team_provider_ids",
        # Gamma piggyback
        "gamma_spread", "gamma_one_day_price_change",
        "gamma_one_month_price_change", "gamma_competitive",
        "gamma_liquidity_dollars", "gamma_volume_dollars",
        "gamma_accepting_orders",
        # CLOB price-history
        "clob_price_change_30m", "clob_price_change_1h",
        "clob_price_change_4h", "clob_price_change_24h",
        "clob_price_path_sparkline", "clob_price_history",
        # Tipoff
        "game_start_time", "expected_expiration_time",
    )
    print("All markets:")
    for n in field_names:
        _report(n, [getattr(m, n) for m in all_markets])
    if no_markets:
        print("\nNO-clone subset only (verifies inversion populated correctly):")
        for n in (
            "yes_bid_size_top", "yes_ask_size_top",
            "yes_bid_book_dollars", "yes_ask_book_dollars",
            "yes_bid_depth", "yes_ask_depth",
            "clob_price_change_24h", "clob_price_path_sparkline",
        ):
            _report(n, [getattr(m, n) for m in no_markets])


def _tennis_stats_audit(events: list[PolymarketEvent]) -> None:
    tennis_events = [ev for ev in events if ev.tennis_stats is not None]
    if not tennis_events:
        print("\n=== TennisStatsContext: NO events have tennis_stats populated ===")
        return
    print(
        f"\n=== TennisStatsContext fields "
        f"({len(tennis_events)}/{len(events)} events with stats) ==="
    )
    ctxs = [ev.tennis_stats for ev in tennis_events]
    print("Top-level fields:")
    for n in (
        "provider", "fetched_at", "surface", "tournament",
        "player_a", "player_b", "head_to_head",
    ):
        _report(n, [getattr(c, n, None) for c in ctxs])
    # All TennisPlayerStats fields, audited symmetrically across A + B
    sub_fields = (
        "name", "api_player_id",
        "rank_singles", "rank_points", "best_rank_singles",
        "age_years", "plays",
        "ytd_win_loss", "surface_win_loss",
        "last_10_form", "recent_matches", "last_match_date",
        "first_serve_in_pct", "first_serve_win_pct", "second_serve_win_pct",
        "first_serve_return_win_pct", "second_serve_return_win_pct",
        "break_point_save_pct", "break_point_convert_pct",
        "record_vs_top_5", "record_vs_top_10",
        "record_at_grand_slam", "record_at_masters",
        "career_titles",
        "career_tiebreak_record", "career_decider_record",
        "career_comeback_record", "career_close_match_record",
        "break_point_save_pct_180d",
    )
    for side in ("player_a", "player_b"):
        print(f"\n{side} sub-fields:")
        players = [getattr(c, side, None) for c in ctxs]
        players = [p for p in players if p is not None]
        for n in sub_fields:
            _report(n, [getattr(p, n, None) for p in players])
    # Head-to-head
    h2hs = [c.head_to_head for c in ctxs if c.head_to_head is not None]
    if h2hs:
        print(f"\nhead_to_head sub-fields ({len(h2hs)}/{len(ctxs)} present):")
        for n in (
            "a_wins", "b_wins", "surface_h2h", "recent_meetings",
            "a_in_matchup", "b_in_matchup",
        ):
            _report(n, [getattr(h, n, None) for h in h2hs])


def _uw_context_audit(events: list[PolymarketEvent]) -> None:
    uw_events = [ev for ev in events if ev.uw_context is not None]
    if not uw_events:
        print(f"\n=== UW context: 0/{len(events)} events attached ===")
        return
    print(
        f"\n=== UW context "
        f"({len(uw_events)}/{len(events)} events with uw_context) ==="
    )
    ctxs = [ev.uw_context for ev in uw_events]
    for n in (
        "tag_scores", "liquidity", "mci", "smart_trades",
        "contrarian_trades", "insider_trades",
    ):
        _report(n, [getattr(c, n, None) for c in ctxs])


async def main() -> None:
    config = cfg.Config.from_env(require_llm=False)
    async with (
        UnusualWhalesClient(config.unusual_whales_api_key) as uw,
        httpx.AsyncClient(timeout=20.0) as http,
        build_tennis_provider(config) as tennis_provider,
    ):
        # ----- slate -----
        events = await fetch_gamma_slate(http, [], 24, sports=["tennis"])
        print(f"slate: fetched {len(events)} tennis events")
        # Cap at 10 to keep CLOB calls bounded.
        events = events[:10]

        # ----- tipoff overlay + horizon -----
        await overlay_matchstats_tipoffs(events, tennis_provider)
        events = apply_horizon_filter(events, horizon_hours=24)

        # ----- CLOB enrichment -----
        resolver = GammaTokenResolver(http)
        clob_sem = asyncio.Semaphore(cfg.CLOB_FETCH_SEM)
        uw_sem = asyncio.Semaphore(cfg.UW_FETCH_SEM)
        tennis_sem = asyncio.Semaphore(cfg.TENNIS_STATS_FETCH_SEM)
        await resolve_uw_context(uw, events, uw_sem, resolver=resolver)
        await enrich_clob_book(events, resolver, http, clob_sem)
        await enrich_price_history(events, resolver, http, clob_sem)

        # ----- tennis stats + sim + GBT -----
        errors: list = []
        await enrich_tennis_stats(tennis_provider, events, tennis_sem, errors)
        enrich_tennis_simulation(events, errors)
        enrich_tennis_gbt(events, errors)

        print(f"\npost-pipeline: {len(events)} events, {len(errors)} errors")
        if errors:
            for e in errors[:5]:
                print(f"  err {e.stage}: {e.error[:80]}")

        # ----- audit -----
        _event_fields_audit(events)
        _market_fields_audit(events)
        _tennis_stats_audit(events)
        _uw_context_audit(events)


if __name__ == "__main__":
    asyncio.run(main())
