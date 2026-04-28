from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import httpx

from skimsmarkets import config as cfg
from skimsmarkets.pipeline import (
    fetch_gamma_events,
    fetch_gamma_league_slate,
    fetch_polymarket_slate,
    run_pipeline,
)
from skimsmarkets.polymarket import PolymarketClient
from skimsmarkets.reporting import print_events_table, print_run_summary


class _HttpxMinLevelFilter(logging.Filter):
    """Hide sub-threshold httpx / httpcore records from the terminal handler.

    Attached to the stream handler (not the logger itself) so the records
    keep their original INFO severity and any additional handler — e.g. a
    file log, or pytest's capture — still sees them.
    """

    def __init__(self, min_level: int) -> None:
        super().__init__()
        self.min_level = min_level

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name.startswith(("httpx", "httpcore")):
            return record.levelno >= self.min_level
        return True


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # httpx / httpcore emit one INFO line per request, which drowns out the
    # pipeline's own INFO logs during a normal run. In normal mode, hide
    # anything below WARNING from the terminal; in verbose mode, show everything.
    if not verbose:
        handler_filter = _HttpxMinLevelFilter(logging.WARNING)
        for handler in logging.getLogger().handlers:
            handler.addFilter(handler_filter)


async def _fetch_only(
    league: str | None,
    gamma_slugs: list[str],
    gamma_leagues: list[str],
    skip_us: bool,
) -> int:
    poly_sem = asyncio.Semaphore(cfg.POLYMARKET_FETCH_SEM)
    gamma_sem = asyncio.Semaphore(cfg.GAMMA_FETCH_SEM)
    if skip_us:
        events: list = []
    else:
        async with PolymarketClient() as pm:
            events = await fetch_polymarket_slate(
                pm, league, cfg.DEFAULT_HORIZON_HOURS, poly_sem
            )
    # Offshore fallback (--gamma-slug / --gamma-league). Gamma is
    # unauthenticated, so we open a standalone `httpx.AsyncClient` here
    # rather than reusing the UW one — the fetch-only path doesn't need
    # the UW context.
    if gamma_slugs or gamma_leagues:
        async with httpx.AsyncClient(timeout=20.0) as http:
            offshore: list = []
            if gamma_slugs:
                offshore += await fetch_gamma_events(
                    http, gamma_slugs, cfg.DEFAULT_HORIZON_HOURS, gamma_sem
                )
            if gamma_leagues:
                offshore += await fetch_gamma_league_slate(
                    http, gamma_leagues, cfg.DEFAULT_HORIZON_HOURS
                )
        # Dedupe by event id (a slug supplied via both flags lands once).
        seen = {ev.id for ev in events}
        for ev in offshore:
            if ev.id in seen:
                continue
            seen.add(ev.id)
            events.append(ev)
    print_events_table(events, league, horizon_hours=cfg.DEFAULT_HORIZON_HOURS)
    return 0


async def _full_run(
    league: str | None,
    dry_run: bool,
    gamma_slugs: list[str],
    gamma_leagues: list[str],
    skip_us: bool,
) -> int:
    result = await run_pipeline(
        league=league,
        dry_run=dry_run,
        gamma_slugs=gamma_slugs or None,
        gamma_leagues=gamma_leagues or None,
        skip_us=skip_us,
    )
    print_run_summary(result)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="skims",
        description=(
            "Fetch live Polymarket sports markets and run the multi-agent "
            f"confidence-ranker pipeline. Horizon fixed at "
            f"{cfg.DEFAULT_HORIZON_HOURS}h (set DEFAULT_HORIZON_HOURS in config.py to change)."
        ),
    )
    parser.add_argument(
        "--league",
        default=None,
        help=(
            "Restrict to a single Polymarket league by series-slug prefix "
            "(e.g. 'nba' matches 'nba-2025'). Default: all live sports."
        ),
    )
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Print live events + markets as a table without invoking LLMs (zero cost).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full pipeline against a single event only (~$0.30 of LLM spend).",
    )
    parser.add_argument(
        "--gamma-slug",
        action="append",
        default=[],
        metavar="SLUG",
        help=(
            "Add a specific offshore-Polymarket event by slug (gamma-api fallback). "
            "Use for matches not listed on polymarket-us — typically international "
            "soccer (e.g. 'lib-lan-lqu-2026-04-28'). Repeatable. Offshore rows are "
            "tagged [OFFSHORE] in the leaderboard and are NOT tradable on US."
        ),
    )
    parser.add_argument(
        "--gamma-league",
        action="append",
        default=[],
        metavar="PREFIX",
        help=(
            "Bulk-pull offshore-Polymarket events by slug prefix (gamma-api). "
            "Mirrors --league but on the offshore venue, where leagues are "
            "encoded as slug prefixes: 'lib' = Copa Libertadores, 'ucl' = "
            "Champions League, 'arg' = Argentina Primera, 'epl' = EPL, 'spl' = "
            "Saudi Pro League, etc. Repeatable. Independent of --league because "
            "US and offshore use different league code conventions."
        ),
    )
    parser.add_argument(
        "--skip-us",
        action="store_true",
        help=(
            "Bypass the polymarket-us fetch entirely. Useful with --gamma-slug "
            "or --gamma-league when you only want offshore events (e.g. "
            "'--skip-us --gamma-league lib' for Copa Libertadores only). "
            "Silently ignores --league when set (it's a US-only filter)."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    if args.fetch_only:
        return asyncio.run(
            _fetch_only(
                args.league, args.gamma_slug, args.gamma_league, args.skip_us
            )
        )
    return asyncio.run(
        _full_run(
            args.league,
            args.dry_run,
            args.gamma_slug,
            args.gamma_league,
            args.skip_us,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
