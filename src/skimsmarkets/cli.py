from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import httpx

from skimsmarkets import config as cfg
from skimsmarkets.backtest.dataset import build_dataset
from skimsmarkets.pipeline import SlateOptions, fetch_slate, run_pipeline
from skimsmarkets.reporting import print_events_table, print_run_summary


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _slate_opts_from_args(args: argparse.Namespace) -> SlateOptions:
    """Translate the shared slate-flags namespace into a `SlateOptions`.

    `argparse` gives us bare lists for repeatable flags (always a list,
    possibly empty), so we don't need to coerce `None`. Horizon comes from
    `cfg` and is not currently exposed on the CLI.
    """
    return SlateOptions(
        leagues=args.league,
        slugs=args.slug,
        sports=args.sport,
        horizon_hours=cfg.DEFAULT_HORIZON_HOURS,
    )


async def _cmd_rank(args: argparse.Namespace) -> int:
    """Run the full pipeline: build the slate, then rank with specialists +
    director. Persists results to `logs/runs/<run_id>.jsonl`.
    """
    opts = _slate_opts_from_args(args)
    result = await run_pipeline(
        leagues=opts.leagues or None,
        dry_run=args.dry_run,
        horizon_hours=opts.horizon_hours,
        slugs=opts.slugs or None,
        sports=opts.sports or None,
        fetcher_provider=args.fetcher_provider,
    )
    print_run_summary(result)
    return 0


async def _cmd_fetch(args: argparse.Namespace) -> int:
    """Display-only: build the same slate `rank` would consume and print it
    as a table without invoking any LLM. Shares `fetch_slate` with
    `run_pipeline` so the displayed slate matches what would be ranked.

    Gamma is unauthenticated, so we open a standalone `httpx.AsyncClient`
    rather than reusing the UW one — fetch has no UW context.
    """
    opts = _slate_opts_from_args(args)
    gamma_sem = asyncio.Semaphore(cfg.GAMMA_FETCH_SEM)
    async with httpx.AsyncClient(timeout=20.0) as http:
        events = await fetch_slate(opts, http=http, gamma_sem=gamma_sem)
    print_events_table(events, opts.leagues, horizon_hours=opts.horizon_hours)
    return 0


async def _cmd_backtest(args: argparse.Namespace) -> int:
    """Build the backtest dataset → `backtest_cache/dataset.parquet`. Prints a
    head + by-league summary so the cache can be sanity-checked at a glance.
    """
    df = await build_dataset(max_events=args.max_events)
    unique = df["event_slug"].nunique() if not df.empty else 0
    print(f"rows={len(df)} unique_events={unique}")
    if not df.empty:
        print(df.head())
        print()
        print("by league:")
        print(df.groupby("league").size().nlargest(15))
    return 0


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


# Subcommand names — used both to register subparsers and to detect when the
# user invoked `skims` with bare slate flags (no subcommand) so we can default
# to `rank`. Kept as a module-level constant so the default-injection in
# `main()` and the subparser registration stay in sync.
_SUBCOMMANDS = ("rank", "fetch", "backtest")


def _build_slate_parser() -> argparse.ArgumentParser:
    """Parent parser holding flags shared by `rank` and `fetch`. `add_help=False`
    so subparsers can attach their own `-h` without conflicting.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--league",
        action="append",
        default=[],
        metavar="PREFIX",
        help=(
            "Filter the slate by league slug prefix (e.g. 'epl' matches "
            "'epl-lee-bur-2026-05-01'). Repeatable: `--league epl --league nba` "
            "unions both. Empty = all live sports."
        ),
    )
    p.add_argument(
        "--slug",
        action="append",
        default=[],
        metavar="SLUG",
        help=(
            "Show a specific event by slug, bypassing the horizon filter. "
            "Repeatable. When passed alone (no --league / --sport), the "
            "default browse is skipped and ONLY the requested slugs land "
            "in the slate. Combine with --league or --sport to union: "
            "filtered default browse + explicit slugs."
        ),
    )
    p.add_argument(
        "--sport",
        action="append",
        default=[],
        metavar="TAG",
        help=(
            "Filter the slate at the gamma API level via tag_slug "
            "(e.g. 'tennis', 'soccer', 'nba', 'mma', 'ufc', 'mlb', 'wnba', "
            "'ice-hockey'). Repeatable: each tag is queried separately and "
            "the results unioned. Different mechanic from --league, which "
            "is a client-side slug-prefix filter applied after the listing "
            "call. Combine the two to narrow further: e.g. "
            "`--sport tennis --league atp` keeps only ATP-prefixed slugs "
            "from the tennis listing. Empty = umbrella tag_slug=sports."
        ),
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    return p


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skims",
        description=(
            "Fetch live Polymarket sports markets and run the multi-agent "
            f"confidence-ranker pipeline. Horizon fixed at "
            f"{cfg.DEFAULT_HORIZON_HOURS}h (set DEFAULT_HORIZON_HOURS in config.py to change)."
        ),
    )
    slate = _build_slate_parser()
    sub = parser.add_subparsers(dest="command", metavar="{rank,fetch,backtest}")

    p_rank = sub.add_parser(
        "rank",
        parents=[slate],
        help="Build the slate and run the full ranking pipeline (default).",
    )
    p_rank.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full pipeline against a single event only (~$0.30 of LLM spend).",
    )
    p_rank.add_argument(
        "--fetcher-provider",
        choices=cfg.FETCHER_PROVIDERS,
        default=None,
        metavar="PROVIDER",
        help=(
            "Per-lens fetcher provider. 'grok' uses xai_sdk + grok-4.20-multi-agent "
            "(agent_count=4 ensemble); 'gemini' uses google-genai + gemini-3.1-pro "
            "single-pass (no x_search — Twitter lookups go through site:x.com on "
            "google_search). Overrides the FETCHER_PROVIDER env var; falls back to "
            f"'{cfg.DEFAULT_FETCHER_PROVIDER}' when neither is set."
        ),
    )

    sub.add_parser(
        "fetch",
        parents=[slate],
        help="Print the slate as a table without invoking any LLM (zero cost).",
    )

    p_backtest = sub.add_parser(
        "backtest",
        help="Build/refresh the backtest dataset cache (no LLM).",
    )
    p_backtest.add_argument("--max-events", type=int, default=800)
    p_backtest.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )

    return parser


def main() -> int:
    parser = _build_parser()

    # Default-to-`rank` so muscle memory like `skims --league nba` still
    # works after the subcommand split. We inject "rank" only when the user
    # passes a subcommand-flag at the top level — never when they ask for
    # help (`-h`/`--help` belongs to the top-level parser) or invoke a real
    # subcommand. Bare `skims` falls through to argparse, which prints
    # usage and exits.
    argv = sys.argv[1:]
    if argv and argv[0] not in _SUBCOMMANDS and argv[0] not in ("-h", "--help"):
        argv = ["rank", *argv]

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return 1

    _setup_logging(args.verbose)

    dispatch = {
        "rank": _cmd_rank,
        "fetch": _cmd_fetch,
        "backtest": _cmd_backtest,
    }
    return asyncio.run(dispatch[args.command](args))


if __name__ == "__main__":
    raise SystemExit(main())
