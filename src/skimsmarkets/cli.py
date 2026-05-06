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
from skimsmarkets.selection import select_top_events
from skimsmarkets.tennis.provider import build_tennis_provider


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


# Third-party SDK loggers that emit one INFO line per request (or per
# tool-loop entry) and drown out our own pipeline INFO logs during a
# normal run. Keep the list narrow — anything added here is silenced
# from the terminal in non-verbose mode.
_NOISY_SDK_LOGGER_PREFIXES = ("httpx", "httpcore", "google_genai")


class _NoisySDKMinLevelFilter(logging.Filter):
    """Hide sub-threshold records from known-noisy SDK loggers on the
    terminal handler.

    Attached to the stream handler (not the logger itself) so the records
    keep their original INFO severity and any additional handler — e.g. a
    file log, or pytest's capture — still sees them.
    """

    def __init__(self, min_level: int) -> None:
        super().__init__()
        self.min_level = min_level

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name.startswith(_NOISY_SDK_LOGGER_PREFIXES):
            return record.levelno >= self.min_level
        return True


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # In normal mode, hide INFO chatter from third-party SDKs (httpx
    # request lines, google_genai's per-call "AFC enabled" notice) so
    # our own pipeline INFO logs stay readable. Verbose (`-v`) shows
    # everything for debugging.
    if not verbose:
        handler_filter = _NoisySDKMinLevelFilter(logging.WARNING)
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
        horizon_hours=opts.horizon_hours,
        slugs=opts.slugs or None,
        sports=opts.sports or None,
        tennis_stats_disabled=args.no_tennis_stats,
    )
    print_run_summary(result)
    return 0


async def _cmd_fetch(args: argparse.Namespace) -> int:
    """Display-only: build the same slate `rank` would consume and print it
    as a table without invoking any LLM. Shares `fetch_slate` AND
    `select_top_events` with `run_pipeline` so the displayed slate
    matches what would be ranked — including the fundamental-imbalance
    cap that selects the top `MAX_SLATE_EVENTS` by player-rank ratio
    and team-record delta. Without re-applying selection here, fetch
    would print the full filtered slate (often 100+ events) while
    `rank` would silently consume only the top-N — a confusing drift
    between what the user sees and what gets ranked.

    Gamma is unauthenticated, so we open a standalone `httpx.AsyncClient`
    rather than reusing the UW one — fetch has no UW context.
    """
    opts = _slate_opts_from_args(args)
    gamma_sem = asyncio.Semaphore(cfg.GAMMA_FETCH_SEM)
    # Use the env-driven config — `fetch` doesn't take provider flags
    # (those are rank-specific). Tennis provider defaults to whatever
    # `TENNIS_STATS_API_KEY` resolves to: real adapter when set, stub
    # when not. Stub means selection scoring sees `lookup_player_rank`
    # / `lookup_player_form` returning None for every tennis event,
    # which falls through to team_record + tipoff cleanly.
    config = cfg.Config.from_env()
    async with (
        httpx.AsyncClient(timeout=20.0) as http,
        build_tennis_provider(config) as tennis_provider,
    ):
        events = await fetch_slate(opts, http=http, gamma_sem=gamma_sem)
        events = await select_top_events(
            events,
            max_events=cfg.MAX_SLATE_EVENTS,
            tennis_provider=tennis_provider,
        )
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


async def _cmd_retro(args: argparse.Namespace) -> int:
    """Self-improvement layer — read past JSONL run logs, resolve outcomes
    against gamma, compute hit-rate cuts, and (Step 3) run a batched
    LLM pattern call comparing wins vs losses.

    Three steps; all three run by default. Outputs land in `logs/retro/`.
    `--run-id` narrows to a single run log; without it every log under
    `logs/runs/` is processed (resolution sidecars are idempotent so
    reruns are cheap).

    `--sport` filters the Step 3 LLM call only — Steps 1 & 2 always
    cover everything in scope. Repeatable.
    """
    from skimsmarkets.retro.orchestrator import (
        run_step_all,
        run_step_analyze,
        run_step_calibrate,
        run_step_resolve,
    )

    sports_filter: set[str] | None = (
        set(args.sport) if args.sport else None
    )
    if args.step == "resolve":
        paths = await run_step_resolve(args.run_id)
        print(f"wrote {len(paths)} resolution sidecar(s)")
        return 0
    if args.step == "calibrate":
        run_step_calibrate(run_id=args.run_id)
        return 0
    if args.step == "analyze":
        findings, path = await run_step_analyze(
            sports_filter=sports_filter, run_id=args.run_id,
        )
        print(f"wrote findings for {len(findings)} sport(s) to {path}")
        return 0
    # default: all
    md_path = await run_step_all(
        sports_filter=sports_filter, run_id=args.run_id,
    )
    print(f"retro report: {md_path}")
    return 0


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


# Subcommand names — used both to register subparsers and to detect when the
# user invoked `skims` with bare slate flags (no subcommand) so we can default
# to `rank`. Kept as a module-level constant so the default-injection in
# `main()` and the subparser registration stay in sync.
_SUBCOMMANDS = ("rank", "fetch", "backtest", "retro")


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
    sub = parser.add_subparsers(
        dest="command", metavar="{rank,fetch,backtest,retro}"
    )

    p_rank = sub.add_parser(
        "rank",
        parents=[slate],
        help="Build the slate and run the full ranking pipeline (default).",
    )
    p_rank.add_argument(
        "--no-tennis-stats",
        action="store_true",
        help=(
            "Force the stub tennis-stats provider even when TENNIS_STATS_API_KEY "
            "is set. Useful for token-cost A/B comparisons or when the vendor is "
            "down. No-op when no key is configured (already running the stub)."
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

    p_retro = sub.add_parser(
        "retro",
        help=(
            "Retro / self-improvement: resolve past predictions against "
            "gamma, compute hit-rate cuts, and run an LLM pattern call."
        ),
    )
    p_retro.add_argument(
        "--step",
        choices=("resolve", "calibrate", "analyze", "all"),
        default="all",
        help=(
            "Which step to run. `all` (default) runs Step 1 → 2 → 3 in "
            "sequence and writes a combined report.md. `resolve` only "
            "writes the gamma-resolution sidecars (cheap, no LLM)."
        ),
    )
    p_retro.add_argument(
        "--run-id",
        default=None,
        metavar="RUN_ID",
        help=(
            "Operate on a single run log (e.g. `8f55201f`). When omitted, "
            "every log under logs/runs/ is processed."
        ),
    )
    p_retro.add_argument(
        "--sport",
        action="append",
        default=[],
        metavar="SPORT",
        help=(
            "Filter the Step 3 LLM call to one or more sport types "
            "(e.g. `--sport tennis`). Repeatable. Steps 1 & 2 are NOT "
            "filtered — they always cover everything resolved."
        ),
    )
    p_retro.add_argument(
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
        "retro": _cmd_retro,
    }
    return asyncio.run(dispatch[args.command](args))


if __name__ == "__main__":
    raise SystemExit(main())
