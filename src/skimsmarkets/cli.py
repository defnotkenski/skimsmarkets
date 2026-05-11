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
    possibly empty), so we don't need to coerce `None`. `--horizon` and
    `--max-prob` default to `None` at the argparse layer; the
    fall-through below resolves to the config constants when the user
    didn't pass an override, so passing the flag wins and omitting it
    quietly inherits config.
    """
    return SlateOptions(
        leagues=args.league,
        slugs=args.slug,
        sports=args.sport,
        horizon_hours=(
            args.horizon if args.horizon is not None else cfg.DEFAULT_HORIZON_HOURS
        ),
        max_implied_probability=(
            args.max_prob
            if args.max_prob is not None
            else cfg.MAX_IMPLIED_PROBABILITY
        ),
    )


async def _cmd_rank(args: argparse.Namespace) -> int:
    """Run the full pipeline: build the slate, then rank with specialists +
    director. Persists results to `logs/runs/<run_id>.jsonl`.
    """
    opts = _slate_opts_from_args(args)
    result = await run_pipeline(
        leagues=opts.leagues or None,
        horizon_hours=opts.horizon_hours,
        max_implied_probability=opts.max_implied_probability,
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


async def _cmd_gbt(args: argparse.Namespace) -> int:
    """Tennis GBT spike — backfill MatchStat box-score history and train
    the catboost prior. Two subcommands:

    - `backfill` (offline, ~50–80s for top-50 × 2 tours × 3 pages):
       hits MatchStat past-matches with `?include=stat,tournament,round`
       for every top-N player on each tour, dedups by `match.id`, and
       writes `data/tennis_gbt/raw_matches.parquet` plus a small
       profile lookup. Idempotent — overwrites both files.
    - `train` (offline, fast feature build then ~10–15min with sim
       compare): walks the parquet chronologically with point-in-time
       discipline, fits catboost on the walk-forward train fold,
       evaluates on the holdout fold, and writes
       `models/tennis_gbt_spike.cbm` + a metrics scorecard.
    """
    if args.gbt_command == "backfill":
        from skimsmarkets.tennis.gbt_backfill import run_backfill_cli
        matches_path, profiles_path = await run_backfill_cli(
            tours=list(args.tour) or ["atp", "wta"],
            top_n=args.top,
            pages=args.pages,
            page_size=args.page_size,
        )
        print(f"wrote {matches_path}")
        print(f"wrote {profiles_path}")
        return 0
    if args.gbt_command == "train":
        from skimsmarkets.tennis.gbt_train import run_train_cli
        metrics = run_train_cli(
            features_only=args.features_only,
            skip_sim_compare=args.skip_sim_compare,
        )
        # Print a one-line scorecard summary; full metrics persist to
        # the .metrics.json sidecar for retro inspection.
        if "holdout" in metrics:
            h = metrics["holdout"]
            print(
                f"holdout n={metrics['holdout_n']}  "
                f"brier={h['brier']:.4f}  "
                f"log_loss={h['log_loss']:.4f}  "
                f"auc={h.get('auc') or float('nan'):.4f}"
            )
            if "sim_compare" in metrics:
                sc = metrics["sim_compare"]
                print(
                    f"GBT vs sim (n_paired={sc['n_paired']}): "
                    f"GBT brier={sc['gbt_brier']:.4f}  "
                    f"sim brier={sc['sim_brier']:.4f}  "
                    f"delta={sc['gbt_brier'] - sc['sim_brier']:+.4f}"
                )
        else:
            print(f"feature-only smoke: {metrics}")
        return 0
    print(f"unknown gbt subcommand: {args.gbt_command}", file=sys.stderr)
    return 1


async def _cmd_execute(args: argparse.Namespace) -> int:
    """Place Kalshi market-buy orders against `--run-id`'s ranked predictions.

    Reads `logs/runs/<run_id>.jsonl`, applies the deterministic filter
    set (`--confidence`, `--min-defensibility`, `--no-negative-edge`,
    `--sport tennis`), matches each survivor to a Kalshi market by
    player-surname pair, and (if `--live`) places one market-buy order
    capped at `--bet-size-cents`. Defaults to `--dry-run`.

    Audit log: `logs/trades/<run_id>.jsonl`, one row per filtered
    prediction whether placed, skipped, or dry-run.
    """
    from skimsmarkets.execute.trader import ExecuteOptions, run_execute

    # Fall-through: an explicit CLI value (truthy list, non-None scalar,
    # explicit boolean) wins; otherwise consult the config constants.
    # `--no-negative-edge` produces `args.negative_edge=False`, so the
    # `is not None` check distinguishes that from the omitted case.
    confidence = (
        list(args.confidence) if args.confidence
        else (list(cfg.KALSHI_DEFAULT_CONFIDENCE_TIERS) or None)
    )
    min_defensibility = (
        args.min_defensibility if args.min_defensibility is not None
        else cfg.KALSHI_DEFAULT_MIN_DEFENSIBILITY
    )
    if args.negative_edge is None:
        no_negative_edge = cfg.KALSHI_DEFAULT_NO_NEGATIVE_EDGE
    else:
        # `--negative-edge` (True) means allow them through → don't filter.
        # `--no-negative-edge` (False) means drop them → filter on.
        no_negative_edge = not args.negative_edge
    sports = (
        list(args.sport) if args.sport
        else (list(cfg.KALSHI_DEFAULT_SPORTS) or None)
    )

    opts = ExecuteOptions(
        run_id=args.run_id,
        dry_run=not args.live,
        bet_size_cents=args.bet_size_cents,
        max_position_cents=args.max_position_cents,
        max_daily_spend_cents=args.max_daily_spend_cents,
        confidence=confidence,
        min_defensibility=min_defensibility,
        no_negative_edge=no_negative_edge,
        sports=sports,
    )
    config = cfg.Config.from_env(require_llm=False)
    summary = await run_execute(opts, config=config)
    print(
        f"execute: predictions={summary.total_predictions} "
        f"passed={summary.passed_filters} "
        f"filled={summary.filled} "
        f"partial={summary.partial} "
        f"submitted={summary.submitted} "
        f"dry_run={summary.skipped_dry_run} "
        f"skipped={summary.skipped} "
        f"total_cost_cents={summary.total_filled_cost_cents}"
    )
    if summary.skip_reasons:
        reasons = ", ".join(
            f"{k}={v}" for k, v in sorted(summary.skip_reasons.items())
        )
        print(f"skip reasons: {reasons}")
    return 0


async def _cmd_retro(args: argparse.Namespace) -> int:
    """Self-improvement layer — read past JSONL run logs, resolve outcomes
    against gamma, compute hit-rate cuts, and run a batched LLM pattern
    call comparing wins vs losses.

    Two steps: `calibrate` (cuts only) or `analyze` (default — cuts +
    post-match + LLM findings, joined into one `report.md`). Each step
    auto-refreshes gamma resolution sidecars at the start — no manual
    resolve step needed. Outputs land in `logs/retro/`. `--run-id`
    narrows to a single run log; without it every log under
    `logs/runs/` is processed (resolution sidecars are idempotent so
    reruns are cheap).

    `--sport` filters the analyze LLM call only — calibrate cuts and
    the implicit resolve step always cover everything in scope.
    Repeatable.
    """
    from skimsmarkets.retro.orchestrator import (
        run_step_analyze,
        run_step_calibrate,
    )

    sports_filter: set[str] | None = (
        set(args.sport) if args.sport else None
    )
    if args.step == "calibrate":
        await run_step_calibrate(run_id=args.run_id)
        return 0
    # default: analyze (full pass — cuts + post-match + LLM + report.md)
    md_path = await run_step_analyze(
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
_SUBCOMMANDS = ("rank", "fetch", "backtest", "retro", "gbt", "execute")


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
        "--horizon",
        type=int,
        default=None,
        metavar="HOURS",
        help=(
            f"Override the horizon window. Markets whose earliest "
            f"game_start_time sits further out than this are dropped "
            f"from the slate before LLM spend. Defaults to "
            f"{cfg.DEFAULT_HORIZON_HOURS}h from config.py."
        ),
    )
    p.add_argument(
        "--max-prob",
        type=float,
        default=None,
        metavar="PROB",
        help=(
            f"Override the favorite-blowout threshold. Events whose "
            f"favorite is priced at or above this on the YES mid are "
            f"dropped before the LLM path. Range [0, 1]. Defaults to "
            f"{cfg.MAX_IMPLIED_PROBABILITY:.2f} from config.py."
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
            "confidence-ranker pipeline. Horizon and favorite-blowout "
            f"threshold default to {cfg.DEFAULT_HORIZON_HOURS}h / "
            f"{cfg.MAX_IMPLIED_PROBABILITY:.2f} from config.py; override "
            "per-invocation with --horizon / --max-prob."
        ),
    )
    slate = _build_slate_parser()
    sub = parser.add_subparsers(
        dest="command", metavar="{rank,fetch,backtest,retro,gbt,execute}"
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

    p_execute = sub.add_parser(
        "execute",
        help=(
            "Place Kalshi market-buy orders against a ranked run "
            "(tennis only in v1). Defaults to --dry-run; --live is "
            "the explicit opt-in to send real orders."
        ),
    )
    p_execute.add_argument(
        "--run-id",
        required=True,
        metavar="RUN_ID",
        help="Run log under logs/runs/ to consume (e.g. `8f55201f`).",
    )
    p_execute.add_argument(
        "--live",
        action="store_true",
        help=(
            "Place real Kalshi orders. Default is --dry-run, which "
            "writes audit rows without hitting the order endpoint. "
            "--live requires KALSHI_API_KEY_ID and "
            "KALSHI_PRIVATE_KEY_PATH."
        ),
    )
    p_execute.add_argument(
        "--bet-size-cents",
        type=int,
        default=cfg.KALSHI_DEFAULT_BET_SIZE_CENTS,
        metavar="N",
        help=(
            "Hard spend cap per trade in cents (passed to Kalshi as "
            f"`buy_max_cost`). Default: {cfg.KALSHI_DEFAULT_BET_SIZE_CENTS} "
            "(${:.2f}).".format(cfg.KALSHI_DEFAULT_BET_SIZE_CENTS / 100)
        ),
    )
    p_execute.add_argument(
        "--max-position-cents",
        type=int,
        default=cfg.KALSHI_DEFAULT_MAX_POSITION_CENTS,
        metavar="N",
        help=(
            "Per-trade ceiling in cents — execute refuses to send any "
            "order whose `bet_size_cents` exceeds this. Belt-and-"
            f"suspenders against an accidental large bet. Default: "
            f"{cfg.KALSHI_DEFAULT_MAX_POSITION_CENTS}."
        ),
    )
    p_execute.add_argument(
        "--max-daily-spend-cents",
        type=int,
        default=cfg.KALSHI_DEFAULT_MAX_DAILY_SPEND_CENTS,
        metavar="N",
        help=(
            "Calendar-day spend cap (UTC). The trader globs every "
            "`logs/trades/*.jsonl`, sums today's fills, adds the "
            "current trade's ceiling, and refuses if the total would "
            "exceed this. Default: "
            f"{cfg.KALSHI_DEFAULT_MAX_DAILY_SPEND_CENTS}."
        ),
    )
    # Filter flags use `default=None` so we can distinguish "user
    # didn't pass it" (fall through to KALSHI_DEFAULT_* in config.py)
    # from "user passed it explicitly" (override config). Without this
    # split, action="append" with a config-driven non-empty default
    # would APPEND user flags rather than replace — confusing.
    p_execute.add_argument(
        "--confidence",
        action="append",
        choices=("low", "medium", "high"),
        default=None,
        metavar="TIER",
        help=(
            "Keep only predictions at these confidence tiers. "
            "Repeatable. Falls through to "
            "`KALSHI_DEFAULT_CONFIDENCE_TIERS` in config.py when "
            "omitted; empty config tuple = all tiers pass."
        ),
    )
    p_execute.add_argument(
        "--min-defensibility",
        type=float,
        default=None,
        metavar="SCORE",
        help=(
            "Drop predictions whose judge defensibility_score is below "
            "this cutoff (or None — missing scores fail this gate). "
            "Falls through to `KALSHI_DEFAULT_MIN_DEFENSIBILITY` in "
            "config.py when omitted."
        ),
    )
    # BooleanOptionalAction generates BOTH `--negative-edge` (allow)
    # and `--no-negative-edge` (drop) so the user can override either
    # direction of the config default. `default=None` signals "fall
    # through to config".
    p_execute.add_argument(
        "--negative-edge",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "`--no-negative-edge` drops predictions flagged "
            "`negative_edge=True` (director agrees with the market "
            "but with lower conviction) or `None` (market-implied "
            "missing). `--negative-edge` keeps them. Falls through "
            "to `KALSHI_DEFAULT_NO_NEGATIVE_EDGE` in config.py when "
            "neither is passed."
        ),
    )
    p_execute.add_argument(
        "--sport",
        action="append",
        default=None,
        metavar="SPORT",
        help=(
            "Restrict to one or more sport types. v1 only supports "
            "`tennis` — other sports raise at startup. Falls through "
            "to `KALSHI_DEFAULT_SPORTS` in config.py when omitted."
        ),
    )
    p_execute.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging.",
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
        choices=("calibrate", "analyze"),
        default="analyze",
        help=(
            "Which step to run. `analyze` (default) is the full pass: "
            "renders calibrate hit-rate cuts to the terminal, fetches "
            "post-match stats, runs the LLM pattern call per sport, and "
            "writes a combined `report.md`. `calibrate` is the "
            "lightweight cuts-only path. Both auto-refresh gamma "
            "resolutions first (idempotent) so output always reflects "
            "the latest settlements."
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
            "Filter the analyze LLM call to one or more sport types "
            "(e.g. `--sport tennis`). Repeatable. Calibrate cuts and "
            "the implicit resolve step are NOT filtered — they always "
            "cover everything in scope."
        ),
    )
    p_retro.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )

    # `skims gbt` — tennis GBT feasibility-spike commands.
    p_gbt = sub.add_parser(
        "gbt",
        help=(
            "Tennis GBT spike: backfill MatchStat history and train the "
            "catboost prior. Outputs land in `data/tennis_gbt/` (raw "
            "matches, gitignored) and `models/` (model artefact, "
            "committed to repo)."
        ),
    )
    gbt_sub = p_gbt.add_subparsers(
        dest="gbt_command", metavar="{backfill,train}"
    )

    p_backfill = gbt_sub.add_parser(
        "backfill",
        help=(
            "Hit MatchStat past-matches for top-N players × tour and "
            "write `data/tennis_gbt/raw_matches.parquet`."
        ),
    )
    p_backfill.add_argument(
        "--tour", action="append", default=[], metavar="TOUR",
        help="Tour to backfill (atp/wta). Repeatable. Default: both.",
    )
    p_backfill.add_argument(
        "--top", type=int, default=50,
        help="Top-N players per tour (default: 50).",
    )
    p_backfill.add_argument(
        "--pages", type=int, default=3,
        help="Past-matches pages per player (default: 3, ~300 matches).",
    )
    p_backfill.add_argument(
        "--page-size", type=int, default=100,
        help="Past-matches page size (vendor cap is generous; default: 100).",
    )
    p_backfill.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )

    p_train = gbt_sub.add_parser(
        "train",
        help=(
            "Walk-forward train the catboost prior on the backfilled "
            "parquet and write `models/tennis_gbt_spike.cbm` + metrics."
        ),
    )
    p_train.add_argument(
        "--features-only", action="store_true",
        help=(
            "Build the training table and exit (no fit) — smoke-test "
            "the feature pipeline without paying the catboost training "
            "cost."
        ),
    )
    p_train.add_argument(
        "--skip-sim-compare", action="store_true",
        help=(
            "Skip the iid Monte Carlo baseline comparison (which runs "
            "thousands of sims and adds ~5-10 min). Useful for fast "
            "iteration on hyperparameters."
        ),
    )
    p_train.add_argument(
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
        "gbt": _cmd_gbt,
        "execute": _cmd_execute,
    }
    return asyncio.run(dispatch[args.command](args))


if __name__ == "__main__":
    raise SystemExit(main())
