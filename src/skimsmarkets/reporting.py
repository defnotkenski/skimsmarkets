"""Rich-formatted summary tables for CLI output."""

from __future__ import annotations

from datetime import UTC, datetime

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from skimsmarkets.agents.pricing import cost_usd
from skimsmarkets.agents.schemas import MarketPrediction, TokenUsage
from skimsmarkets.classify import (
    BUCKET_AVOID,
    BUCKET_COINFLIP,
    BUCKET_LEAN,
    BUCKET_LOCK,
    BUCKET_UNRATED,
    bucket_rank,
)
from skimsmarkets.pipeline import RunResult
from skimsmarkets.polymarket.models import PolymarketEvent, PolymarketMarket

# Pastel palette — used everywhere instead of bright ANSI colors.
_MINT = "#a8e6cf"  # positive / high confidence
_ROSE = "#ffaaa5"  # negative / low confidence / errors
_PEACH = "#ffd3b6"  # medium / warnings (replaces yellow)
_SKY = "#a8dadc"  # cyan-equivalent for identifiers
_LAVENDER = "#d4a5e8"  # winner / headline accents
_DIM = "#b0b0b0"  # muted
_CREAM = "#fff3b0"  # table title headings


def _rel_time(ts: datetime | None) -> str:
    if ts is None:
        return "—"
    delta = ts - datetime.now(tz=UTC)
    secs = int(delta.total_seconds())
    if secs < 0:
        return "live/past"
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _confidence_style(c: str) -> str:
    return {"high": _MINT, "medium": _PEACH, "low": _ROSE}.get(c, "")


def _risk_style(bucket: str) -> str:
    return {
        BUCKET_LOCK: f"bold {_MINT}",
        BUCKET_LEAN: _MINT,
        BUCKET_COINFLIP: _PEACH,
        BUCKET_AVOID: _ROSE,
        BUCKET_UNRATED: _DIM,
    }.get(bucket, "")


def _defensibility_stars(score: float) -> str:
    """Render a [0,1] defensibility score as a 5-slot bar. Bucket
    boundaries are 0.85 / 0.65 / 0.45 / 0.25 — chosen so a typical 0.74
    lands at four-fifths filled and 0.30 at two, matching a "Yelp rating"
    mental model. The numeric score is still surfaced verbatim on the
    JSONL row when precision matters; the bar is the glanceable form.

    Block characters (FULL BLOCK / LIGHT SHADE) are single-width across
    terminal fonts, so a 1-bar row and a 5-bar row produce identical
    cell widths — no padding hack needed. The previous ⭐ emoji form
    drifted on some terminals because of font fallback width.
    """
    if score >= 0.85:
        filled = 5
    elif score >= 0.65:
        filled = 4
    elif score >= 0.45:
        filled = 3
    elif score >= 0.25:
        filled = 2
    else:
        filled = 1
    return "█" * filled + "░" * (5 - filled)


def _compact_tokens(n: int) -> str:
    """Format a token count as 1.2M / 234K / 567 for terminal display.

    Threshold chosen so a typical slate (~6 events × 4 calls × ~10K tokens)
    renders as ~240K rather than 240,000 — easier to scan when the cost
    line sits next to a leaderboard.
    """
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _format_phase_timings(
    stage_timings: dict[str, float], total: float,
) -> str:
    """Roll up `result.stage_timings` into the four progress-bar phases
    for a compact one-line display in the run summary panel.

    The per-stage keys are the same ones used by `_time_stage` in
    `pipeline.py`. We bucket them by which progress phase they belong
    to so the user sees `total Xs — slate Ys · enrich Zs · predict
    Ws · judge Vs` matching the visual phases they just watched fill.
    Missing stages contribute 0 (e.g. an early-exit run after `select`
    has no `process_events` time).
    """
    slate = sum(
        stage_timings.get(k, 0.0) for k in (
            "fetch_slate",
            "overlay_matchstats_tipoffs",
            "apply_horizon_filter",
            "select",
        )
    )
    enrich = sum(
        stage_timings.get(k, 0.0) for k in (
            "enrich_uw",
            "enrich_clob_book",
            "enrich_clob_history",
            "enrich_tennis_stats",
            "enrich_tennis_sim",
            "enrich_tennis_gbt",
        )
    )
    predict = stage_timings.get("process_events", 0.0)
    judge = stage_timings.get("judge", 0.0)
    return (
        f"total {total:.1f}s — "
        f"slate {slate:.1f}s · enrich {enrich:.1f}s · "
        f"predict {predict:.1f}s · judge {judge:.1f}s"
    )


def _summarize_llm_spend(
    usage: list[TokenUsage],
) -> tuple[float, int, int, int, int]:
    """Walk all LLM calls in a run and return:
    (cost_usd_total, input_total, output_total, cache_read_total,
    n_unpriced_calls).

    `cost_usd` returns None for unregistered models — those calls
    contribute 0 to the cost total and are surfaced as `n_unpriced` so
    the user can tell when the displayed dollar figure is missing
    non-Anthropic spend (Grok, Gemini) that hasn't been priced yet.
    """
    cost_total = 0.0
    in_total = 0
    out_total = 0
    cache_read_total = 0
    n_unpriced = 0
    for u in usage:
        in_total += u.input_tokens or 0
        out_total += u.output_tokens or 0
        cache_read_total += u.cache_read_input_tokens or 0
        c = cost_usd(u)
        if c is None:
            n_unpriced += 1
        else:
            cost_total += c
    return cost_total, in_total, out_total, cache_read_total, n_unpriced


def _pick_favorite(event: PolymarketEvent) -> PolymarketMarket:
    """Favorite = the side with the highest implied probability.

    `fetch_polymarket_slate` guarantees every event has at least one labeled
    market with bid/ask, so this always returns a market — no None case.
    """
    return max(
        event.markets,
        key=lambda m: m.yes_implied_probability or 0.0,
    )


def print_events_table(
    events: list[PolymarketEvent],
    leagues: list[str],
    horizon_hours: int | None = None,
    *,
    console: Console | None = None,
) -> None:
    """One row per event: the favorite side. Sorted by Polymarket dollar
    volume so the busiest markets lead the list.

    The row count here is the slate the full pipeline will process — both
    paths consume the same `fetch_slate` output.
    """
    pairs = [(ev, _pick_favorite(ev)) for ev in events]
    pairs.sort(key=lambda pair: pair[1].volume_dollars or 0.0, reverse=True)

    # Accept a shared Console (set up in `cli._CONSOLE`) so the table
    # routes through the same Live-aware instance the `RichHandler`
    # uses — preserves log-vs-display ordering when invoked alongside
    # active progress bars. Falls back to a fresh `Console()` for
    # standalone callers.
    console = console or Console()
    # Breathing room between the preceding pipeline log lines and the
    # table title — without it the title sits flush against the last
    # `INFO ...` line.
    console.print()
    horizon_note = f", within {horizon_hours}h" if horizon_hours is not None else ""
    league_note = f" — leagues={','.join(leagues)}" if leagues else ""
    title = (
        "Live sports events (Polymarket)"
        + league_note
        + horizon_note
        + f" ({len(events)} events)"
    )
    table = Table(
        title=f"[{_CREAM}]{title}[/]",
        title_justify="left",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        header_style=_LAVENDER,
    )
    table.add_column("League", style=_SKY)
    table.add_column("Event")
    table.add_column("Favorite side")
    table.add_column("Bid/ask", justify="right")
    table.add_column("Implied", justify="right")
    table.add_column("Volume", justify="right")
    # Gamma's `liquidity` field — resting CLOB book depth in dollars
    # (forward-looking, populated from minute one by market makers).
    # Same number `MIN_OPEN_INTEREST_DOLLARS` filters on.
    table.add_column("Open interest", justify="right")
    table.add_column("Tips in", justify="right")

    for ev, m in pairs:
        implied = m.yes_implied_probability
        bidask = (
            f"{m.yes_bid_dollars:.2f}/{m.yes_ask_dollars:.2f}"
            if m.yes_bid_dollars is not None and m.yes_ask_dollars is not None
            else "—"
        )
        side_label = (m.yes_sub_title or "—")[:30]
        # Append the team's W/L record inline ("Cavaliers (28-6)") rather
        # than adding a column — keeps the table width manageable and
        # ties the record to the side it belongs to.
        if m.team_record:
            side_label += f" ({m.team_record})"
        if m.is_no_side:
            side_label += " [NO]"
        # Read the canonical `open_interest_dollars` field. The legacy
        # `liquidity_dollars` alias still exists on the model for any
        # external consumer that hasn't migrated, but in-tree readers
        # should prefer the unambiguous name.
        table.add_row(
            ev.series_slug or "—",
            (ev.title or ev.id)[:40],
            side_label,
            bidask,
            f"{implied:.2f}" if implied is not None else "—",
            f"${m.volume_dollars:,.0f}" if m.volume_dollars is not None else "—",
            (
                f"${m.open_interest_dollars:,.0f}"
                if m.open_interest_dollars is not None
                else "—"
            ),
            _rel_time(m.game_start_time),
        )

    if not events:
        console.print(
            f"[{_PEACH}]No live markets found"
            + (f" for leagues={','.join(leagues)}" if leagues else "")
            + ".[/]"
        )
        return
    console.print(table)


def print_run_summary(
    result: RunResult, *, console: Console | None = None,
) -> None:
    # Accept a shared Console (set up in `cli._CONSOLE`) so the rule,
    # leaderboard, and summary panel print through the same Live-aware
    # instance the `ProgressReporter` used during the run and the
    # `RichHandler` uses for logging. Falling back to a fresh
    # `Console()` keeps non-CLI callers (tests, ad-hoc imports)
    # working without setup.
    console = console or Console()
    # Breathing room between the preceding pipeline log lines and the
    # rule that opens the summary.
    console.print()
    console.rule(f"[bold {_LAVENDER}]Run {result.run_id}[/]", style=_LAVENDER)
    # NOTE: the previous plain-text counts line ("Fetched events: 24 …")
    # was folded into the bottom `Run summary` panel together with the
    # LLM cost and pipeline timing. Leaderboard now sits directly under
    # the rule as the visual headline.

    if result.predictions:
        # Leaderboard primary sort: risk bucket (Lock → Lean → Coin-flip →
        # Avoid → Unrated). Within a bucket, judge `defensibility_score`
        # descending, then predicted probability descending. Events with no
        # risk classification (e.g. a directly-constructed RunResult in a
        # test) fall back to the Unrated bucket; events with no judge score
        # use the -1.0 defensibility sentinel so they lose every tiebreak.
        # When the whole judge call failed, every event is Unrated and the
        # sort collapses to predicted-probability order — the legacy
        # fallback behavior.
        def _sort_key(p: MarketPrediction) -> tuple[int, float, float]:
            rc = result.risk_classifications.get(p.event_id)
            bucket = rc[0] if rc is not None else BUCKET_UNRATED
            da = result.defensibility_assessments.get(p.event_id)
            score = da.defensibility_score if da is not None else -1.0
            # Negate the rank so the best bucket (Lock, rank 0) sorts first
            # under reverse=True.
            return (-bucket_rank(bucket), score, p.predicted_yes_probability)

        ranked = sorted(result.predictions, key=_sort_key, reverse=True)
        any_classified = bool(result.risk_classifications)
        any_judged = bool(result.defensibility_assessments)
        title_text = (
            "Risk-graded slate (Lock → Avoid)"
            if any_classified
            else "Defensibility leaderboard (most defensible case first)"
            if any_judged
            else "Confidence leaderboard (highest predicted probability first)"
        )

        leaderboard = Table(
            title=f"[{_CREAM}]{title_text}[/]",
            title_justify="left",
            box=box.SIMPLE_HEAVY,
            show_lines=False,
            header_style=_LAVENDER,
        )
        leaderboard.add_column("#", justify="right", style=_DIM)
        leaderboard.add_column("Event", style=_SKY, overflow="fold", min_width=24)
        leaderboard.add_column(
            "Winner", style=f"bold {_LAVENDER}", overflow="fold", min_width=14
        )
        leaderboard.add_column("Risk", justify="center")
        leaderboard.add_column("Case", justify="center")
        leaderboard.add_column("Pred", justify="right")
        leaderboard.add_column("Poly impl", justify="right")
        leaderboard.add_column("Conf", justify="center")

        for rank, p in enumerate(ranked, start=1):
            event_display = p.event_title or p.event_id
            poly_impl_str = (
                f"{p.polymarket_implied_probability:.3f}"
                if p.polymarket_implied_probability is not None
                else "—"
            )
            da = result.defensibility_assessments.get(p.event_id)
            # Stars only — flags (`lens_disagreement`, `concentrated_weights`,
            # etc.) added clutter without changing the read at a glance. They
            # still persist on the JSONL row for retrospective grading.
            case_cell = (
                _defensibility_stars(da.defensibility_score) if da is not None else "—"
            )
            rc = result.risk_classifications.get(p.event_id)
            risk_bucket = rc[0] if rc is not None else BUCKET_UNRATED
            risk_cell = f"[{_risk_style(risk_bucket)}]{risk_bucket}[/]"
            leaderboard.add_row(
                str(rank),
                event_display,
                p.predicted_winner,
                risk_cell,
                case_cell,
                f"{p.predicted_yes_probability:.3f}",
                poly_impl_str,
                f"[{_confidence_style(p.confidence)}]{p.confidence}[/]",
            )
        console.print(leaderboard)

    # Detail tables (headline / defensibility rationale / UW flow notes /
    # disagreement flags) used to print here. They moved to the JSONL
    # row's top level (see `_persist_run`) so retrospective grading and
    # ad-hoc inspection can `jq '.headline' / '.uw_flow_note'` etc.
    # without re-running. Errors stay terminal-only because they aren't
    # persisted to JSONL — losing them would silence dropped events.

    if result.errors:
        err_table = Table(
            title=f"[{_CREAM}]Errors[/]",
            title_justify="left",
            box=box.SIMPLE,
            show_lines=False,
            header_style=_LAVENDER,
        )
        err_table.add_column("Event", style=_SKY)
        err_table.add_column("Stage", style=_ROSE)
        err_table.add_column("Error")
        for e in result.errors:
            err_table.add_row(e.event_id, e.stage, e.error)
        console.print(err_table)

    if not result.predictions and not result.errors:
        console.print(f"[{_PEACH}]No predictions generated.[/]")

    # ── Run summary panel ─────────────────────────────────────────
    # One Rich panel below the leaderboard / errors table consolidating
    # what used to be three scattered plain-text blocks: the counts
    # line above the leaderboard, the LLM cost line below it, and the
    # per-stage timing breakdown that previously only landed in
    # stderr logs. Order inside the panel mirrors how the user reads
    # the run: what came in (counts), what it cost (LLM), how long it
    # took (timing).
    summary_lines: list[str] = [
        f"  [{_SKY}]fetched events:[/] {result.fetched_events}    "
        f"[{_SKY}]considered:[/] {result.considered_events}    "
        f"[{_MINT}]predictions:[/] {len(result.predictions)}    "
        f"[{_ROSE}]errors:[/] {len(result.errors)}",
    ]

    # LLM spend summary. Walks both per-event reasoner/director calls
    # and the slate-level judge call. Anthropic-only today: Grok/Gemini
    # fetcher calls land in `n_unpriced` until their rates are
    # registered in `agents/pricing.py`. Matches the `cost_usd_total`
    # in the JSONL meta row so terminal and persisted log show the
    # same number.
    all_calls = [
        u for ev_list in result.token_usage.values() for u in ev_list
    ] + result.slate_token_usage
    if all_calls:
        cost_total, in_total, out_total, cache_read_total, n_unpriced = (
            _summarize_llm_spend(all_calls)
        )
        unpriced_note = (
            f"; {n_unpriced} unpriced "
            f"{'call' if n_unpriced == 1 else 'calls'}"
            if n_unpriced
            else ""
        )
        summary_lines.append("")
        summary_lines.append(
            f"  [{_CREAM}]LLM cost:[/] "
            f"[bold {_MINT}]${cost_total:.4f}[/]  "
            f"[{_DIM}]({_compact_tokens(in_total)} input / "
            f"{_compact_tokens(out_total)} output / "
            f"{_compact_tokens(cache_read_total)} cache-read"
            f"{unpriced_note})[/]"
        )

    if result.stage_timings:
        summary_lines.append("")
        summary_lines.append(
            f"  [{_SKY}]pipeline timing:[/] [{_DIM}]"
            f"{_format_phase_timings(result.stage_timings, result.total_seconds)}"
            f"[/]"
        )

    console.print(
        Panel(
            "\n".join(summary_lines),
            title=f"[{_CREAM}]Run summary[/]",
            title_align="left",
            border_style=_LAVENDER,
            padding=(0, 0),
        )
    )
