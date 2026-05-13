"""Rich-formatted summary tables for CLI output."""

from __future__ import annotations

from datetime import UTC, datetime

from rich import box
from rich.console import Console
from rich.table import Table

from skimsmarkets.agents.schemas import MarketPrediction
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
) -> None:
    """One row per event: the favorite side. Sorted by Polymarket dollar
    volume so the busiest markets lead the list.

    The row count here is the slate the full pipeline will process — both
    paths consume the same `fetch_slate` output.
    """
    pairs = [(ev, _pick_favorite(ev)) for ev in events]
    pairs.sort(key=lambda pair: pair[1].volume_dollars or 0.0, reverse=True)

    console = Console()
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


def print_run_summary(result: RunResult) -> None:
    console = Console()
    console.rule(f"[bold {_LAVENDER}]Run {result.run_id}[/]", style=_LAVENDER)
    console.print(
        f"Fetched events: [{_SKY}]{result.fetched_events}[/]  "
        f"Considered events: [{_SKY}]{result.considered_events}[/]  "
        f"Predictions: [{_MINT}]{len(result.predictions)}[/]  "
        f"Errors: [{_ROSE}]{len(result.errors)}[/]"
    )

    if result.predictions:
        # Leaderboard primary sort: judge `defensibility_score` descending
        # (higher = stronger case). Tiebreak: predicted probability
        # descending. Un-scored events (judge failed or didn't cover them)
        # sort to the bottom via the -1.0 sentinel — outside the [0,1]
        # valid range so they always lose to real scores. When the entire
        # judge call failed, every event hits the sentinel and the tuple
        # sort collapses to predicted-probability order, which is the
        # legacy behavior we're falling back to.
        def _sort_key(p: MarketPrediction) -> tuple[float, float]:
            da = result.defensibility_assessments.get(p.event_id)
            score = da.defensibility_score if da is not None else -1.0
            return (score, p.predicted_yes_probability)

        ranked = sorted(result.predictions, key=_sort_key, reverse=True)
        any_judged = bool(result.defensibility_assessments)
        title_text = (
            "Defensibility leaderboard (most defensible case first)"
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
            leaderboard.add_row(
                str(rank),
                event_display,
                p.predicted_winner,
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
