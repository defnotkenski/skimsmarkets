"""Rich-formatted summary tables for CLI output."""

from __future__ import annotations

from datetime import UTC, datetime

from rich import box
from rich.console import Console
from rich.table import Table

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
    league: str | None,
    horizon_hours: int | None = None,
) -> None:
    """One row per event: the favorite side. Sorted by Polymarket dollar
    volume so the busiest markets lead the list.

    The row count here is the slate the full pipeline will process — both
    paths consume the same `fetch_polymarket_slate` output.
    """
    pairs = [(ev, _pick_favorite(ev)) for ev in events]
    pairs.sort(key=lambda pair: pair[1].volume_dollars or 0.0, reverse=True)

    console = Console()
    horizon_note = f", within {horizon_hours}h" if horizon_hours is not None else ""
    title = (
        "Live sports events (Polymarket)"
        + (f" — league={league}" if league else "")
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
    # polymarket-us doesn't publish order-book liquidity as a dollar figure;
    # what we have is dollar open interest (outstanding shares × price), which
    # is a market-size proxy rather than "how much can I trade right now."
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
        # Mark offshore rows so the user doesn't mistake them for US-tradable
        # markets — the prices come from a different liquidity pool.
        if ev.venue == "offshore":
            side_label += f" [{_PEACH}][OFFSHORE][/]"
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
            + (f" for league={league}" if league else "")
            + ".[/]"
        )
        return
    console.print(table)
    if any(ev.venue == "offshore" for ev in events):
        console.print(
            f"[{_PEACH}]Note: [OFFSHORE] rows come from gamma-api (offshore Polymarket) "
            "and are NOT tradable on polymarket-us.[/]"
        )


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
        # Leaderboard ranks events by the director's confidence in the winner —
        # highest predicted probability first.
        ranked = sorted(
            result.predictions,
            key=lambda p: p.predicted_yes_probability,
            reverse=True,
        )

        leaderboard = Table(
            title=f"[{_CREAM}]Confidence leaderboard (highest predicted probability first)[/]",
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
        leaderboard.add_column("Pred", justify="right")
        leaderboard.add_column("Poly impl", justify="right")
        leaderboard.add_column("Conf", justify="center")

        for rank, p in enumerate(ranked, start=1):
            event_display = p.event_title or p.event_id
            if p.venue == "offshore":
                event_display = f"{event_display} [{_PEACH}][OFFSHORE][/]"
            poly_impl_str = (
                f"{p.polymarket_implied_probability:.3f}"
                if p.polymarket_implied_probability is not None
                else "—"
            )
            leaderboard.add_row(
                str(rank),
                event_display,
                p.predicted_winner,
                f"{p.predicted_yes_probability:.3f}",
                poly_impl_str,
                f"[{_confidence_style(p.confidence)}]{p.confidence}[/]",
            )
        console.print(leaderboard)

        # At-a-glance: one sentence per event so the table fits on screen during a
        # time crunch. The director's full multi-sentence `reasoning` is still on
        # the model — print it from the detail/debug path when needed.
        headline_table = Table(
            title=f"[{_CREAM}]Director headline (same order)[/]",
            title_justify="left",
            box=box.SIMPLE_HEAVY,
            show_lines=False,
            header_style=_LAVENDER,
        )
        headline_table.add_column("Event", style=_SKY, overflow="fold", min_width=20)
        headline_table.add_column(
            "Winner", style=f"bold {_LAVENDER}", overflow="fold", min_width=14
        )
        headline_table.add_column("Headline", overflow="fold")
        for p in ranked:
            headline_table.add_row(
                p.event_title or p.event_id,
                p.predicted_winner,
                p.headline,
            )
        console.print(headline_table)

        # UW flow notes get their own table (rather than a column on the
        # reasoning table) — the 2-4 sentence notes are too wide to share a
        # row with the reasoning text without squishing both. Only events
        # with a non-null `uw_flow_note` appear here; everything else is
        # silently omitted, including events with no UW coverage at all.
        uw_rows = [p for p in ranked if p.uw_flow_note]
        if uw_rows:
            uw_table = Table(
                title=f"[{_CREAM}]Unusual Whales flow (where UW had coverage)[/]",
                title_justify="left",
                box=box.SIMPLE_HEAVY,
                show_lines=True,
                header_style=_LAVENDER,
            )
            uw_table.add_column("Event", style=_SKY, overflow="fold", min_width=20)
            uw_table.add_column(
                "Winner", style=f"bold {_LAVENDER}", overflow="fold", min_width=14
            )
            uw_table.add_column("UW flow", overflow="fold")
            for p in uw_rows:
                uw_table.add_row(
                    p.event_title or p.event_id,
                    p.predicted_winner,
                    p.uw_flow_note or "",
                )
            console.print(uw_table)

        # Flags get their own table so long notes don't break the leaderboard rows.
        flag_rows: list[tuple[str, str]] = []
        for p in ranked:
            label = p.event_title or p.event_id
            for disagreement in p.disagreements_flagged:
                flag_rows.append((label, disagreement))
        if flag_rows:
            flag_table = Table(
                title=f"[{_CREAM}]Flags[/]",
                title_justify="left",
                box=box.SIMPLE,
                show_lines=False,
                header_style=_LAVENDER,
            )
            flag_table.add_column("Event", style=_SKY)
            flag_table.add_column("Note")
            for label, note in flag_rows:
                flag_table.add_row(label, note)
            console.print(flag_table)

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
