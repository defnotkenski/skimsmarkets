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


def _pick_favorite(event: PolymarketEvent) -> PolymarketMarket | None:
    """Favorite = the side with the highest implied probability among tradable
    sides with a label. Used for the --fetch-only table's one-row-per-event view.
    """
    scored = [
        (m.yes_implied_probability or -1.0, m) for m in event.markets if m.yes_sub_title
    ]
    if not scored:
        return None
    scored.sort(key=lambda s: s[0], reverse=True)
    top_prob, top_market = scored[0]
    return top_market if top_prob >= 0 else None


def print_events_table(
    events: list[PolymarketEvent],
    league: str | None,
    horizon_hours: int | None = None,
) -> None:
    """One row per event: the favorite side. Sorted by Polymarket dollar
    volume so the busiest markets lead the list.
    """
    pairs: list[tuple[PolymarketEvent, PolymarketMarket]] = []
    for ev in events:
        favorite = _pick_favorite(ev)
        if favorite is None:
            continue
        pairs.append((ev, favorite))
    pairs.sort(key=lambda pair: pair[1].volume_dollars or 0.0, reverse=True)

    console = Console()
    horizon_note = f", within {horizon_hours}h" if horizon_hours is not None else ""
    title = (
        "Live sports events (Polymarket)"
        + (f" — league={league}" if league else "")
        + horizon_note
        + f" ({len(pairs)} shown / {len(events)} total)"
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
        if m.is_no_side:
            side_label += " [NO]"
        table.add_row(
            ev.series_slug or "—",
            (ev.title or ev.id)[:40],
            side_label,
            bidask,
            f"{implied:.2f}" if implied is not None else "—",
            f"${m.volume_dollars:,.0f}" if m.volume_dollars is not None else "—",
            f"${m.liquidity_dollars:,.0f}" if m.liquidity_dollars is not None else "—",
            _rel_time(m.game_start_time),
        )

    if not pairs:
        console.print(
            f"[{_PEACH}]No live markets found"
            + (f" for league={league}" if league else "")
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
        # Leaderboard ranks events by the director's confidence in the winner —
        # highest predicted probability first. Kelly is reference sizing only;
        # it doesn't drive the rank order.
        ranked = sorted(
            result.predictions,
            key=lambda sm: sm.prediction.predicted_yes_probability,
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
        leaderboard.add_column("Entry ask", justify="right")
        leaderboard.add_column("Capped ½K", justify="right")
        leaderboard.add_column("Conf", justify="center")

        for rank, s in enumerate(ranked, start=1):
            p, z = s.prediction, s.sizing
            event_display = p.event_title or p.event_id
            poly_impl_str = (
                f"{p.polymarket_implied_probability:.3f}"
                if p.polymarket_implied_probability is not None
                else "—"
            )
            entry_str = (
                f"${z.entry_price_dollars:.2f}"
                if z.entry_price_dollars is not None
                else "—"
            )
            leaderboard.add_row(
                str(rank),
                event_display,
                p.predicted_winner,
                f"{p.predicted_yes_probability:.3f}",
                poly_impl_str,
                entry_str,
                f"[bold {_MINT}]{z.capped_half_kelly_fraction:.1%}[/]",
                f"[{_confidence_style(p.confidence)}]{p.confidence}[/]",
            )
        console.print(leaderboard)

        reasoning_table = Table(
            title=f"[{_CREAM}]Director reasoning (same order)[/]",
            title_justify="left",
            box=box.SIMPLE_HEAVY,
            show_lines=True,
            header_style=_LAVENDER,
        )
        reasoning_table.add_column("Event", style=_SKY, overflow="fold", min_width=20)
        reasoning_table.add_column(
            "Winner", style=f"bold {_LAVENDER}", overflow="fold", min_width=14
        )
        reasoning_table.add_column("Reasoning", overflow="fold")
        for s in ranked:
            p = s.prediction
            reasoning_table.add_row(
                p.event_title or p.event_id,
                p.predicted_winner,
                p.reasoning,
            )
        console.print(reasoning_table)

        # UW flow notes get their own table (rather than a column on the
        # reasoning table) — the 2-4 sentence notes are too wide to share a
        # row with the reasoning text without squishing both. Only events
        # with a non-null `uw_flow_note` appear here; everything else is
        # silently omitted, including events with no UW coverage at all.
        uw_rows = [s for s in ranked if s.prediction.uw_flow_note]
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
            for s in uw_rows:
                p = s.prediction
                uw_table.add_row(
                    p.event_title or p.event_id,
                    p.predicted_winner,
                    p.uw_flow_note or "",
                )
            console.print(uw_table)

        # Flags get their own table so long notes don't break the leaderboard rows.
        flag_rows: list[tuple[str, str, str]] = []
        for s in ranked:
            label = s.prediction.event_title or s.prediction.event_id
            for note in s.sizing.notes:
                flag_rows.append((label, "sizing", note))
            for disagreement in s.prediction.disagreements_flagged:
                flag_rows.append((label, "director", disagreement))
        if flag_rows:
            flag_table = Table(
                title=f"[{_CREAM}]Flags[/]",
                title_justify="left",
                box=box.SIMPLE,
                show_lines=False,
                header_style=_LAVENDER,
            )
            flag_table.add_column("Event", style=_SKY)
            flag_table.add_column("Source", style=_PEACH)
            flag_table.add_column("Note")
            for label, source, note in flag_rows:
                flag_table.add_row(label, source, note)
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
