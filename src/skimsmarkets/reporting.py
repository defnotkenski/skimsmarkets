"""Rich-formatted summary tables for CLI output."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from rich import box
from rich.console import Console
from rich.table import Table

from skimsmarkets.kalshi.models import KalshiEvent, KalshiMarket
from skimsmarkets.pipeline import RunResult

# Pastel palette — used everywhere instead of bright ANSI colors.
_MINT = "#a8e6cf"       # positive / buy / high confidence
_ROSE = "#ffaaa5"       # negative / low confidence / errors
_PEACH = "#ffd3b6"      # medium / warnings (replaces yellow)
_SKY = "#a8dadc"        # cyan-equivalent for identifiers
_LAVENDER = "#d4a5e8"   # winner / headline accents
_DIM = "#b0b0b0"        # pass / muted
_CREAM = "#fff3b0"      # table title headings


def _rel_time(ts: datetime | None) -> str:
    if ts is None:
        return "—"
    delta = ts - datetime.now(tz=UTC)
    secs = int(delta.total_seconds())
    if secs < 0:
        return "closed"
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _rec_style(rec: str) -> str:
    return {
        "buy_yes": f"bold {_MINT}",
        "pass": _DIM,
    }.get(rec, "")


def _confidence_style(c: str) -> str:
    return {"high": _MINT, "medium": _PEACH, "low": _ROSE}.get(c, "")


def _within_horizon(market: KalshiMarket, hours: int) -> bool:
    if market.expected_expiration_time is None:
        return True  # keep unknowns visible rather than silently dropping
    return market.expected_expiration_time <= datetime.now(tz=UTC) + timedelta(hours=hours)


def print_events_table(
    events: list[KalshiEvent],
    series_filter: str | None,
    horizon_hours: int | None = None,
) -> None:
    # One row per event: keep the favorite side (implied probability >= 0.5).
    # Markets with unknown implied probability are kept so they stay visible.
    pairs = [
        (e, m)
        for e in events
        for m in e.markets
        if (m.yes_implied_probability is None or m.yes_implied_probability >= 0.5)
        and (horizon_hours is None or _within_horizon(m, horizon_hours))
    ]
    pairs.sort(key=lambda pair: pair[1].volume_24h_fp or 0.0, reverse=True)

    console = Console()
    horizon_note = f", within {horizon_hours}h" if horizon_hours is not None else ""
    title = (
        "Live Kalshi sports events"
        + (f" — series={series_filter}" if series_filter else "")
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
    table.add_column("Series", style=_SKY)
    table.add_column("Event")
    table.add_column("Market (yes side)")
    table.add_column("Yes bid/ask", justify="right")
    table.add_column("Implied", justify="right")
    table.add_column("24h vol", justify="right")
    table.add_column("Settles in", justify="right")

    for e, m in pairs:
        implied = m.yes_implied_probability
        bidask = (
            f"{m.yes_bid_dollars:.2f}/{m.yes_ask_dollars:.2f}"
            if m.yes_bid_dollars is not None and m.yes_ask_dollars is not None
            else "—"
        )
        table.add_row(
            e.series_ticker or "—",
            (e.title or e.event_ticker)[:40],
            (m.yes_sub_title or m.title or "—")[:30],
            bidask,
            f"{implied:.2f}" if implied is not None else "—",
            f"{m.volume_24h_fp:,.0f}" if m.volume_24h_fp is not None else "—",
            _rel_time(m.expected_expiration_time),
        )
    rows = len(pairs)

    if rows == 0:
        console.print(
            f"[{_PEACH}]No live markets found"
            + (f" for {series_filter}" if series_filter else "")
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
        ranked = sorted(
            result.predictions,
            key=lambda s: s.sizing.capped_half_kelly_fraction,
            reverse=True,
        )

        pred_table = Table(
            title=f"[{_CREAM}]Predictions (sorted by capped-Kelly stake)[/]",
            title_justify="left",
            box=box.SIMPLE_HEAVY,
            show_lines=False,
            header_style=_LAVENDER,
        )
        pred_table.add_column("Event", style=_SKY, overflow="fold", min_width=24)
        pred_table.add_column("Winner", style=f"bold {_LAVENDER}", overflow="fold", min_width=14)
        pred_table.add_column("Rec", justify="center")
        pred_table.add_column("Pred", justify="right")
        pred_table.add_column("Kalshi", justify="right")
        pred_table.add_column("Edge (bps)", justify="right")
        pred_table.add_column("Conf", justify="center")

        for s in ranked:
            p = s.prediction
            edge_color = (
                _MINT if p.edge_bps > 0 else (_ROSE if p.edge_bps < 0 else "")
            )
            event_display = p.event_title or p.event_ticker
            pred_table.add_row(
                event_display,
                p.predicted_winner,
                f"[{_rec_style(p.recommendation)}]{p.recommendation}[/]",
                f"{p.predicted_yes_probability:.3f}",
                f"{p.kalshi_implied_probability:.3f}",
                f"[{edge_color}]{p.edge_bps:+d}[/]"
                if edge_color
                else f"{p.edge_bps:+d}",
                f"[{_confidence_style(p.confidence)}]{p.confidence}[/]",
            )
        console.print(pred_table)

        sizing_table = Table(
            title=f"[{_CREAM}]Kelly sizing (same order)[/]",
            title_justify="left",
            box=box.SIMPLE_HEAVY,
            show_lines=False,
            header_style=_LAVENDER,
        )
        sizing_table.add_column("Event", style=_SKY, overflow="fold", min_width=24)
        sizing_table.add_column("Winner", style=f"bold {_LAVENDER}", overflow="fold", min_width=14)
        sizing_table.add_column("Entry", justify="right")
        sizing_table.add_column("Full K", justify="right")
        sizing_table.add_column("Capped ½K", justify="right")

        for s in ranked:
            p, z = s.prediction, s.sizing
            event_display = p.event_title or p.event_ticker
            sizing_table.add_row(
                event_display,
                p.predicted_winner,
                f"${z.entry_price_dollars:.2f}"
                if z.entry_price_dollars is not None
                else "—",
                f"{z.full_kelly_fraction:.1%}",
                f"[bold {_MINT}]{z.capped_half_kelly_fraction:.1%}[/]",
            )
        console.print(sizing_table)

        reasoning_table = Table(
            title=f"[{_CREAM}]Director reasoning (same order)[/]",
            title_justify="left",
            box=box.SIMPLE_HEAVY,
            show_lines=True,
            header_style=_LAVENDER,
        )
        reasoning_table.add_column("Event", style=_SKY, overflow="fold", min_width=20)
        reasoning_table.add_column("Winner", style=f"bold {_LAVENDER}", overflow="fold", min_width=14)
        reasoning_table.add_column("Reasoning", overflow="fold")
        for s in ranked:
            p = s.prediction
            reasoning_table.add_row(
                p.event_title or p.event_ticker,
                p.predicted_winner,
                p.reasoning,
            )
        console.print(reasoning_table)

        # Flags get their own table so long notes don't break the main-table rows.
        flag_rows: list[tuple[str, str, str]] = []
        for s in ranked:
            label = s.prediction.event_title or s.prediction.event_ticker
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
            err_table.add_row(e.event_ticker, e.stage, e.error)
        console.print(err_table)

    if not result.predictions and not result.errors:
        console.print(f"[{_PEACH}]No predictions generated.[/]")
