"""Rich-formatted summary tables for CLI output."""

from __future__ import annotations

from datetime import UTC, datetime

from rich import box
from rich.console import Console
from rich.table import Table

from skimsmarkets.kalshi.models import KalshiEvent
from skimsmarkets.pipeline import RunResult


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
    return {"buy_yes": "bold green", "buy_no": "bold red", "pass": "dim"}.get(rec, "")


def _confidence_style(c: str) -> str:
    return {"high": "green", "medium": "yellow", "low": "red"}.get(c, "")


def print_events_table(events: list[KalshiEvent], series_filter: str | None) -> None:
    console = Console()
    title = (
        "Live Kalshi sports events"
        + (f" — series={series_filter}" if series_filter else "")
        + f" ({len(events)} events)"
    )
    table = Table(title=title, box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("Series", style="cyan")
    table.add_column("Event")
    table.add_column("Market (yes side)")
    table.add_column("Yes bid/ask", justify="right")
    table.add_column("Implied", justify="right")
    table.add_column("24h vol", justify="right")
    table.add_column("Closes in", justify="right")

    rows = 0
    for e in events:
        for m in e.markets:
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
                _rel_time(m.close_time),
            )
            rows += 1

    if rows == 0:
        console.print(
            "[yellow]No live markets found"
            + (f" for {series_filter}" if series_filter else "")
            + ".[/yellow]"
        )
        return
    console.print(table)


def print_run_summary(result: RunResult) -> None:
    console = Console()
    console.rule(f"[bold]Run {result.run_id}[/bold]")
    console.print(
        f"Fetched events: [cyan]{result.fetched_events}[/cyan]  "
        f"Considered markets: [cyan]{result.considered_markets}[/cyan]  "
        f"Predictions: [green]{len(result.predictions)}[/green]  "
        f"Errors: [red]{len(result.errors)}[/red]"
    )

    if result.predictions:
        table = Table(
            title="Predictions (sorted by capped-Kelly stake)",
            box=box.SIMPLE_HEAVY,
            show_lines=False,
        )
        table.add_column("Market", style="cyan", no_wrap=False)
        table.add_column("Rec", justify="center")
        table.add_column("Pred", justify="right")
        table.add_column("Kalshi", justify="right")
        table.add_column("Edge (bps)", justify="right")
        table.add_column("Conf", justify="center")
        table.add_column("Side", justify="center")
        table.add_column("Entry", justify="right")
        table.add_column("Full K", justify="right")
        table.add_column("Capped ½K", justify="right")
        table.add_column("Flags")

        ranked = sorted(
            result.predictions,
            key=lambda s: s.sizing.capped_half_kelly_fraction,
            reverse=True,
        )
        for s in ranked:
            p, z = s.prediction, s.sizing
            edge_color = (
                "green" if p.edge_bps > 0 else ("red" if p.edge_bps < 0 else "")
            )
            flags = (
                "; ".join(z.notes)
                if z.notes
                else (
                    "; ".join(p.disagreements_flagged)
                    if p.disagreements_flagged
                    else ""
                )
            )
            table.add_row(
                p.market_ticker,
                f"[{_rec_style(p.recommendation)}]{p.recommendation}[/]",
                f"{p.predicted_yes_probability:.3f}",
                f"{p.kalshi_implied_probability:.3f}",
                f"[{edge_color}]{p.edge_bps:+d}[/]"
                if edge_color
                else f"{p.edge_bps:+d}",
                f"[{_confidence_style(p.confidence)}]{p.confidence}[/]",
                z.side,
                f"${z.entry_price_dollars:.2f}"
                if z.entry_price_dollars is not None
                else "—",
                f"{z.full_kelly_fraction:.1%}",
                f"[bold]{z.capped_half_kelly_fraction:.1%}[/]",
                flags[:60] + ("…" if len(flags) > 60 else ""),
            )
        console.print(table)

    if result.errors:
        err_table = Table(title="Errors", box=box.SIMPLE, show_lines=False)
        err_table.add_column("Market", style="cyan")
        err_table.add_column("Stage", style="yellow")
        err_table.add_column("Error")
        for e in result.errors:
            err_table.add_row(e.market_ticker, e.stage, e.error[:120])
        console.print(err_table)

    if not result.predictions and not result.errors:
        console.print("[yellow]No predictions generated.[/yellow]")
