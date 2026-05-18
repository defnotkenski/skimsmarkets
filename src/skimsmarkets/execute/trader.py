"""Execute orchestrator: ranked JSONL → filter → match → safety → order → audit.

One async entry point (`run_execute`) reads a `logs/runs/<run_id>.jsonl`,
walks every passing row through the matcher and (if `--live`) the
Kalshi order endpoint, and persists one `TradeRow` per row to
`logs/trades/<run_id>.jsonl`.

Every row produces an audit entry — skip, dry-run, or filled — so the
log is a complete record of what execute considered, why it acted or
didn't, and what Kalshi returned. The run never aborts mid-slate;
per-row failures degrade into `skipped` rows with `skip_reason` set,
matching the ranker's "per-event drops, not per-run aborts" invariant.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

import httpx

from skimsmarkets import config as cfg
from skimsmarkets.config import Config
from skimsmarkets.execute.audit import (
    executed_event_ids,
    write_trade_row,
)
from skimsmarkets.execute.filters import filter_rows
from skimsmarkets.execute.reporting import ExecuteDisplay
from skimsmarkets.kalshi.client import KalshiClient, KalshiOrderError
from skimsmarkets.kalshi.matcher import MatchOutcome, find_kalshi_match
from skimsmarkets.kalshi.models import (
    KalshiEvent,
    MarketPosition,
    OrderRequest,
    OrderResponse,
)
from skimsmarkets.ev import compute_ev_per_dollar
from skimsmarkets.retro.jsonl import (
    iter_predictions,
    run_path_for_id,
    trades_log_path,
)
from skimsmarkets.retro.models import PredictionRow, TradeRow

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecuteOptions:
    """User-facing parameters for one `skims execute` invocation."""

    run_id: str
    dry_run: bool
    bet_size_cents: int
    max_position_cents: int
    max_open_exposure_cents: int
    confidence: list[Literal["low", "medium", "high"]] | None = None
    min_defensibility: float | None = None
    no_negative_edge: bool = False
    sports: list[str] | None = None
    risk_buckets: list[str] | None = None
    # Parallel slate-side filter for `--mode ev`. CLI dispatch zeroes
    # `risk_buckets` and populates `ev_buckets` when mode=ev (and vice-
    # versa) so the operator picks exactly one bucket-dimension filter
    # per run. `filter_rows` itself ANDs both if a caller passes both —
    # the mutual-exclusion semantic is a CLI-layer convention, not a
    # type-system guarantee, to keep the filter primitive orthogonal.
    ev_buckets: list[str] | None = None
    # Informational — surfaces in logs / future routing. The actual
    # bucket-dimension filter switch is driven by which of
    # `risk_buckets` / `ev_buckets` is non-None.
    mode: Literal["confidence", "ev"] = "confidence"
    min_market_implied_prob: float | None = None
    # Skip a matched trade when the *current* Kalshi YES ask is at or above
    # this implied-probability ceiling. The rank-time slate filter only saw
    # the Polymarket price; by execute time the Kalshi line can have drifted
    # past the threshold. None = gate inactive (the CLI resolves it to
    # `cfg.MAX_IMPLIED_PROBABILITY`, the same constant the rank slate uses).
    max_implied_probability: float | None = None
    # EV-sizing knobs (opt-in). `bankroll_cents=None` keeps the legacy
    # uniform-sizing path; setting a positive value flips the trader into
    # fractional-Kelly sizing where the per-trade ceiling is
    # `int(bankroll_cents * min(f_star * kelly_multiplier, kelly_max_fraction))`
    # — see `kelly_bet_size`. The Kelly fraction is computed off the LIVE
    # Kalshi yes_ask (the price we'll actually pay), not the rank-time
    # Polymarket implied. `kelly_max_fraction` is intentionally a config
    # constant rather than a CLI flag — the 2% cap is a safety rail, not a
    # tuning knob.
    bankroll_cents: int | None = None
    kelly_multiplier: float = 0.25
    kelly_max_fraction: float = 0.02
    # When set, skip any matched row whose live-line EV-per-dollar (computed
    # from Kalshi yes_ask + predicted_yes_probability) is below this floor.
    # Independent of Kelly mode — uniform-sizing users can also gate on EV.
    # None = gate inactive (every matched row passes).
    min_ev_threshold: float | None = None


@dataclass
class ExecuteSummary:
    """Tally of what happened in one run (printed at the end)."""

    total_predictions: int = 0
    passed_filters: int = 0
    filled: int = 0
    partial: int = 0
    submitted: int = 0
    skipped: int = 0
    skipped_dry_run: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)
    total_filled_cost_cents: int = 0


async def run_execute(
    opts: ExecuteOptions,
    *,
    config: Config,
    display: ExecuteDisplay | None = None,
) -> tuple[ExecuteSummary, int]:
    """Main entry point — async.

    Returns `(summary, open_exposure_cents_post_run)` so the CLI's
    final summary panel can surface the post-run exposure utilisation
    alongside the dollar total. `open_exposure_cents` is the snapshot
    taken at pre-flight (NOT updated for this-run fills), which keeps
    the panel's "cap used" line aligned with the same number the
    exposure-cap gate enforced during the loop.

    When `display` is None (tests, non-CLI callers) all visual hooks
    no-op and behaviour is unchanged from the pre-Rich version.
    """
    _validate(opts, config)

    run_path = run_path_for_id(opts.run_id)
    if not run_path.exists():
        raise RuntimeError(f"No run log at {run_path}")

    if display is not None:
        display.start_phase("load")
    rows = list(iter_predictions(run_path))
    summary = ExecuteSummary(total_predictions=len(rows))
    if display is not None:
        display.complete_phase("load")
    if not rows:
        log.warning("execute: run %s has no prediction rows", opts.run_id)
        return summary, 0

    if display is not None:
        display.start_phase("filter")
    filtered = list(filter_rows(
        rows,
        confidence=opts.confidence,
        min_defensibility=opts.min_defensibility,
        no_negative_edge=opts.no_negative_edge,
        sports=opts.sports,
        risk_buckets=opts.risk_buckets,
        ev_buckets=opts.ev_buckets,
        min_market_implied_prob=opts.min_market_implied_prob,
    ))
    summary.passed_filters = len(filtered)
    log.info(
        "execute: %d / %d rows passed filters", len(filtered), len(rows),
    )
    if display is not None:
        display.complete_phase("filter")
    if not filtered:
        return summary, 0

    audit_path = trades_log_path(opts.run_id)
    # Intra-run idempotency: any prediction whose event already has an
    # executed audit row in this run's log is skipped — re-running
    # `skims execute --live` against the same run_id should not place
    # a second order for a prediction we already acted on.
    already_done = executed_event_ids(opts.run_id)
    if already_done:
        log.info(
            "execute: %d event(s) already executed in prior run of %s — "
            "will skip on dedup",
            len(already_done), opts.run_id,
        )
    this_run_pending_cents = 0

    async with httpx.AsyncClient(timeout=20.0) as http:
        client = KalshiClient(
            base_url=cfg.KALSHI_API_BASE,
            http=http,
            api_key_id=config.kalshi_api_key_id,
            private_key_path=config.kalshi_private_key_path,
            private_key_pem=config.kalshi_private_key_pem,
        )
        if display is not None:
            display.start_phase("exposure")
        open_exposure_cents = await _prefetch_open_exposure(
            client, opts=opts, config=config,
        )
        if display is not None:
            display.complete_phase("exposure")
        log.info(
            "execute: open Kalshi exposure = %d cents (cap %d)",
            open_exposure_cents, opts.max_open_exposure_cents,
        )

        if display is not None:
            display.start_phase("events")
        events = await _prefetch_events(client)
        if display is not None:
            display.complete_phase("events")
        log.info(
            "execute: pre-fetched %d Kalshi events across %d series",
            len(events), len(cfg.KALSHI_TENNIS_SERIES_TICKERS),
        )

        # Seed all pending trade rows up front so the user sees the full
        # pipeline of trades that are coming — each row then updates in
        # place via `update_trade` as `_process_row` resolves it.
        if display is not None:
            for row in filtered:
                display.add_pending(row)

        for row in filtered:
            audit_row = await _process_row(
                row=row,
                events=events,
                client=client,
                opts=opts,
                open_exposure_cents=open_exposure_cents,
                this_run_pending_cents=this_run_pending_cents,
                already_done=already_done,
            )
            write_trade_row(audit_row, audit_path)
            this_run_pending_cents += audit_row.fill_total_cost_cents
            _bump(summary, audit_row)
            if display is not None:
                display.update_trade(row, audit_row)

    return summary, open_exposure_cents


def _validate(opts: ExecuteOptions, config: Config) -> None:
    if not opts.dry_run:
        has_key_material = bool(
            config.kalshi_private_key_path or config.kalshi_private_key_pem
        )
        if not config.kalshi_api_key_id or not has_key_material:
            raise RuntimeError(
                "--live requires KALSHI_API_KEY_ID and either "
                "KALSHI_PRIVATE_KEY_PATH (file) or "
                "KALSHI_PRIVATE_KEY_PEM (inline) in your environment. "
                "Drop --live or configure credentials."
            )
    if opts.bet_size_cents > opts.max_position_cents:
        raise RuntimeError(
            f"--bet-size-cents ({opts.bet_size_cents}) exceeds "
            f"--max-position-cents ({opts.max_position_cents}). Either "
            "lower the bet or raise the position cap."
        )
    if opts.bet_size_cents > opts.max_open_exposure_cents:
        raise RuntimeError(
            f"--bet-size-cents ({opts.bet_size_cents}) exceeds "
            f"--max-open-exposure-cents ({opts.max_open_exposure_cents}). "
            "A single trade can't be larger than the entire portfolio "
            "exposure cap — either lower the bet or raise the cap."
        )
    if opts.bankroll_cents is not None and opts.bankroll_cents <= 0:
        raise RuntimeError(
            f"--bankroll-cents must be positive when set, got "
            f"{opts.bankroll_cents}."
        )
    if not (0.0 < opts.kelly_multiplier <= 1.0):
        raise RuntimeError(
            f"--kelly-multiplier must be in (0, 1], got "
            f"{opts.kelly_multiplier}."
        )
    if not (0.0 < opts.kelly_max_fraction <= 1.0):
        raise RuntimeError(
            f"kelly_max_fraction must be in (0, 1], got "
            f"{opts.kelly_max_fraction}."
        )
    if opts.min_ev_threshold is not None and opts.min_ev_threshold < 0:
        raise RuntimeError(
            f"--min-ev-threshold must be non-negative, got "
            f"{opts.min_ev_threshold}."
        )
    if opts.sports:
        invalid = [s for s in opts.sports if s.lower() != "tennis"]
        if invalid:
            raise RuntimeError(
                f"--sport: only `tennis` is supported in v1, got: {invalid}"
            )


def kelly_bet_size(
    *,
    model_p: float | None,
    market_p: float | None,
    bankroll_cents: int,
    kelly_multiplier: float,
    max_fraction: float,
) -> int:
    """Fractional-Kelly bet size in cents — 0 when no edge or invalid inputs.

    Standard Kelly:
      b = (1 - market_p) / market_p          # net odds per $1 risked
      f_star = (b * p - q) / b               # full Kelly fraction (q = 1 - p)
      fraction = min(f_star * kelly_multiplier, max_fraction)
      bet = int(bankroll_cents * fraction)

    Returns 0 (the trader treats this as a skip) when:
      - either probability is None or out of [0, 1]
      - market_p is at the degenerate edges 0 / 1 (payoff undefined)
      - f_star <= 0 (model agrees with or trails the market — no edge)
      - bankroll_cents <= 0

    `int()` truncates (round-down) rather than rounding, so the resulting
    size is a conservative under-bet on fractional cents.
    """
    if model_p is None or market_p is None:
        return 0
    if not (0.0 <= model_p <= 1.0):
        return 0
    if not (0.0 < market_p < 1.0):
        return 0
    if bankroll_cents <= 0:
        return 0
    b = (1.0 - market_p) / market_p
    f_star = (b * model_p - (1.0 - model_p)) / b
    if f_star <= 0:
        return 0
    fraction = min(f_star * kelly_multiplier, max_fraction)
    return int(bankroll_cents * fraction)


async def _prefetch_open_exposure(
    client: KalshiClient, *, opts: ExecuteOptions, config: Config,
) -> int:
    """Sum `market_exposure_dollars` across the account's open positions.

    `--live` requires this number to be accurate or the cap is unsafe —
    surface any failure as a hard error so the operator notices instead
    of silently trading without a cap. (`_validate` already guaranteed
    credentials exist for `--live`, so a missing-cred raise here would
    only fire on dry-run.)

    `--dry-run` is best-effort. If credentials are missing or the API
    call fails, log a warning and treat exposure as 0 so dry-runs
    without Kalshi access still work (preview mode); the gate becomes
    informational rather than enforcing.
    """
    has_credentials = bool(
        config.kalshi_api_key_id
        and (config.kalshi_private_key_path or config.kalshi_private_key_pem)
    )
    if not has_credentials:
        log.warning(
            "execute: dry-run without Kalshi credentials — open-exposure "
            "gate disabled (treating exposure as 0)",
        )
        return 0
    try:
        positions = await client.list_positions()
    except Exception as e:  # noqa: BLE001
        if opts.dry_run:
            log.warning(
                "execute: dry-run open-positions fetch failed (%s); "
                "treating exposure as 0 for preview", e,
            )
            return 0
        raise
    total_cents = sum_exposure_cents(positions)
    log.info(
        "execute: open positions = %d markets, exposure = %d cents",
        len(positions), total_cents,
    )
    return total_cents


def sum_exposure_cents(positions: list[MarketPosition]) -> int:
    """Sum `market_exposure_dollars` across positions, in cents.

    `market_exposure_dollars=None` rows are skipped rather than counted
    as 0 — a None means we couldn't parse the field, not that the
    position is risk-free. Exported (no leading underscore) so the
    smoke tests can exercise the same arithmetic without spinning up
    an HTTP client.
    """
    total = 0
    for pos in positions:
        if pos.market_exposure_dollars is None:
            continue
        total += int(round(pos.market_exposure_dollars * 100))
    return total


async def _prefetch_events(client: KalshiClient) -> list[KalshiEvent]:
    """Fetch all open events across every tennis match-level series.

    Two-step:
      1. Auto-discover the current tennis series tickers via
         `/series` (ATP/WTA prefix + MATCH suffix). Catches new
         sub-tours Kalshi adds without code changes.
      2. Fetch open events for each discovered series.

    Falls back to `cfg.KALSHI_TENNIS_SERIES_TICKERS` if discovery
    returns empty (Kalshi API down, schema change, etc.) so trading
    keeps working on the main tour even when discovery breaks. Logs
    the discovered set so operators can verify on each run.
    """
    try:
        discovered = await client.list_tennis_match_series()
    except Exception as e:  # noqa: BLE001
        log.warning(
            "execute: tennis-series discovery failed (%s); "
            "falling back to hardcoded %s", e, cfg.KALSHI_TENNIS_SERIES_TICKERS,
        )
        discovered = []
    series_list: tuple[str, ...] = tuple(discovered) or cfg.KALSHI_TENNIS_SERIES_TICKERS
    log.info("execute: tennis series in play: %s", series_list)
    events: list[KalshiEvent] = []
    for series in series_list:
        chunk = await client.list_events(series_ticker=series)
        events.extend(chunk)
    return events


def _implied_at_or_above_max(
    yes_ask: float, max_implied_probability: float | None,
) -> bool:
    """True when the live Kalshi YES ask hits the implied-probability ceiling.

    Mirrors the rank-time slate filter (`polymarket/slate.py`), which drops a
    market whose favorite mid sits *at or above* `max_implied_probability` —
    so the execute-time gate uses the same `>=` comparison. A `None` ceiling
    means the gate is inactive and every ask passes.
    """
    return (
        max_implied_probability is not None
        and yes_ask >= max_implied_probability
    )


async def _process_row(
    *,
    row: PredictionRow,
    events: list[KalshiEvent],
    client: KalshiClient,
    opts: ExecuteOptions,
    open_exposure_cents: int,
    this_run_pending_cents: int,
    already_done: set[str],
) -> TradeRow:
    """Run one filtered prediction through match/safety/order → `TradeRow`.

    Pure-ish: only side effect is the Kalshi POST (only on `--live` +
    matched + within caps + not already executed). All other branches
    return a fully-formed audit row without touching the network.

    Per-trade sizing pipeline:
      1. matcher → live Kalshi yes_ask
      2. implied-prob ceiling gate (skip if drifted past threshold)
      3. EV gate (skip if live-line EV below `min_ev_threshold`)
      4. compute per-trade size — Kelly if `bankroll_cents` is set,
         else uniform `opts.bet_size_cents` (legacy path)
      5. exposure-cap reconciliation — pro-rate to fit headroom in Kelly
         mode, refuse in uniform mode (preserves legacy behavior)

    Skipped rows record `bet_size_cents` differently per branch — pre-
    sizing skips carry `opts.bet_size_cents` (the would-have-been ceiling)
    so the audit row's meaning is consistent with uniform-mode runs;
    post-sizing skips carry the actually computed Kelly size so the
    operator can see what the trader had concluded before the cap or
    other gate intervened.
    """
    base = _audit_base(row, opts, bet_size_cents=opts.bet_size_cents)
    # Intra-run idempotency check BEFORE the matcher — even cheaper:
    # no need to scan Kalshi events for a row we're going to skip.
    if row.event_id in already_done:
        return TradeRow(
            **base,
            fill_status="skipped",
            skip_reason="already_executed_in_run",
        )
    outcome = find_kalshi_match(row, events)

    if outcome.kind != "matched":
        return _skip_row(base, outcome, reason=_skip_reason(outcome))

    market = outcome.market
    assert market is not None  # outcome.kind == "matched"
    yes_ask = market.yes_ask_dollars

    # Current-price gate: the rank-time slate filter only saw Polymarket's
    # implied probability. By the time we execute, the Kalshi YES ask can
    # have drifted to or past the configured ceiling — re-check the live
    # line and skip rather than buy in above threshold. `find_kalshi_match`
    # guarantees `yes_ask` is a real float for a matched outcome.
    if _implied_at_or_above_max(yes_ask, opts.max_implied_probability):
        log.info(
            "execute: %s current ask %.3f at/above max implied %.3f — skip",
            market.ticker, yes_ask, opts.max_implied_probability,
        )
        return _skip_row(
            base, outcome, reason="implied_at_or_above_max", market=market,
        )

    # EV gate (active when --min-ev-threshold is set). The persisted
    # `row.ev_per_dollar` was computed off the rank-time Polymarket
    # probability; we recompute here against the LIVE Kalshi yes_ask so
    # the gate reflects the actual line we'd take. Skips when EV is
    # uncomputable (degenerate inputs) or below the floor.
    if opts.min_ev_threshold is not None:
        live_ev = compute_ev_per_dollar(
            row.predicted_yes_probability, yes_ask,
        )
        if live_ev is None or live_ev < opts.min_ev_threshold:
            log.info(
                "execute: %s live EV %s below threshold %.3f — skip",
                market.ticker,
                f"{live_ev:.3f}" if live_ev is not None else "None",
                opts.min_ev_threshold,
            )
            return _skip_row(
                base, outcome, reason="ev_below_threshold", market=market,
            )

    # Per-trade sizing. Kelly mode (when --bankroll-cents is set) sizes
    # off the LIVE Kalshi yes_ask — the price we'll actually pay — rather
    # than the rank-time Polymarket implied. `opts.bet_size_cents` and
    # `opts.max_position_cents` cap the Kelly result from above as
    # belt-and-suspenders ceilings.
    if opts.bankroll_cents is not None:
        kelly_size = kelly_bet_size(
            model_p=row.predicted_yes_probability,
            market_p=yes_ask,
            bankroll_cents=opts.bankroll_cents,
            kelly_multiplier=opts.kelly_multiplier,
            max_fraction=opts.kelly_max_fraction,
        )
        if kelly_size <= 0:
            # No positive Kelly fraction — model agrees with / trails the
            # market on this row's live line, OR inputs were degenerate.
            # In practice the EV gate above already catches the no-edge
            # case when active; this branch fires for `--bankroll-cents`
            # without `--min-ev-threshold`.
            log.info(
                "execute: %s Kelly size 0 (no edge at live ask %.3f) — skip",
                market.ticker, yes_ask,
            )
            return _skip_row(
                _audit_base(row, opts, bet_size_cents=0),
                outcome, reason="kelly_zero_size", market=market,
            )
        bet_size_cents = min(
            kelly_size, opts.bet_size_cents, opts.max_position_cents,
        )
    else:
        bet_size_cents = opts.bet_size_cents

    # Re-stamp base now that we know the actual per-trade size for this row;
    # downstream audit branches read from `base` for the final TradeRow shape.
    base = _audit_base(row, opts, bet_size_cents=bet_size_cents)

    # Portfolio exposure cap. Uniform mode preserves the legacy
    # all-or-nothing behaviour (refuse the row if the ceiling won't fit).
    # Kelly mode pro-rates instead — a partial bet within headroom is
    # strictly better than skipping, because Kelly already over-shrinks
    # vs the un-capped optimum.
    if opts.bankroll_cents is not None:
        headroom = opts.max_open_exposure_cents - (
            open_exposure_cents + this_run_pending_cents
        )
        if headroom <= 0:
            log.info(
                "execute: %s exposure cap exhausted "
                "(open=%d + pending=%d ≥ cap=%d) — skip",
                market.ticker, open_exposure_cents, this_run_pending_cents,
                opts.max_open_exposure_cents,
            )
            return _skip_row(
                _audit_base(row, opts, bet_size_cents=0),
                outcome, reason="exposure_cap_exhausted", market=market,
            )
        if headroom < bet_size_cents:
            log.info(
                "execute: %s pro-rating Kelly size %d → %d cents to fit headroom",
                market.ticker, bet_size_cents, headroom,
            )
            bet_size_cents = headroom
            base = _audit_base(row, opts, bet_size_cents=bet_size_cents)
    else:
        # We use the per-trade ceiling (`bet_size_cents`) rather than the
        # expected ask so the gate is monotone — a partial fill that ends
        # up cheaper than expected won't retroactively let a later trade
        # slip through.
        projected = (
            open_exposure_cents + this_run_pending_cents + bet_size_cents
        )
        if projected > opts.max_open_exposure_cents:
            log.info(
                "execute: %s would breach exposure cap (projected=%d, cap=%d) — skip",
                market.ticker, projected, opts.max_open_exposure_cents,
            )
            return _skip_row(
                base, outcome, reason="exposure_cap_exceeded", market=market,
            )

    if opts.dry_run:
        return TradeRow(
            **base,
            kalshi_event_ticker=outcome.event_ticker,
            market_ticker=market.ticker,
            kalshi_yes_ask_dollars_at_decision=yes_ask,
            fill_status="skipped_dry_run",
        )

    # Live path. Idempotency token uniquely identifies this attempt;
    # Kalshi dedupes retries by it. Audit row keeps it so a manual
    # post-hoc reconciliation can pair audit ↔ Kalshi order.
    client_order_id = str(uuid4())
    # Order construction. Two budget-enforcing knobs:
    #   - yes_price: per-contract price ceiling (current ask + small slippage)
    #   - count: max contracts; sized so `count × yes_price ≤ bet_size_cents`
    # We deliberately don't set `buy_max_cost` — per Kalshi's docs it
    # forces FOK behaviour, which rejects on insufficient resting volume
    # (the common case for thin tennis books). Instead we use
    # `time_in_force="immediate_or_cancel"` (model default) so partial
    # fills land and the unfilled remainder cancels.
    yes_ask_cents = int(round(yes_ask * 100)) if yes_ask else 1
    yes_price_cents = min(
        99, yes_ask_cents + cfg.KALSHI_MARKET_ORDER_SLIPPAGE_CENTS,
    )
    # floor-div, not ceil: keeps `count × yes_price ≤ bet_size_cents` as
    # a hard worst-case ceiling. At ask=46 + 5¢ buffer = 51, bet=2500,
    # count=49 → max spend 49 × 51 = 2499 ≤ 2500.
    count = max(1, bet_size_cents // yes_price_cents)
    order_req = OrderRequest(
        ticker=market.ticker,
        action="buy",
        side="yes",
        count=count,
        yes_price=yes_price_cents,
        client_order_id=client_order_id,
    )
    try:
        order_resp, raw = await client.place_order(order_req)
    except KalshiOrderError as e:
        # Surface BOTH the response body and our request body so the
        # audit row carries enough context to diagnose schema drift
        # (Kalshi field rename, missing required field, etc.) without
        # needing to retry the live POST.
        log.warning(
            "execute: order POST → %d for %s: %s",
            e.status, market.ticker, e.detail,
        )
        return TradeRow(
            **base,
            kalshi_event_ticker=outcome.event_ticker,
            market_ticker=market.ticker,
            kalshi_yes_ask_dollars_at_decision=yes_ask,
            client_order_id=client_order_id,
            fill_status="skipped",
            skip_reason="api_error",
            raw_response_excerpt={
                "kalshi_status": e.status,
                "kalshi_detail": e.detail,
                "request_body": e.request_body,
            },
        )
    except httpx.HTTPError as e:
        # Network / timeout / connection — Kalshi never saw the request
        # or we never saw the response. No body to capture.
        log.warning("execute: order POST failed for %s (%s)", market.ticker, e)
        return TradeRow(
            **base,
            kalshi_event_ticker=outcome.event_ticker,
            market_ticker=market.ticker,
            kalshi_yes_ask_dollars_at_decision=yes_ask,
            client_order_id=client_order_id,
            fill_status="skipped",
            skip_reason="api_error",
            raw_response_excerpt={"error": repr(e)},
        )

    return _from_response(
        base=base,
        outcome=outcome,
        market_ticker=market.ticker,
        yes_ask=yes_ask,
        client_order_id=client_order_id,
        order_resp=order_resp,
        raw=raw,
        requested_count=count,
    )


def _audit_base(
    row: PredictionRow, opts: ExecuteOptions, *, bet_size_cents: int,
) -> dict[str, Any]:
    """Fields shared by every audit row regardless of outcome path.

    `bet_size_cents` is passed in (rather than read from `opts`) so the
    Kelly path can record the per-row sized amount on the audit row —
    the legacy uniform path always passes `opts.bet_size_cents`, so
    semantics are unchanged when Kelly mode is off.
    """
    return {
        "record_type": "trade",
        "run_id": opts.run_id,
        "audit_timestamp": datetime.now(UTC),
        "event_id": row.event_id,
        "market_slug": row.market_slug,
        "sport_type": row.sport_type,
        "event_title": row.event_title,
        "predicted_winner": row.predicted_winner,
        "predicted_yes_probability": row.predicted_yes_probability,
        "confidence": row.confidence,
        "defensibility_score": row.defensibility_score,
        "negative_edge": row.negative_edge,
        "side": "yes",
        "bet_size_cents": bet_size_cents,
        "dry_run": opts.dry_run,
    }


_SKIP_REASON_BY_KIND: dict[str, str] = {
    "unparseable_players": "unparseable_players",
    "no_kalshi_match": "no_kalshi_match",
    "ambiguous_match": "ambiguous_match",
    "no_yes_market": "no_yes_market",
    "market_closed": "market_closed",
}


def _skip_reason(outcome: MatchOutcome) -> str:
    return _SKIP_REASON_BY_KIND[outcome.kind]


def _skip_row(
    base: dict[str, Any],
    outcome: MatchOutcome,
    *,
    reason: str,
    market: Any = None,
) -> TradeRow:
    market = market or outcome.market
    return TradeRow(
        **base,
        kalshi_event_ticker=outcome.event_ticker,
        market_ticker=market.ticker if market else None,
        kalshi_yes_ask_dollars_at_decision=(
            market.yes_ask_dollars if market else None
        ),
        fill_status="skipped",
        skip_reason=reason,
    )


def _from_response(
    *,
    base: dict[str, Any],
    outcome: MatchOutcome,
    market_ticker: str,
    yes_ask: float | None,
    client_order_id: str,
    order_resp: OrderResponse,
    raw: dict[str, Any],
    requested_count: int,
) -> TradeRow:
    """Map Kalshi's order response to a `TradeRow` fill status.

    With IOC time-in-force:
      - `filled`: fill_count == requested_count
      - `partial`: 0 < fill_count < requested_count
      - `skipped` (`no_fill`): fill_count == 0 (book empty at price)
      - `submitted`: order_id present, status not "executed", no fill
        info — async fill or unfamiliar status string.

    All money on the wire is **dollars**; we convert to cents here so
    the audit row carries the same units as `bet_size_cents`.
    """
    fill_count = int(order_resp.fill_count_fp or 0)
    # Contract cost: taker is what we pay when crossing into resting
    # offers (market buy → always taker). Maker would be non-zero only
    # if part of the order rested and matched later — IOC cancels rest,
    # so it's typically 0.
    cost_dollars = (
        (order_resp.taker_fill_cost_dollars or 0.0)
        + (order_resp.maker_fill_cost_dollars or 0.0)
    )
    cost_cents = int(round(cost_dollars * 100))
    fee_dollars = (
        (order_resp.taker_fees_dollars or 0.0)
        + (order_resp.maker_fees_dollars or 0.0)
    )
    fee_cents = int(round(fee_dollars * 100))
    avg_cents = (
        int(round(cost_cents / fill_count)) if fill_count > 0 else None
    )

    # Kalshi response `status` field is one of "resting" / "canceled" /
    # "executed" per the docs. We map onto our four states:
    #   any fills present → filled / partial
    #   canceled with no fills → skipped (book empty at price)
    #   unrecognised status with order_id → submitted (parser couldn't
    #     classify but Kalshi accepted; surface for inspection)
    #   nothing → skipped (no_fill)
    kalshi_status = (order_resp.status or "").lower()
    status: Literal["filled", "partial", "submitted", "skipped"]
    skip_reason: str | None = None
    if fill_count > 0 and fill_count >= requested_count:
        status = "filled"
    elif fill_count > 0:
        status = "partial"
    elif kalshi_status == "canceled":
        status = "skipped"
        skip_reason = "no_fill"
    elif order_resp.order_id:
        status = "submitted"
    else:
        status = "skipped"
        skip_reason = "no_fill"
    return TradeRow(
        **base,
        kalshi_event_ticker=outcome.event_ticker,
        market_ticker=market_ticker,
        kalshi_yes_ask_dollars_at_decision=yes_ask,
        client_order_id=client_order_id,
        order_id=order_resp.order_id,
        fill_contracts=fill_count,
        fill_total_cost_cents=cost_cents,
        fill_avg_price_cents=avg_cents,
        fill_fees_cents=fee_cents,
        fill_status=status,
        skip_reason=skip_reason,
        raw_response_excerpt=_excerpt(raw),
    )


def _excerpt(raw: Any) -> dict[str, Any] | None:
    """Trim the raw response to a forensic snapshot — top-level dict, no nested explosions."""
    if not isinstance(raw, dict):
        return {"raw": str(raw)[:500]}
    # Keep top-level keys verbatim; truncate long string values.
    return {
        k: (v if not isinstance(v, str) else v[:200])
        for k, v in raw.items()
    }


def _bump(summary: ExecuteSummary, row: TradeRow) -> None:
    if row.fill_status == "filled":
        summary.filled += 1
        summary.total_filled_cost_cents += row.fill_total_cost_cents
    elif row.fill_status == "partial":
        summary.partial += 1
        summary.total_filled_cost_cents += row.fill_total_cost_cents
    elif row.fill_status == "submitted":
        summary.submitted += 1
        summary.total_filled_cost_cents += row.fill_total_cost_cents
    elif row.fill_status == "skipped_dry_run":
        summary.skipped_dry_run += 1
    else:
        summary.skipped += 1
        if row.skip_reason:
            summary.skip_reasons[row.skip_reason] = (
                summary.skip_reasons.get(row.skip_reason, 0) + 1
            )
