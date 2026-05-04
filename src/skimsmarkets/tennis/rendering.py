"""Render `TennisStatsContext` as a compact prompt-friendly text block.

Same posture as `unusual_whales/rendering.py:render_uw_block` — pure
formatting, kept out of the agent modules so the rendering can be
unit-tested without spinning up a fetcher and so any future consumer
(CLI debug, JSONL pretty-print) shares the same shape.

Rendered block goes onto the per-event user message of the *statistics*
fetcher only, appended after `render_context(event)` and the existing
`render_sport_hint(...)` output. The reasoner sees the same string
because reasoners receive the same event context the fetcher does
(`agents/reasoners.py`).

Token budget: aim for ~150–300 tokens per match. Anything denser starts
to compete with the existing tennis sport hint without adding new signal.
"""

from __future__ import annotations

from skimsmarkets.tennis.models import (
    TennisHeadToHead,
    TennisPlayerStats,
    TennisStatsContext,
)


def _fmt_record(rec: tuple[int, int] | None) -> str:
    if rec is None:
        return "?"
    wins, losses = rec
    return f"{wins}-{losses}"


def _fmt_pct(v: float | None) -> str | None:
    """Render a [0,1] ratio as an integer percent, or None to skip the line."""
    if v is None:
        return None
    return f"{v * 100:.0f}%"


def _fmt_player(label: str, p: TennisPlayerStats) -> list[str]:
    """Render one player's lines under a `Player A:` / `Player B:` header.

    Each piece of data lives on its own indented sub-line so the LLM
    parses the block as a list rather than dense prose. Order matches
    the reasoner's likely query order — rank first, then recency, then
    career underlying-skill metrics last. Lines whose data is absent are
    suppressed entirely (rather than rendered with `?`s) so the block
    stays compact on thinly-covered players.
    """
    lines = [f"  {label}: {p.name}"]
    bits: list[str] = []
    if p.rank_singles is not None:
        rp = f"{p.rank_singles}" + (
            f" ({p.rank_points:,} pts)" if p.rank_points is not None else ""
        )
        bits.append(f"rank={rp}")
    if p.best_rank_singles is not None:
        # Career-high — meaningful as relative context to the current
        # rank (e.g. `rank=80 best=4` flags a descending veteran).
        bits.append(f"best={p.best_rank_singles}")
    if p.ytd_win_loss is not None:
        bits.append(f"ytd={_fmt_record(p.ytd_win_loss)}")
    if bits:
        lines.append("    " + "  ".join(bits))
    if p.surface_win_loss:
        # Stable surface order so the prompt doesn't churn the cache for
        # the dict-iteration-order reason.
        ordered = []
        for surf in ("hard", "clay", "grass", "carpet"):
            rec = p.surface_win_loss.get(surf)
            if rec is not None:
                ordered.append(f"{surf}={_fmt_record(rec)}")
        # Append any vendor-supplied surfaces we didn't anticipate so
        # nothing is silently dropped.
        for surf, rec in p.surface_win_loss.items():
            if surf not in {"hard", "clay", "grass", "carpet"}:
                ordered.append(f"{surf}={_fmt_record(rec)}")
        if ordered:
            lines.append(f"    surface: {'  '.join(ordered)}")
    if p.last_10_form:
        lines.append(f"    form: {p.last_10_form} (oldest→newest)")
    if p.last_match_date is not None:
        lines.append(f"    last_match: {p.last_match_date.isoformat()}")
    # Career serve/return percentages on a single dense line — the
    # reasoner can read the relative ordering between players at a
    # glance ("70% vs 62% first-serve win" beats "70% first-serve win"
    # alone). Suppressed entirely when the vendor returned no
    # serve/return record.
    serve_bits: list[str] = []
    if (s := _fmt_pct(p.first_serve_in_pct)) is not None:
        serve_bits.append(f"1stIn={s}")
    if (s := _fmt_pct(p.first_serve_win_pct)) is not None:
        serve_bits.append(f"1stWon={s}")
    if (s := _fmt_pct(p.second_serve_win_pct)) is not None:
        serve_bits.append(f"2ndWon={s}")
    if serve_bits:
        lines.append(f"    serve (career): {'  '.join(serve_bits)}")
    bp_bits: list[str] = []
    if (s := _fmt_pct(p.break_point_save_pct)) is not None:
        bp_bits.append(f"saved={s}")
    if (s := _fmt_pct(p.break_point_convert_pct)) is not None:
        bp_bits.append(f"converted={s}")
    if bp_bits:
        lines.append(f"    break-points (career): {'  '.join(bp_bits)}")
    # Tier records — current-year W/L vs elite competition and at the
    # biggest events. Single line, suppressed entirely when none of the
    # three populated.
    tier_bits: list[str] = []
    if p.record_vs_top_10 is not None:
        tier_bits.append(f"vs_top10={_fmt_record(p.record_vs_top_10)}")
    if p.record_at_grand_slam is not None:
        tier_bits.append(f"slam={_fmt_record(p.record_at_grand_slam)}")
    if p.record_at_masters is not None:
        tier_bits.append(f"masters={_fmt_record(p.record_at_masters)}")
    if tier_bits:
        lines.append(f"    tier (YTD): {'  '.join(tier_bits)}")
    return lines


def _fmt_wins_total(rec: tuple[int, int] | None) -> str | None:
    """Render `(wins, total)` as `W/T` (sample-size visible) or None."""
    if rec is None:
        return None
    wins, total = rec
    return f"{wins}/{total}"


def _fmt_h2h(h2h: TennisHeadToHead, name_a: str, name_b: str) -> list[str]:
    lines = [f"  H2H: {name_a} {h2h.a_wins}-{h2h.b_wins} {name_b}"]
    if h2h.last_meeting is not None:
        last_parts = [f"date={h2h.last_meeting.isoformat()}"]
        if h2h.last_meeting_winner:
            last_parts.append(f"winner={h2h.last_meeting_winner}")
        if h2h.last_meeting_surface:
            last_parts.append(f"surface={h2h.last_meeting_surface}")
        if h2h.last_meeting_round:
            # Vendor uses fraction shorthand ("1/2" = SF, "1/4" = QF) —
            # tour-native, no translation.
            last_parts.append(f"round={h2h.last_meeting_round}")
        if h2h.last_meeting_result:
            # Score line distinguishes a straight-sets win from a
            # five-set comeback — adds context the bare winner doesn't.
            last_parts.append(f"score={h2h.last_meeting_result}")
        lines.append(f"    last: {'  '.join(last_parts)}")
    # Matchup-specific clutch records — only meaningful when the pair
    # has actually played; skip the whole line when both players have
    # zero deciders + zero tiebreaks against each other.
    decider_a = _fmt_wins_total(h2h.decider_record_a)
    decider_b = _fmt_wins_total(h2h.decider_record_b)
    tiebreak_a = _fmt_wins_total(h2h.tiebreak_record_a)
    tiebreak_b = _fmt_wins_total(h2h.tiebreak_record_b)
    if decider_a or decider_b:
        lines.append(
            f"    deciders (in matchup): {name_a}={decider_a or '?'}  "
            f"{name_b}={decider_b or '?'}"
        )
    if tiebreak_a or tiebreak_b:
        lines.append(
            f"    tiebreaks (in matchup): {name_a}={tiebreak_a or '?'}  "
            f"{name_b}={tiebreak_b or '?'}"
        )
    cb_a = _fmt_pct(h2h.first_set_lost_match_won_pct_a)
    cb_b = _fmt_pct(h2h.first_set_lost_match_won_pct_b)
    if cb_a is not None or cb_b is not None:
        lines.append(
            f"    comeback rate (set-1 down → match win, in matchup): "
            f"{name_a}={cb_a or '?'}  {name_b}={cb_b or '?'}"
        )
    return lines


def render_tennis_stats_block(ctx: TennisStatsContext) -> str:
    """Compact tennis player-stats render for the statistics fetcher.

    Returns a multi-line string with a labelled header so the reasoner
    can find it deterministically (the statistics reasoner prompt
    references this header literally). Empty bodies aren't expected —
    `has_actionable_signal` is the gate the caller should check first.
    """
    lines: list[str] = []
    header_extras: list[str] = []
    if ctx.surface:
        header_extras.append(f"surface={ctx.surface}")
    if ctx.tournament:
        header_extras.append(f"tournament={ctx.tournament}")
    extras_s = f" ({', '.join(header_extras)})" if header_extras else ""
    # Header provider tag mirrors UW's `Flow signals (Unusual Whales, ...)`
    # convention so the reasoner can identify the block by its leading
    # parenthesis on a single grep.
    lines.append(f"--- Tennis stats (vendor: {ctx.provider}){extras_s} ---")
    lines.extend(_fmt_player("Player A", ctx.player_a))
    lines.extend(_fmt_player("Player B", ctx.player_b))
    if ctx.head_to_head is not None:
        lines.extend(_fmt_h2h(ctx.head_to_head, ctx.player_a.name, ctx.player_b.name))
    return "\n".join(lines)
