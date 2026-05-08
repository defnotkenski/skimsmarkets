"""Render `TennisStatsContext` as a compact prompt-friendly text block.

Same posture as `unusual_whales/rendering.py:render_uw_block` — pure
formatting, kept out of the agent modules so the rendering can be
unit-tested without spinning up a fetcher and so any future consumer
(CLI debug, JSONL pretty-print) shares the same shape.

Three lens-scoped renderers, one source of truth:
- `render_tennis_form_block` — career serve/return rates, surface
  splits, recent form, tier records, career titles. Goes onto the
  `tennis_form_and_surface` fetcher. EXCLUDES H2H, matchup-conditioned
  primitives, career break-points, and handedness — those belong to
  the matchup-and-clutch lens.
- `render_tennis_matchup_block` — H2H counts + per-surface H2H +
  recent meetings + matchup-conditioned per-player records (deciders,
  tiebreaks, set-1 conversions, in-matchup serve/BP) + career
  BP save/convert + handedness. Goes onto the
  `tennis_matchup_and_clutch` fetcher. Returns None when nothing
  clutch-relevant is populated.
- `render_tennis_fatigue_block` — NARROW fatigue-only slice (days
  since last match, match count in last 14d). Goes onto the
  `tennis_conditions_and_context` fetcher. Derives its primitives from
  `last_match_date` + `recent_matches` (the form block also carries
  these) — same source data, different scoped view.

Each lens sees only the slice it owns; this preserves the silo posture
from CLAUDE.md without paying for unused context tokens.

Reasoners see the same strings because reasoners receive the same
event context their fetchers do (`agents/reasoners.py`).

Token budget: form block ~250-330 tokens; matchup block ~80-150 tokens
when H2H is rich, ~30 when only career BP/handedness; fatigue block
~30-40. Each line suppressed independently when its data is absent.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from skimsmarkets.tennis.models import (
    TennisGbtContext,
    TennisH2HMeeting,
    TennisHeadToHead,
    TennisInMatchupStats,
    TennisPlayerStats,
    TennisRecentMatch,
    TennisSimulationContext,
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


def _fmt_wins_total(rec: tuple[int, int] | None) -> str | None:
    """Render `(wins, total)` as `W/T` (sample-size visible) or None."""
    if rec is None:
        return None
    wins, total = rec
    return f"{wins}/{total}"


def _fmt_titles(titles: dict[str, int] | None) -> str | None:
    """Render career titles as a single line `slam=4 masters=9 ...`.

    Stable tier order so the cache doesn't churn on dict iteration order.
    Returns None when nothing populated so the caller can skip the line.
    """
    if not titles:
        return None
    bits: list[str] = []
    for tier_key, label in (
        ("grand_slam", "slam"),
        ("masters", "masters"),
        ("main_tour", "tour"),
        ("tour_finals", "finals"),
    ):
        n = titles.get(tier_key)
        if n is not None and n > 0:
            bits.append(f"{label}={n}")
    return "  ".join(bits) if bits else None


def _fmt_recent_match(m: TennisRecentMatch) -> str:
    """One line per recent match: `WL  DATE  vs OPP  SCORE  ROUND@TIER`.

    Suppress missing pieces (round/tournament/score) so a thinly-populated
    row still renders compactly. The leading W/L is the most informative
    char; placing it first lets the reasoner skim outcomes vertically.
    """
    bits: list[str] = ["W" if m.won else "L"]
    if m.date is not None:
        bits.append(m.date.isoformat())
    bits.append(f"vs {m.opponent_name}")
    if m.result:
        bits.append(m.result)
    rt_bits: list[str] = []
    if m.round:
        rt_bits.append(m.round)
    if m.tournament_tier:
        rt_bits.append(m.tournament_tier)
    if rt_bits:
        bits.append("@" + "/".join(rt_bits))
    return "  ".join(bits)


def _fmt_meeting(meeting: TennisH2HMeeting) -> str:
    """One line per H2H meeting: `DATE  WINNER  SCORE  ROUND@SURFACE/TIER`.

    Same column posture as `_fmt_recent_match` but keyed on winner
    instead of subject-W/L (since the meeting is between *both* subjects
    here). Suppress missing pieces.
    """
    bits: list[str] = []
    if meeting.date is not None:
        bits.append(meeting.date.isoformat())
    if meeting.winner_name:
        bits.append(f"won={meeting.winner_name}")
    if meeting.result:
        bits.append(meeting.result)
    rt_bits: list[str] = []
    if meeting.round:
        rt_bits.append(meeting.round)
    if meeting.surface:
        rt_bits.append(meeting.surface)
    if meeting.tournament_tier:
        rt_bits.append(meeting.tournament_tier)
    if rt_bits:
        bits.append("@" + "/".join(rt_bits))
    return "  ".join(bits)


def _fmt_player_form(label: str, p: TennisPlayerStats) -> list[str]:
    """Render the form-and-surface scoped slice of one player.

    Career rates, surface splits, recent form, tier records, career
    titles. EXCLUDES career break-points and handedness — those are
    style/clutch primitives owned by the matchup-and-clutch lens via
    `_fmt_player_clutch`. The split keeps each lens's user message
    scoped to what it owns.
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
    if p.age_years is not None:
        bits.append(f"age={p.age_years}")
    if p.ytd_win_loss is not None:
        bits.append(f"ytd={_fmt_record(p.ytd_win_loss)}")
    if bits:
        lines.append("    " + "  ".join(bits))
    titles_line = _fmt_titles(p.career_titles)
    if titles_line:
        # Career-achievement bedrock — distinct from YTD records below.
        lines.append(f"    career titles: {titles_line}")
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
            lines.append(f"    surface (YTD): {'  '.join(ordered)}")
    if p.last_10_form:
        lines.append(f"    form: {p.last_10_form} (oldest→newest)")
    # `last_match_date` is redundant with `recent_matches[0].date` when
    # the digest is populated — suppress to avoid duplicating the same
    # date on consecutive lines.
    if p.last_match_date is not None and not p.recent_matches:
        lines.append(f"    last_match: {p.last_match_date.isoformat()}")
    if p.recent_matches:
        # Capped at 3 in the renderer; the fetcher pulls 5 for redundancy.
        # Newest first per the vendor's reverse-chronological order.
        lines.append("    recent (newest first):")
        for m in p.recent_matches[:3]:
            lines.append(f"      {_fmt_recent_match(m)}")
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
    return_bits: list[str] = []
    if (s := _fmt_pct(p.first_serve_return_win_pct)) is not None:
        return_bits.append(f"1stRet={s}")
    if (s := _fmt_pct(p.second_serve_return_win_pct)) is not None:
        return_bits.append(f"2ndRet={s}")
    if return_bits:
        # Distinct from BP-convert (which is conditional on having a BP);
        # this is points-won on every opponent serve.
        lines.append(f"    return (career): {'  '.join(return_bits)}")
    # Tier records — current-year W/L vs elite competition and at the
    # biggest events. Single line, suppressed entirely when none of the
    # four cells populated.
    tier_bits: list[str] = []
    if p.record_vs_top_5 is not None:
        tier_bits.append(f"vs_top5={_fmt_record(p.record_vs_top_5)}")
    if p.record_vs_top_10 is not None:
        tier_bits.append(f"vs_top10={_fmt_record(p.record_vs_top_10)}")
    if p.record_at_grand_slam is not None:
        tier_bits.append(f"slam={_fmt_record(p.record_at_grand_slam)}")
    if p.record_at_masters is not None:
        tier_bits.append(f"masters={_fmt_record(p.record_at_masters)}")
    if tier_bits:
        lines.append(f"    tier (YTD): {'  '.join(tier_bits)}")
    return lines


def _fmt_player_clutch(label: str, p: TennisPlayerStats) -> list[str]:
    """Render the matchup-and-clutch scoped slice of one player.

    Handedness + career break-point save / convert percentages +
    career-aggregate clutch records (tiebreak / decider / comeback /
    close-match) + 180d-recency BP-save. Returns [] when nothing is
    populated so the caller drops the player header.

    The matchup-and-clutch lens owns these because they're style +
    clutch primitives, distinct from the form lens's career
    serve/return rates. Each line is suppressed independently when
    its data is absent so thinly-covered players still render
    compactly.
    """
    career_clutch_present = any(
        v is not None
        for v in (
            p.career_tiebreak_record,
            p.career_decider_record,
            p.career_comeback_record,
            p.career_close_match_record,
            p.break_point_save_pct_180d,
        )
    )
    has_bp = (
        p.break_point_save_pct is not None
        or p.break_point_convert_pct is not None
    )
    if not p.plays and not has_bp and not career_clutch_present:
        return []
    lines: list[str] = [f"  {label}: {p.name}"]
    if p.plays:
        # `plays` carries handedness + backhand style (e.g. "Right-Handed,
        # Two-Handed Backhand"). Pass through verbatim.
        lines.append(f"    plays: {p.plays}")
    bp_bits: list[str] = []
    if (s := _fmt_pct(p.break_point_save_pct)) is not None:
        bp_bits.append(f"saved={s}")
    if (s := _fmt_pct(p.break_point_convert_pct)) is not None:
        bp_bits.append(f"converted={s}")
    if bp_bits:
        lines.append(f"    break-points (career): {'  '.join(bp_bits)}")
    if (s := _fmt_pct(p.break_point_save_pct_180d)) is not None:
        # Recency-windowed BP-save (180d). Read alongside the career
        # rate above — divergence flags an upswing or slump the
        # career figure smooths over.
        lines.append(f"    break-points (180d): saved={s}")
    clutch_bits: list[str] = []
    if (s := _fmt_wins_total(p.career_tiebreak_record)) is not None:
        clutch_bits.append(f"tiebreaks={s}")
    if (s := _fmt_wins_total(p.career_decider_record)) is not None:
        clutch_bits.append(f"deciders={s}")
    if (s := _fmt_wins_total(p.career_comeback_record)) is not None:
        # Comeback denominator is matches where this player lost set 1,
        # so the rate's variance is high — keep the (W/T) form for
        # sample-size visibility rather than a rate alone.
        clutch_bits.append(f"comeback={s}")
    if (s := _fmt_wins_total(p.career_close_match_record)) is not None:
        clutch_bits.append(f"close={s}")
    if clutch_bits:
        # Window: the most recent 50 past-matches (provider's
        # _PAST_MATCHES_FETCH_SIZE). Not "career" in the lifetime sense
        # but stable across the relevant performance arc; the line
        # label says "career" to match the shape of the BP line above
        # and keep the prompt scannable.
        lines.append(f"    career clutch: {'  '.join(clutch_bits)}")
    return lines


def _fmt_in_matchup(
    label: str, name: str, im: TennisInMatchupStats
) -> list[str]:
    """Render one player's matchup-conditioned aggregates as 2-3 lines.

    Single line each for: bo3/bo5 split, decider/tiebreak, set-1
    conversions, in-matchup serve+BP. Suppressed independently when
    each pair of cells is None.
    """
    lines: list[str] = []
    fmt_bits: list[str] = []
    bo3 = _fmt_wins_total(im.bo3_record)
    bo5 = _fmt_wins_total(im.bo5_record)
    if bo3:
        fmt_bits.append(f"bo3={bo3}")
    if bo5:
        # Slams = bo5 (men's). The split distinguishes a matchup that's
        # lopsided AT slams from lopsided overall.
        fmt_bits.append(f"bo5={bo5}")
    if fmt_bits:
        lines.append(f"      {label} ({name}) format: {'  '.join(fmt_bits)}")
    clutch_bits: list[str] = []
    if (s := _fmt_wins_total(im.decider_record)) is not None:
        clutch_bits.append(f"deciders={s}")
    if (s := _fmt_wins_total(im.tiebreak_record)) is not None:
        clutch_bits.append(f"tiebreaks={s}")
    if clutch_bits:
        lines.append(f"      {label} ({name}) clutch: {'  '.join(clutch_bits)}")
    set_bits: list[str] = []
    if (s := _fmt_pct(im.first_set_won_match_won_pct)) is not None:
        set_bits.append(f"set1_up→W={s}")
    if (s := _fmt_pct(im.first_set_lost_match_won_pct)) is not None:
        set_bits.append(f"set1_down→W={s}")
    if set_bits:
        lines.append(f"      {label} ({name}) set-1: {'  '.join(set_bits)}")
    serve_bits: list[str] = []
    if (s := _fmt_pct(im.first_serve_win_pct)) is not None:
        # In-matchup 1st-serve-won — distinct from career-overall.
        serve_bits.append(f"1stWon={s}")
    if (s := _fmt_pct(im.break_point_convert_pct)) is not None:
        serve_bits.append(f"BPconv={s}")
    if serve_bits:
        lines.append(f"      {label} ({name}) in-matchup: {'  '.join(serve_bits)}")
    return lines


def _fmt_h2h(h2h: TennisHeadToHead, name_a: str, name_b: str) -> list[str]:
    lines = [f"  H2H: {name_a} {h2h.a_wins}-{h2h.b_wins} {name_b}"]
    if h2h.surface_h2h:
        # Per-surface H2H counts, stable order for cache stability.
        surface_bits: list[str] = []
        for surf in ("hard", "clay", "grass", "carpet"):
            rec = h2h.surface_h2h.get(surf)
            if rec is not None:
                a_w, b_w = rec
                surface_bits.append(f"{surf}={a_w}-{b_w}")
        if surface_bits:
            lines.append(f"    by surface: {'  '.join(surface_bits)}")
    if h2h.recent_meetings:
        lines.append("    recent meetings (newest first):")
        for meeting in h2h.recent_meetings[:3]:
            lines.append(f"      {_fmt_meeting(meeting)}")
    # Matchup-conditioned per-player blocks. Both blocks are independent —
    # suppress each when its data is missing.
    if h2h.a_in_matchup is not None:
        lines.extend(_fmt_in_matchup("A", name_a, h2h.a_in_matchup))
    if h2h.b_in_matchup is not None:
        lines.extend(_fmt_in_matchup("B", name_b, h2h.b_in_matchup))
    return lines


def _header_extras(ctx: TennisStatsContext) -> str:
    """Trailing parens with surface/tournament context, matching the
    convention used elsewhere (e.g. `render_uw_block`).

    Both lens-scoped blocks (form and matchup) want the same surface +
    tournament context in their header so a reasoner reading either
    block knows which match it's looking at.
    """
    bits: list[str] = []
    if ctx.surface:
        bits.append(f"surface={ctx.surface}")
    if ctx.tournament:
        bits.append(f"tournament={ctx.tournament}")
    return f" ({', '.join(bits)})" if bits else ""


def render_tennis_form_block(ctx: TennisStatsContext) -> str:
    """Form-and-surface scoped render for the `tennis_form_and_surface`
    fetcher.

    Career serve/return rates, surface splits, recent form, tier
    records, career titles. EXCLUDES H2H, matchup-conditioned
    primitives, career break-points, and handedness — those are owned
    by the matchup-and-clutch lens via `render_tennis_matchup_block`.

    Empty bodies aren't expected for tennis events that pass the
    selection gate — `has_actionable_signal` is the upstream check.
    """
    lines: list[str] = [
        f"--- Tennis form & surface (vendor: {ctx.provider}){_header_extras(ctx)} ---"
    ]
    lines.extend(_fmt_player_form("Player A", ctx.player_a))
    lines.extend(_fmt_player_form("Player B", ctx.player_b))
    return "\n".join(lines)


def render_tennis_matchup_block(ctx: TennisStatsContext) -> str | None:
    """Matchup-and-clutch scoped render for the
    `tennis_matchup_and_clutch` fetcher.

    H2H counts + per-surface H2H + recent meetings +
    matchup-conditioned per-player records (deciders, tiebreaks,
    set-1 conversions, in-matchup serve/BP) + career BP-save /
    BP-convert + handedness. Returns None when nothing
    clutch-relevant is populated — both players lack handedness AND
    career BP rates AND there's no H2H — so the caller doesn't append
    a bare header for an empty body. First-time meetings between two
    name-only entries hit this path.
    """
    def _has_player_clutch(p: TennisPlayerStats) -> bool:
        return any(
            v is not None
            for v in (
                p.plays,
                p.break_point_save_pct,
                p.break_point_convert_pct,
                p.break_point_save_pct_180d,
                p.career_tiebreak_record,
                p.career_decider_record,
                p.career_comeback_record,
                p.career_close_match_record,
            )
        )

    if not (
        _has_player_clutch(ctx.player_a)
        or _has_player_clutch(ctx.player_b)
        or ctx.head_to_head is not None
    ):
        return None
    lines: list[str] = [
        f"--- Tennis matchup & clutch (vendor: {ctx.provider}){_header_extras(ctx)} ---"
    ]
    lines.extend(_fmt_player_clutch("Player A", ctx.player_a))
    lines.extend(_fmt_player_clutch("Player B", ctx.player_b))
    if ctx.head_to_head is not None:
        lines.extend(_fmt_h2h(ctx.head_to_head, ctx.player_a.name, ctx.player_b.name))
    return "\n".join(lines)


def _fatigue_lines(
    label: str, p: TennisPlayerStats, today: date, cutoff: date
) -> list[str]:
    """Two indented lines per player: days-since-last-match and 14d
    match count. Per-line suppression when one input is absent. Returns
    [] when neither line has data — caller drops the player header.
    """
    last = p.last_match_date
    recent = p.recent_matches
    if last is None and not recent:
        return []
    out: list[str] = [f"  {label}: {p.name}"]
    if last is not None:
        # Bound days_since_last_match at 0 in case the vendor ships a
        # future-dated row (shouldn't happen, but cheap defensive).
        delta = max(0, (today - last).days)
        out.append(
            f"    days_since_last_match: {delta} "
            f"(last match {last.isoformat()})"
        )
    if recent:
        # Count matches whose date falls within the trailing 14d window.
        # Skip rows whose `date` is None — the vendor occasionally ships
        # a digest row with the date stripped, and counting it would
        # over-state load.
        count = sum(
            1 for m in recent if m.date is not None and m.date >= cutoff
        )
        out.append(f"    match_count_last_14d: {count}")
    return out


def render_tennis_simulation_block(ctx: TennisSimulationContext) -> str:
    """Compact career-baseline Monte Carlo render for the director's
    user message.

    Director-only — same posture as `render_uw_block`. Lenses don't see
    this. Format mirrors the labelled-header convention used elsewhere
    so the director can grep the block by its leading parenthesis.
    """
    n_str = f"{ctx.n_sims:,}"
    p = ctx.p_team_a_wins
    lo = ctx.ci_low
    hi = ctx.ci_high
    pa_serve = ctx.point_win_pct_a_serving
    pb_serve = ctx.point_win_pct_b_serving
    return (
        f"--- Tennis match simulator (provider: {ctx.provider}, n={n_str}) ---\n"
        f"  bo{ctx.best_of}; p(team_a wins) = {p:.3f} "
        f"[95% sampling CI {lo:.3f}-{hi:.3f}]\n"
        f"  point-win on team_a's serve: {pa_serve:.3f}; "
        f"on team_b's serve: {pb_serve:.3f}\n"
        f"  assumptions: {ctx.assumptions}"
    )


def render_tennis_gbt_block(ctx: TennisGbtContext) -> str:
    """Compact GBT-prediction render for the director's user message.

    Director-only — same posture as `render_tennis_simulation_block`.
    Lenses don't see this. Format mirrors the labelled-header
    convention so the director can grep the block by its leading
    parenthesis.

    Top features render as a one-line list of `name=±contribution`
    tokens; the sign tells the director which side the model leaned
    toward (positive = team_a if anchor==team_a, but the contribution
    is anchor-relative so we annotate that explicitly).
    """
    p = ctx.p_team_a_wins
    feat_bits: list[str] = []
    for f in ctx.top_features:
        sign = "+" if f.contribution >= 0 else "−"
        feat_bits.append(f"{f.name}={sign}{abs(f.contribution):.3f}")
    feat_line = ("  top features (anchor-relative log-odds): " + "  ".join(feat_bits)) if feat_bits else ""
    return (
        f"--- Tennis GBT prior (provider: {ctx.provider}, version: {ctx.model_version}) ---\n"
        f"  p(team_a wins) = {p:.3f}\n"
        f"  prior matches: a={ctx.n_prior_matches_a}  b={ctx.n_prior_matches_b}\n"
        + (feat_line + "\n" if feat_line else "")
        + f"  assumptions: {ctx.assumptions}"
    )


def render_tennis_fatigue_block(
    ctx: TennisStatsContext, *, now: datetime | None = None
) -> str | None:
    """Compact fatigue-primitives render for the
    `tennis_conditions_and_context` fetcher.

    Derives two primitives per player from data the form/surface block
    already carries (`last_match_date`, `recent_matches`):
    `days_since_last_match` and `match_count_last_14d`. The conditions
    lens uses these as deterministic inputs to its
    `physical_signed_shift` rather than re-discovering them via web
    search.

    Returns `None` when both players lack `last_match_date` AND have no
    `recent_matches` — there's no point appending an empty header.
    Per-line suppression when one player's data is partial.

    `now` defaults to `datetime.now(UTC)` and is injectable for
    deterministic tests / snapshot-style debugging.
    """
    today = (now or datetime.now(UTC)).date()
    cutoff = today - timedelta(days=14)
    a = _fatigue_lines("Player A", ctx.player_a, today, cutoff)
    b = _fatigue_lines("Player B", ctx.player_b, today, cutoff)
    if not a and not b:
        return None
    header = "--- Tennis fatigue (computed from MatchStat recent-matches feed) ---"
    return "\n".join([header] + a + b)
