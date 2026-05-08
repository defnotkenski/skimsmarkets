"""Pydantic shapes for tennis player-stats enrichment.

Same posture as `unusual_whales/models.py:UnusualWhalesContext`:
- Structurally compact — only fields the `tennis_form_and_surface`
  specialist can actually consume in a prompt; we don't mirror the
  vendor's full payload.
- `has_actionable_signal()` lets the renderer skip cleanly when the
  fetched context is too thin to be worth the prompt-token cost.
- All numeric fields are `| None` so a partial fetch (vendor missing a
  player record) is representable rather than triggering validation
  errors that would drop the whole context.

Field naming uses no source prefix because there is one tennis-stats
source per event — unlike the polymarket models where `gamma_*` /
`clob_*` distinguish two providers feeding the same record.
"""

from __future__ import annotations

from datetime import date as _date_t
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Field-name aliasing: several models below have a `date` field. The
# bare `date` type from `datetime` would shadow the field annotation at
# class-body eval time and break Pydantic's type-hint resolution. Import
# the type as `_date_t` and use it everywhere a `datetime.date`
# annotation is needed.


def _coerce_date(v: Any) -> Any:
    """Tolerant date parser for vendor payloads.

    Tennis APIs ship dates in mixed shapes (ISO strings, epoch seconds,
    or already-parsed `date` objects). Mirrors the `_coerce_dt` helper
    in `unusual_whales/models.py` but for `date` rather than `datetime`.
    """
    if v is None or v == "":
        return None
    if isinstance(v, _date_t) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        # Accept ISO date or ISO datetime; trim time portion for the
        # latter so callers can pass either shape.
        try:
            return _date_t.fromisoformat(s[:10])
        except ValueError:
            return None
    return None


class TennisRecentMatch(BaseModel):
    """One row in `TennisPlayerStats.recent_matches`.

    The compact `last_10_form` string ("WWLWWLWWWL") tells the reasoner
    *whether* the player won, but nothing about the quality of those wins.
    A 3-row digest of recent matches with opponent / score / round /
    surface / tournament tier closes that gap — straight-sets wins at a
    Masters event read very differently from grinding three-set wins at
    a 250-level tournament. Three rows is the budget; older matches stay
    folded into `last_10_form`.
    """

    model_config = ConfigDict(extra="ignore")

    date: _date_t | None = None
    # Opponent's name as the vendor ships it. Diacritics preserved — the
    # reasoner doesn't need normalized forms for display.
    opponent_name: str
    won: bool
    # Score line as the vendor ships it (e.g. "6-4 6-2", "7-6(5) 6-3 4-6 6-2",
    # "5-0 ret."). Distinguishes a straight-sets win from a five-setter
    # the same player won.
    result: str | None = None
    # Surface key collapsed to "hard" / "clay" / "grass" / "carpet".
    surface: str | None = None
    # Round name as the vendor ships it ("Final", "1/2" = SF, "1/4" = QF).
    round: str | None = None
    tournament_name: str | None = None
    # Tier label derived from tournament.rankId — distinguishes "won 6
    # straight at a Challenger" from "won 6 straight at a Masters."
    tournament_tier: str | None = None

    @field_validator("date", mode="before")
    @classmethod
    def _d(cls, v: Any) -> Any:
        return _coerce_date(v)


class TennisH2HMeeting(BaseModel):
    """One row in `TennisHeadToHead.recent_meetings`.

    Replaces the previous flat `last_meeting_*` block with a list so the
    reasoner sees the matchup *trajectory* across the last few meetings,
    not just the most recent. Three meetings is enough to spot
    "Sinner-Alcaraz has tilted from 2-3 to 6-3 across the last year"
    without bloating the prompt.
    """

    model_config = ConfigDict(extra="ignore")

    date: _date_t | None = None
    winner_name: str | None = None
    surface: str | None = None
    round: str | None = None
    # Score line, same shape as TennisRecentMatch.result.
    result: str | None = None
    tournament_name: str | None = None
    tournament_tier: str | None = None

    @field_validator("date", mode="before")
    @classmethod
    def _d(cls, v: Any) -> Any:
        return _coerce_date(v)


class TennisInMatchupStats(BaseModel):
    """Per-player stats conditioned on the matchup itself.

    Career averages (on `TennisPlayerStats`) tell the reasoner how a
    player performs against the field. These tell it how the player
    performs specifically against THIS opponent — a player who's 47% on
    BP-conversion overall might be 35% specifically against this opponent.
    Sourced from a single `/h2h/stats` call that ships both players'
    matchup-conditioned aggregates side by side.
    """

    model_config = ConfigDict(extra="ignore")

    # `(wins, total)` so the reasoner sees sample size — a 1-1 deciding-set
    # record carries less weight than 5-2.
    decider_record: tuple[int, int] | None = None
    tiebreak_record: tuple[int, int] | None = None
    # Format-conditioned record. bo3 = standard ATP/WTA tour (250s, 500s,
    # Masters, regular WTA). bo5 = men's slams. The split matters: a
    # matchup that's 5-11 in bo3 but 2-6 in bo5 reads as "deeply lopsided
    # at slams specifically" rather than just "lopsided overall."
    bo3_record: tuple[int, int] | None = None
    bo5_record: tuple[int, int] | None = None
    # Comeback rate when losing the first set (existing field, moved into
    # this nested block).
    first_set_lost_match_won_pct: float | None = None
    # Closeout rate when winning the first set — the natural complement.
    # Together with `first_set_lost_match_won_pct` they characterise how
    # this player handles set 1 outcomes against this specific opponent.
    first_set_won_match_won_pct: float | None = None
    # Career-aggregate first-serve points won and break-point conversion
    # pct, computed ONLY across this matchup's prior meetings. Distinct
    # from the same field on `TennisPlayerStats` (which is across all
    # career opponents). Stored as ratios in [0, 1] for consistency.
    first_serve_win_pct: float | None = None
    break_point_convert_pct: float | None = None


class TennisPlayerStats(BaseModel):
    """Per-player snapshot the `tennis_form_and_surface` specialist
    consumes verbatim.

    Every numeric field is optional because vendors differ in what they
    expose: rapid-API-style indexes have ranking + form but no surface
    splits, while tennisabstract-style sources have rich surface splits
    but no live ranking points. The reasoner is told to use whatever is
    populated and ignore absent fields.

    Career serve / return / break-point percentages come from a single
    extra vendor call (`/player/match-stats/{id}`). They're CAREER
    aggregates — not surface-conditioned, not last-N-match — but the
    relative ordering between two players (e.g. 70% vs 62% first-serve
    win) is itself a load-bearing matchup signal even at the career
    timeframe. Surface-conditioned form lives in `surface_win_loss`.
    """

    model_config = ConfigDict(extra="ignore")

    # Echoed verbatim from `_parse_h2h_question` so the reasoner can match
    # the player back to `team_a_name` / `team_b_name` in the event
    # context without a second normalization pass.
    name: str
    # Vendor-specific canonical id, kept for retro grading ("did the API
    # actually find this player?") rather than for the prompt — providers
    # often need the id internally to fetch H2H. None when the lookup
    # was fuzzy-name only.
    api_player_id: str | None = None
    rank_singles: int | None = None
    rank_points: int | None = None
    # Career-high singles ranking. Useful context: a current rank=80
    # player with best_rank=4 (descending veteran or comeback) reads
    # very differently from a rank=80 player with best_rank=78 (stable
    # journeyman). Free to populate — comes on the same profile call as
    # current rank.
    best_rank_singles: int | None = None
    # Bio fields lifted from `profile.information`. Free — same call we
    # already make for `form` and `bestRank`. Age is the load-bearing
    # one (a 21yo vs a 38yo reads differently from rank delta alone);
    # `plays` ("Right-Handed, Two-Handed Backhand") matters for stylistic
    # matchups (LH/RH split, one- vs two-handed BH on clay).
    age_years: int | None = None
    plays: str | None = None
    # `(wins, losses)` over a vendor-defined window — typically YTD or
    # trailing 52 weeks. Provider docs the window in its rendering tail.
    ytd_win_loss: tuple[int, int] | None = None
    # Surface keys: `"hard"`, `"clay"`, `"grass"`, `"carpet"`. Values are
    # `(wins, losses)`. Surface-conditioned form is more predictive than
    # YTD aggregates per `_FETCHER_HINT_FORM_AND_SURFACE` in
    # `agents/sports/tennis/lens_set.py`.
    surface_win_loss: dict[str, tuple[int, int]] | None = None
    # Compact "WWLWWLWWWL" string — last N matches in chronological order
    # (oldest → newest). N is vendor-defined; we don't try to enforce.
    last_10_form: str | None = None
    # Detailed digest of the most recent N matches with opp / score /
    # round / surface / tier. Complements `last_10_form` (which is just
    # the W/L pattern) by carrying *quality of opposition* and *match
    # tightness*. Newest first; capped at 3 rows in the renderer.
    recent_matches: list[TennisRecentMatch] | None = None
    last_match_date: _date_t | None = None
    # Career serve metrics. All four percentages live in [0.0, 1.0] —
    # the renderer multiplies by 100 for display. The vendor ships raw
    # counters (numerator + denominator) which the provider divides
    # before persisting; we keep ratios rather than counts so the prompt
    # is human-readable without further math.
    first_serve_in_pct: float | None = None
    first_serve_win_pct: float | None = None
    second_serve_win_pct: float | None = None
    # Career return-side percentages. Computed from `match-stats.rtnStats`
    # via 1 − (opponent's serve-win on this player), so this is "career
    # first/second-serve return-points-won %" — the canonical complement
    # to the serve-side trio above. Together with `break_point_convert_pct`
    # (which is BPs *converted given a chance*), these give the full
    # career return-game profile.
    first_serve_return_win_pct: float | None = None
    second_serve_return_win_pct: float | None = None
    # Break-point save % (when serving) and break-point conversion %
    # (when returning). The two together are the canonical "is this
    # player clutch?" pair the tennis sport hint already flags as
    # decisive in tight matches.
    break_point_save_pct: float | None = None
    break_point_convert_pct: float | None = None
    # Current-year W/L vs elite competition and at the biggest events.
    # Sourced from `/player/perf-breakdown` (the vendor's year × tier
    # matrix). These rows answer "does this player show up against good
    # opponents and at the big stages?". `top5` distinguishes "elite-tier
    # slayer" from "merely beats top-10s" (a third of Sinner's top-10
    # wins YTD are vs ranks 6-10 specifically). Other rows (`top1`,
    # `top20`, `top50`, futures, challengers) are available on the same
    # response but compete for prompt space; we pick the ones with the
    # cleanest signal-per-token.
    record_vs_top_5: tuple[int, int] | None = None
    record_vs_top_10: tuple[int, int] | None = None
    record_at_grand_slam: tuple[int, int] | None = None
    record_at_masters: tuple[int, int] | None = None
    # Career titles per tier. Sourced from `/player/titles`; one extra
    # HTTP call per player but distinct signal — career achievement
    # baseline that rank can't capture (a 28yo with 0 slam titles + 15
    # mainTour reads differently from a 22yo with 4 slams). Keys:
    # "grand_slam", "masters", "main_tour", "tour_finals". Lower tiers
    # (futures, challengers, team_cup) skipped — not load-bearing for
    # tour-level Polymarket markets.
    career_titles: dict[str, int] | None = None
    # Career-aggregate clutch records, derived by parsing past-matches
    # score strings (provider.parse_score_details). Sample-size visible
    # so the lens render can show e.g. `tiebreaks=42-31` — Claude
    # calibrates clutch shifts around denominators, not raw rates.
    # Distinct from `TennisHeadToHead.{a,b}_in_matchup` which is the
    # OPPONENT-conditioned slice; these are career-aggregate, span the
    # last 50 past-matches the provider pulls. `(wins, total)` pairs.
    # All None when past-matches lacks score strings (older rows
    # occasionally drop them) — the renderer suppresses absent lines.
    career_tiebreak_record: tuple[int, int] | None = None
    career_decider_record: tuple[int, int] | None = None
    # `comeback`: matches won given set 1 lost. Denominator counts only
    # matches where this player lost set 1 — an absent denominator is
    # itself signal (player rarely loses set 1) but we surface the
    # ratio anyway since a pure denominator delta is hard to read.
    career_comeback_record: tuple[int, int] | None = None
    # `close_match`: final-set margin ≤2 OR final set was a tiebreak.
    # Captures matches decided by small-edge skill rather than gulfs in
    # quality.
    career_close_match_record: tuple[int, int] | None = None
    # Recency-windowed BP-save (last 180 days). Complements the career
    # `break_point_save_pct` (no time bound) by surfacing form arcs:
    # 75% recent vs 65% career flags an upswing the GBT's career
    # feature would smooth over. Computed from per-match BP counters
    # in past-matches.stats — same data the existing career rate uses,
    # filtered by date.
    break_point_save_pct_180d: float | None = None

    @field_validator("last_match_date", mode="before")
    @classmethod
    def _d(cls, v: Any) -> Any:
        return _coerce_date(v)


class PerMatchStats(BaseModel):
    """Single-match box score for one player.

    Sourced from MatchStat's `past-matches?include=stat,tournament,round`
    response (the `stats.player1` / `stats.player2` blocks). Fetched by
    the retro/self-improvement layer AFTER an event resolves, to compare
    what the player ACTUALLY did that day against the career-baseline
    `TennisPlayerStats` snapshot taken at prediction time. Pre-match
    `tennis_stats` carries career means; this carries the single-match
    realisation. Divergence between the two answers "did the player play
    to baseline or did they over/underperform, and did our lenses see
    any reason to expect that?"

    Vendor ships fraction pairs (e.g. `winningOnFirstServe=47`,
    `winningOnFirstServeOf=65`); we divide upfront and store ratios in
    [0,1] for direct comparison with `TennisPlayerStats` fields. None
    when the denominator was zero or null on the vendor row — common on
    live-suspended matches and walkovers, never on completed matches in
    practice.
    """

    model_config = ConfigDict(extra="ignore")

    # Per-player percentages — same shape as the corresponding career
    # baseline fields on `TennisPlayerStats`. Direct subtraction yields
    # divergence in points (e.g. baseline 0.65 - actual 0.51 = 0.14
    # underperformance on first-serve win %).
    first_serve_in_pct: float | None = None
    first_serve_win_pct: float | None = None
    second_serve_win_pct: float | None = None
    break_point_convert_pct: float | None = None
    # Counts the vendor exposes that don't have a per-match denominator
    # to convert into a ratio. Useful for the retro feature row even at
    # raw count granularity (a 13-ace match vs 1-ace match is signal
    # regardless of total points played). Aces / DFs are sometimes null
    # on the vendor row for old matches; total points is always present
    # on completed matches.
    aces: int | None = None
    double_faults: int | None = None
    total_points_won: int | None = None


class TennisHeadToHead(BaseModel):
    """Head-to-head history between the two players in this match.

    The matchup-specific clutch records (decider, tiebreak, comeback,
    bo3/bo5, matchup serve/BP) live on the per-player `a_in_matchup` /
    `b_in_matchup` blocks rather than as flat `_a/_b` fields, because
    the field count grew enough that flat naming was getting noisy. The
    reasoner is told to weight these matchup-conditioned numbers above
    career averages when both are present.
    """

    model_config = ConfigDict(extra="ignore")

    a_wins: int = 0
    b_wins: int = 0
    # Per-surface H2H counts: surface key → `(a_wins, b_wins)`. The vendor
    # ships h2h/info as a per-court list anyway — we used to sum across
    # surfaces, but for surface-conditioned matchups (e.g. clay slate
    # match between players whose H2H is wildly different on clay vs hard)
    # the per-surface breakdown is the load-bearing read. Surface keys
    # match `TennisPlayerStats.surface_win_loss` ("hard" / "clay" /
    # "grass" / "carpet").
    surface_h2h: dict[str, tuple[int, int]] | None = None
    # Most recent meetings, newest first. Capped at 3 in the fetcher
    # (pageSize=3 on /h2h/matches). Replaces the previous flat
    # `last_meeting_*` fields with a list so the matchup arc — has it
    # tilted recently? — is visible without flattening to "last winner."
    recent_meetings: list[TennisH2HMeeting] | None = None
    # Per-player matchup-conditioned aggregates. None when the matchup
    # has no prior meetings or /h2h/stats was unavailable.
    a_in_matchup: TennisInMatchupStats | None = None
    b_in_matchup: TennisInMatchupStats | None = None


class TennisStatsContext(BaseModel):
    """Compact per-match tennis stats blob attached to a `PolymarketEvent`.

    Built by the active `TennisStatsProvider` for events where
    `tennis_match_identity` returns a non-None identity. Lives on
    `event.tennis_stats` (None when the gate failed, the vendor is down,
    or the API returned no usable record).
    """

    model_config = ConfigDict(extra="ignore")

    # Vendor name (e.g. `"stub"`, `"rapid_api_tennis"`). Persisted to the
    # JSONL row so retro grading can group hit-rate by vendor the same
    # way `fetcher_provider` does for the search providers.
    provider: str
    fetched_at: datetime
    # Tournament + surface come from the vendor when available; fall back
    # to None and let the reasoner read the surface out of the event
    # context's existing tennis sport hint when the vendor doesn't say.
    surface: str | None = None
    tournament: str | None = None
    player_a: TennisPlayerStats
    player_b: TennisPlayerStats
    head_to_head: TennisHeadToHead | None = None

    def has_actionable_signal(self) -> bool:
        """True iff the fetched context has anything worth prompting on.

        Drops contexts that came back populated-but-empty — every numeric
        field None, no H2H, just two echoed names. Letting those through
        would cost prompt tokens for no information.

        Threshold: at least one of (a) any ranking populated on either
        player, (b) any surface or YTD split, (c) recent matches detail,
        (d) any non-zero H2H. UW's `has_actionable_signal` filter applies
        the same posture for flow signals.
        """
        for player in (self.player_a, self.player_b):
            if player.rank_singles is not None:
                return True
            if player.ytd_win_loss is not None:
                return True
            if player.surface_win_loss:
                return True
            if player.last_10_form:
                return True
            if player.recent_matches:
                return True
        h2h = self.head_to_head
        if h2h is not None and (h2h.a_wins > 0 or h2h.b_wins > 0):
            return True
        return False


class TennisGbtFeatureContribution(BaseModel):
    """One feature-importance row for the director-facing top-N display.

    Sourced from catboost's `get_feature_importance(type=PredictionValuesChange)`
    re-evaluated on the slate row (so the importance is local to THIS
    prediction, not the global model). The director uses the top 3-5 to
    explain "what the model leaned on for this match" — e.g. a +0.12
    contribution from `surface_first_serve_win_pct_diff` says the model's
    surface-conditioned serve read tilted toward the anchor.
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    contribution: float = Field(
        description=(
            "Signed contribution to the anchor-side log-odds. Positive "
            "pushes toward the anchor (lower MatchStat id), negative "
            "toward the opponent."
        )
    )


class TennisGbtContext(BaseModel):
    """Per-event gradient-boosted-tree prediction.

    Director-only — same architectural posture as `TennisSimulationContext`.
    Computed deterministically from the career box-score primitives on
    `TennisStatsContext` at pipeline time in `enrich_tennis_gbt`. The
    director uses this as a THIRD deterministic prior alongside Polymarket
    bid/ask and the iid Monte Carlo sim; lenses don't see it.

    Where the sim asks "what's the long-run rate from career averages?"
    the GBT asks "given the full feature vector — career rates, surface
    conditioning, recent form, age — what does history say P(team_a
    wins) is?" The two diverge usefully when surface or form actually
    matter, and the director's anti-anchoring instructions explain
    material divergence in `reasoning`.
    """

    model_config = ConfigDict(extra="ignore")

    # Versioned tag so a future v2 (e.g. recency-weighted, surface
    # submodels) can co-exist on JSONL rows without ambiguity.
    provider: Literal["gbt_spike_v1"] = "gbt_spike_v1"
    computed_at: datetime
    # Model artifact identity. The training script stamps the hash of
    # the artifact bytes here so retro grading can detect retraining
    # boundaries on the time axis.
    model_version: str
    p_team_a_wins: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Model probability that team_a wins, mapped from the "
            "anchor-relative prediction by the slate-time predictor."
        ),
    )
    # Per-side prior-match counts. The cold-start gate drops events
    # where either side has < 20 priors; persisting the counts lets
    # retro grading see how thin the model's evidence base was for
    # marginal cases (e.g. comeback player with a 12-match window).
    n_prior_matches_a: int = Field(ge=0)
    n_prior_matches_b: int = Field(ge=0)
    # Top-N feature contributions by absolute value. Ordered descending
    # by |contribution|. Capped at 5 — the director doesn't need a full
    # SHAP plot, just the model's top reads.
    top_features: list[TennisGbtFeatureContribution] = Field(
        default_factory=list
    )
    # One-line description of model scope, mirrors the sim's
    # `assumptions` field. Surfaces in the rendered block so the
    # director treats the GBT as a finite-window historical prior, not
    # a contextual probability.
    assumptions: str


class TennisSimulationContext(BaseModel):
    """Per-event Monte Carlo career-baseline simulation result.

    Director-only — same architectural posture as `UnusualWhalesContext`.
    Computed deterministically (fixed seed per event) from the career
    serve/return primitives on `TennisStatsContext` at pipeline time
    in `enrich_tennis_simulation`. The director uses this as a SECOND
    deterministic prior alongside Polymarket bid/ask; lenses don't see
    it (a long-run baseline shouldn't be second-guessed at the lens
    layer — that's the director's synthesis job).

    Intentionally limited to iid + career-baseline so it doesn't fight
    the lenses' jobs — surface, form, conditions, and H2H remain the
    lens layer's responsibility. The sim is "the long-run prior," the
    lenses produce "the contextual delta," and the director synthesizes
    both.
    """

    model_config = ConfigDict(extra="ignore")

    # Versioned provider tag so a future v2 (e.g. Klaassen-Magnus tour-
    # adjusted, or surface-conditioned) can co-exist on JSONL rows
    # without an ambiguous "monte_carlo" string. v1 = symmetric-average
    # point-win formula, no surface/form adjustments.
    provider: Literal["monte_carlo_v1"] = "monte_carlo_v1"
    computed_at: datetime
    p_team_a_wins: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of n_sims where team_a won the match. Career-baseline "
            "iid prior; ignores surface, form, conditions, H2H."
        ),
    )
    # 95% sampling-uncertainty CI from the Wilson interval. Captures
    # noise in the Monte Carlo estimator ONLY — does NOT capture model
    # uncertainty (the iid assumption being wrong, career != current
    # form, etc.). Director should treat the CI as a rough sampling
    # band, not a true uncertainty quantification.
    ci_low: float = Field(ge=0.0, le=1.0)
    ci_high: float = Field(ge=0.0, le=1.0)
    n_sims: int = Field(ge=1)
    best_of: Literal[3, 5]
    point_win_pct_a_serving: float = Field(
        ge=0.0,
        le=1.0,
        description="P(team_a wins a point on team_a's serve).",
    )
    point_win_pct_b_serving: float = Field(
        ge=0.0,
        le=1.0,
        description="P(team_b wins a point on team_b's serve).",
    )
    # One-line plain-English description of what this number does and
    # doesn't account for. Surfaces in the rendered block so the
    # director can't confuse the sim with a contextual probability.
    assumptions: str
