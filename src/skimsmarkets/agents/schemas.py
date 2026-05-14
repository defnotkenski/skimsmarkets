"""Cross-sport Pydantic schemas for the agent layer.

Per-sport report schemas (the typed reports each lens emits) live in
`agents/sports/<sport>/schemas.py` — e.g. `agents/sports/tennis/schemas.py`
holds the tennis lens reports.

Only sport-agnostic types live here:
- `LensNotebook` — fetcher Stage A output, free-form research notes +
  citations + computed numbers. No verdict.
- `Citation`, `ComputedNumber` — components of `LensNotebook`.
- `PlayerStatus` — availability primitive (name + team + status + impact
  note) currently used by `tennis.TennisConditionsContextReport.injury_concerns`.
  Lives here rather than under `agents/sports/tennis/` so a future sport
  can reuse the shape without taking a dependency on the tennis package.
- `EventPrediction` — director's structured-output schema.
- `MarketPrediction` — projection of an `EventPrediction` onto the
  predicted-winner's market.
- `DefensibilityAssessment`, `SlateDefensibilityJudgment` — slate judge
  output.

`LensNotebook.lens` is a plain `str` rather than a `Literal[...]` —
with per-sport lens names the union would grow per sport, and lens-name
validation already happens at runtime via
`agents/fetchers/base.py:assert_lens_match` against the LensSet's
declared lens names.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Confidence = Literal["low", "medium", "high"]


class Citation(BaseModel):
    """A single primary-source pull from web_search / x_search.

    `retrieved_value` is the concrete fact lifted from the page (a stat line, a
    line move, an injury status), not a paraphrase — the reasoner should be
    able to use it without re-fetching. URLs MUST be ones the fetcher actually
    retrieved; never fabricated.
    """

    url: str
    claim: str = Field(description="One-line summary of what this source supports.")
    retrieved_value: str | None = Field(
        default=None,
        description="Concrete value pulled (e.g. '28-6 record', 'Out — knee').",
    )


class TokenUsage(BaseModel):
    """Token-count record for one LLM call.

    Appended to a per-event sink by each LLM-call site (fetcher, reasoner,
    director, judge) so the pipeline can persist per-event token totals
    alongside timings. `stage` mirrors the same naming convention used by
    `RunResult.lens_timings` (`fetcher:<lens>`, `reasoner:<lens>`,
    `director`, `judge`) so retro queries can join tokens to timings on
    the stage label.

    Token fields are `int | None` because Gemini's older SDK occasionally
    omits the metadata; treat None as "unknown, do not aggregate."
    `provider` and `model` ride along so retro grading can A/B token cost
    per provider without joining against the run-level meta record.

    `cache_creation_input_tokens` / `cache_read_input_tokens` are
    Anthropic-only buckets — the SDK populates them when `cache_control`
    blocks are on the request. None for non-Anthropic providers (Grok,
    Gemini). The cost computation in `agents/pricing.py` treats both as
    0 when None.
    """

    model_config = ConfigDict(extra="ignore")

    stage: str = Field(
        description=(
            "Stage label matching `RunResult.lens_timings` keys: "
            "`fetcher:<lens>` / `reasoner:<lens>` / `director` / `judge`."
        ),
    )
    provider: str = Field(
        description="Provider name (e.g. 'grok', 'gemini', 'anthropic')."
    )
    model: str = Field(description="Model id (e.g. 'grok-4.3', 'claude-opus-4-7').")
    input_tokens: int | None = Field(
        default=None,
        description="Prompt-side tokens. None when the SDK omitted the metadata.",
    )
    output_tokens: int | None = Field(
        default=None,
        description=(
            "Completion-side tokens. Includes extended-thinking tokens for "
            "Anthropic — billed at the standard output rate. None when the "
            "SDK omitted the metadata."
        ),
    )
    cache_creation_input_tokens: int | None = Field(
        default=None,
        description=(
            "Anthropic-only: tokens billed at the cache-write premium "
            "(1.25× input for 5m TTL). None for non-Anthropic providers."
        ),
    )
    cache_read_input_tokens: int | None = Field(
        default=None,
        description=(
            "Anthropic-only: tokens billed at the cache-hit discount "
            "(0.1× input). None for non-Anthropic providers."
        ),
    )


class ComputedNumber(BaseModel):
    """A number derived via the fetcher's `code_execution` tool.

    Fetchers MUST surface every numeric derivation here (de-vig, log5, on/off
    splits, rating differentials) so the reasoner can use the value as-is
    rather than recomputing. `method` is a one-line note on the math; the
    reasoner trusts it without re-deriving.
    """

    label: str = Field(
        description="What was computed (e.g. 'devig_pinnacle_team_a', 'log5_team_a_baseline').",
    )
    value: float
    method: str = Field(
        description="One-line method note (e.g. 'log5 from team_a 0.62 vs team_b 0.55, neutral 0.50').",
    )


class LensNotebook(BaseModel):
    """Free-form research notebook emitted by a fetcher for one lens.

    The fetcher's job is evidence capture, not judgment — so this schema has
    NO probability, NO signed shift, NO directional verdict. Those land in the
    typed report a downstream Claude reasoner emits from this notebook plus
    the same event context the fetcher saw.

    `lens` is a free-form string set by the fetcher to the lens it was asked
    to run. Validation against the LensSet's declared lens names happens at
    runtime via `assert_lens_match` so prompt-mixup bugs fail loud at fetch
    time.

    `research_notes` is intentionally free-form prose (sectioned by the
    fetcher) so the provider's adaptive search loop ("found X, now look up Y")
    can capture whatever it stumbles on without the schema predicting
    structure upfront. Structured citation + computed-number lists ride
    alongside so URLs and numbers stay machine-extractable.

    `coverage` is the fetcher's self-assessment of how thin/rich the evidence
    is — the reasoner downgrades `confidence` to `low` when this is `thin`.
    """

    model_config = ConfigDict(extra="ignore")

    lens: str = Field(
        description=(
            "The lens this notebook was produced for. Validated against the "
            "active LensSet's declared lens names at runtime."
        ),
    )
    team_a_name: str = Field(
        description="Echoed verbatim from the event context's team_a_name.",
    )
    team_b_name: str = Field(
        description="Echoed verbatim from the event context's team_b_name.",
    )
    research_notes: str = Field(
        description=(
            "Free-form prose. Bullet what was found and what's missing. "
            "Do NOT include a probability, signed shift, or directional verdict."
        ),
    )
    citations: list[Citation] = Field(default_factory=list)
    computed_numbers: list[ComputedNumber] = Field(default_factory=list)
    coverage: Literal["thin", "adequate", "rich"] = Field(
        description="'thin' when primary sources were unavailable.",
    )


class PlayerStatus(BaseModel):
    """Cross-sport availability primitive used by
    `tennis.TennisConditionsContextReport.injury_concerns`. Lives here so
    a future sport's report schema can reuse the shape without taking a
    dependency on the tennis package.
    """

    name: str
    team: str
    status: str = Field(
        description="e.g. 'out', 'questionable', 'probable', 'suspended'."
    )
    impact_note: str = Field(description="How this affects the matchup.")


class RetractedShift(BaseModel):
    """One signed shift the director set aside during synthesis.

    Populated on `EventPrediction.retracted_shifts` when the director's
    final probability deviates from the literal stack math because a
    specific shift's magnitude wasn't supported by its lens's notebook.
    Retro grading aggregates these to find the most-retracted shifts —
    a direct signal that the offending reasoner is over-confident on
    that field.
    """

    lens_name: str = Field(
        description=(
            "The lens whose shift was retracted (e.g. "
            "'tennis_matchup_and_clutch'). Must match a key in "
            "`specialist_weights`."
        ),
    )
    shift_field: str = Field(
        description=(
            "Name of the signed-shift field that was retracted (e.g. "
            "'clutch_signed_shift', 'h2h_signed_shift'). Use the exact "
            "field name from the lens's report schema."
        ),
    )
    original_value: float = Field(
        description="The value the reasoner emitted (within the field's bound).",
    )
    applied_value: float = Field(
        description=(
            "The value you actually used when computing your final probability "
            "(typically 0.0 for a full retraction; non-zero for partial)."
        ),
    )
    reason: str = Field(
        description=(
            "ONE short sentence on why the lens's notebook didn't support "
            "the original value (e.g. 'matchup notebook flagged N=1 H2H "
            "and low confidence; original +0.06 too aggressive')."
        ),
    )


class SpecialistWeight(BaseModel):
    """One per-lens weight entry in the director's synthesis.

    Used as the element type of `EventPrediction.specialist_weights` so the
    field can carry a JSON-schema `minItems: 1` constraint that Anthropic's
    structured-output compiler enforces during generation. Previously this
    was a `dict[str, float]` with `min_length=1` (→ `minProperties: 1`),
    which Anthropic does NOT enforce during generation — the model would
    frequently emit `{}` and parse retries had to clean up. List-of-records
    sidesteps the unenforced-constraint trap.

    Downstream consumers (audit JSONL, judge prompt, retro readers) still
    see a `dict[str, float]` shape via projection at the EventPrediction
    boundary; this list type is only the wire format between the model
    and Pydantic validation.
    """

    model_config = ConfigDict(extra="ignore")

    lens_name: str = Field(
        description=(
            "Exact lens name from the active LensSet — e.g. for tennis: "
            "'tennis_form_and_surface' / 'tennis_matchup_and_clutch' / "
            "'tennis_conditions_and_context'. Must match the report block "
            "names verbatim."
        ),
    )
    weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Weight in [0, 1]. Sum across all entries should roughly equal 1.",
    )


class EventPrediction(BaseModel):
    """Event-level synthesis emitted by the director (LLM output).

    Confidence-ranker framing: the director names the likely winner and how sure
    it is. No buy/pass gate — ranking happens downstream by
    `predicted_winner_probability`.
    """

    event_id: str = Field(
        description="Polymarket event id this prediction applies to.",
    )
    predicted_winner: str = Field(
        description="Exact yes_sub_title of the team expected to win. Must match one of the event's markets.",
    )
    predicted_winner_probability: float = Field(
        ge=0.0, le=1.0,
        description="Probability the predicted winner actually wins, 0-1.",
    )
    confidence: Confidence = Field(
        description=(
            "Robustness of the pick to real-world contingencies — count how many "
            "independent things would have to break against the pick (in the WORLD, "
            "not in the model) for it to lose. NOT a measure of how lopsided the "
            "matchup is. high = multiple independent contingencies would have to "
            "stack against the pick (e.g. ATP top-100 vs unranked qualifier in R32 — "
            "would need late withdrawal AND adverse weather AND in-match upset run); "
            "medium = the pick survives the most common single contingency but a "
            "stacked pair would break it; low = a single common contingency flips "
            "the pick (one starter scratched, one cold shooting half, one early red "
            "card). The point is fragility, not magnitude — a 52-48 call where the "
            "favorite enters fully fit on a neutral surface and no obvious single "
            "contingency could flip it IS still high confidence."
        ),
    )
    # `specialist_weights` is a `list[SpecialistWeight]` rather than a
    # `dict[str, float]` so the `min_length=1` constraint generates JSON
    # schema `minItems: 1` (which Anthropic's structured-output compiler
    # DOES enforce during generation) instead of `minProperties: 1`
    # (which it does NOT enforce — verified against their docs). The
    # model is structurally prevented from emitting an empty list, which
    # closes out the long-running empty-`{}` + trailing-comma failure
    # cluster that the per-event retry loop had been masking. Downstream
    # consumers project this back to a `dict[str, float]` at the
    # EventPrediction → MarketPrediction boundary; the JSONL audit shape
    # and retro readers continue to see the dict form.
    specialist_weights: list[SpecialistWeight] = Field(
        min_length=1,
        description=(
            "One entry per lens you weighted in the synthesis. Each entry has "
            "`lens_name` (exact name from the active LensSet — for tennis: "
            "'tennis_form_and_surface' / 'tennis_matchup_and_clutch' / "
            "'tennis_conditions_and_context') and `weight` in [0, 1]. Weights "
            "across entries should roughly sum to 1. REQUIRED: must contain at "
            "least one entry — the empty list is rejected by the schema's "
            "minItems constraint."
        ),
    )
    headline: str = Field(
        description=(
            "ONE sentence, max ~20 words, plain English. The single most decisive "
            "reason this side wins — readable at a glance with no jargon. Example: "
            "'Lakers win behind a fully-healthy LeBron and a 7-game home win streak.' "
            "This is what shows up in the at-a-glance leaderboard; the long-form "
            "synthesis lives in `reasoning`."
        ),
    )
    reasoning: str = Field(
        description="3-6 sentences explaining the synthesis and how specialists were weighted.",
    )
    disagreements_flagged: list[str] = Field(
        default_factory=list,
        description=(
            "One short string per material disagreement that shaped the "
            "synthesis. Cover, when applicable: (1) directional conflict "
            "between specialists (one shift favors team_a, another team_b); "
            "(2) deviation between your final probability and the literal "
            "stack math when you retracted a shift (>5pp); (3) deviation "
            "between your final probability and a deterministic prior "
            "(market, sim, or GBT) by >10pp. Each entry should name what "
            "diverged and how you resolved it. Empty only when specialists "
            "agree directionally AND your final tracks the stack AND it "
            "tracks the deterministic priors."
        ),
    )
    uw_flow_note: str | None = Field(
        default=None,
        description=(
            "2-4 sentence observation of the Unusual Whales flow signals when "
            "a 'Flow signals (Unusual Whales...)' block was present in the event "
            "context. Cover: which tags fired and their magnitude, the direction "
            "of recent smart-money / contrarian-whale trades (buyers hitting ask "
            "vs sellers hitting bid), notable insider positions, MCI value/delta "
            "if meaningful, and whether flow agreed with or diverged from the "
            "sportsbook consensus. Null when UW had no coverage for this event."
        ),
    )
    retracted_shifts: list[RetractedShift] = Field(
        default_factory=list,
        description=(
            "Per-event audit log of any signed shifts you set aside while "
            "synthesizing — i.e. shifts whose magnitude you concluded were "
            "unsupported by the lens's notebook evidence after re-reading. "
            "ONLY populate when you genuinely retracted a shift; do NOT use "
            "this as a generic 'I down-weighted this lens' field (that "
            "belongs in `specialist_weights`). One entry per retracted shift. "
            "Empty when you accepted the literal stack math. Drives retro "
            "grading of which shifts the director most often retracts — a "
            "signal for tightening the offending reasoner's calibration."
        ),
    )


class MarketPrediction(BaseModel):
    """Per-market projection of an EventPrediction (built deterministically).

    Attaches the winning side's Polymarket slug + implied probability alongside
    the director's prediction so downstream sizing and reporting have a single
    self-contained record. Carries `sport_type` and `lens_set_name` for JSONL
    grouping (`jq 'select(.sport_type=="tennis")'`) without joining against a
    sidecar.
    """

    market_slug: str = Field(
        description="Polymarket slug of the market that represents the predicted winner's side.",
    )
    event_id: str
    event_title: str | None = None
    sport_type: str | None = Field(
        default=None,
        description=(
            "Canonical `event.sport_type` (gamma-tag-derived) at time of "
            "prediction. Used by JSONL retrospective analysis and by reporting "
            "to group predictions by sport. None for events without a "
            "recognized sport tag."
        ),
    )
    lens_set_name: str | None = Field(
        default=None,
        description=(
            "Name of the LensSet this prediction was synthesized through "
            "(e.g. 'tennis'). Matches `sport_type` for the current registry."
        ),
    )
    predicted_winner: str
    predicted_yes_probability: float = Field(ge=0.0, le=1.0)
    polymarket_implied_probability: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Polymarket's implied probability for the predicted winner side (midpoint of yes bid/ask).",
    )
    confidence: Confidence
    headline: str = Field(
        description="One-sentence glanceable summary — projected from EventPrediction.",
    )
    reasoning: str = Field(
        description="3-6 sentences explaining the synthesis and how specialists were weighted.",
    )
    specialist_weights: dict[str, float] = Field(
        description="Per-specialist weight in [0,1]; should roughly sum to 1.",
    )
    disagreements_flagged: list[str] = Field(
        default_factory=list,
        description=(
            "Material disagreements the director resolved during synthesis. "
            "Projected verbatim from the EventPrediction; see that field's "
            "description for the full scope (lens directional conflicts, "
            "stack-vs-final retractions, prior-vs-final deviations)."
        ),
    )
    uw_flow_note: str | None = None
    retracted_shifts: list[RetractedShift] = Field(
        default_factory=list,
        description=(
            "Projected verbatim from EventPrediction.retracted_shifts — "
            "audit log of signed shifts the director set aside during "
            "synthesis. See that field's description for usage."
        ),
    )


class DefensibilityAssessment(BaseModel):
    """Per-event judgment from the slate-level judge.

    Emitted by `judge_slate` (one LLM call per run that reads ALL events'
    director outputs). Replaces `predicted_yes_probability` as the
    leaderboard's primary sort key. Confidence-ranker framing: this scores
    **case defensibility**, not edge or expected value — a high score
    means the director's reasoning is coherent, the lenses agreed, and the
    UW flow (when present) aligns with the prediction. The field name
    intentionally measures the *absence* of risk (defensibility) rather
    than the presence of it, so "higher = better" matches the leaderboard's
    descending-sort direction without inversion.
    """

    event_id: str = Field(
        description="Polymarket event id this assessment applies to. "
        "Must match a MarketPrediction.event_id from the same run.",
    )
    defensibility_score: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Defensibility score in [0,1]. 1.0 = strongest case "
            "(coherent reasoning, lens alignment, UW flow alignment); "
            "0.0 = weakest. Sort direction matches predicted_yes_probability "
            "(descending = better) so the leaderboard mental model is "
            "preserved."
        ),
    )
    defensibility_rationale: str = Field(
        description="≤2 sentences explaining the score in plain English.",
    )
    defensibility_flags: list[str] = Field(
        default_factory=list,
        description=(
            "Up to 3 short snake_case slugs naming the load-bearing "
            "weaknesses. Prefer the JUDGE_SYSTEM vocabulary "
            "('thin_reasoning', 'lens_disagreement', 'uw_contra', "
            "'concentrated_weights', 'unexplained_gap', "
            "'low_confidence_tier', 'live_volatility') and only coin a "
            "new slug when none fits. Empty when the case is clean."
        ),
    )


class SlateDefensibilityJudgment(BaseModel):
    """Wrapper for the structured-output parse of the slate-level judge.

    One per `judge_slate` call. `assessments` should contain one
    `DefensibilityAssessment` per event in the input slate, but downstream
    tolerates partial coverage (un-scored events sort to the bottom of the
    leaderboard via a sentinel score).
    """

    assessments: list[DefensibilityAssessment]
