"""Sport-keyed lens-set primitives — `LensSpec` (one lens) and `LensSet`
(one sport's ordered tuple of lenses + per-sport director tail).

Each sport declares its own ordered list of lenses — tennis ships
`tennis_form_and_surface` / `tennis_matchup_and_clutch` /
`tennis_conditions_and_context` with sport-specific prompts and report
schemas. The dispatch primitives live here so the registry
(`agents/sports/__init__.py`) and the per-sport packages
(`agents/sports/<sport>/`) share one shape.

A `LensSpec` is one lens's full contract:

- `name`: stable identifier (e.g. `"tennis_form_and_surface"`). Surfaces
  in JSONL row keys, `specialist_weights` keys, and lens-discriminator
  validation. Names are unique across the entire registry — providers
  pre-build per-(sport, lens) prompts keyed on a flat dict, and lens
  names not colliding lets that key shape stay simple.
- `fetcher_system_builder(tools_section, notebook_tail) -> str`: builds
  the cached fetcher system prompt from a provider's per-lens tool prose
  and shared notebook tail.
- `reasoner_system`: cached system prompt for the Claude reasoner.
- `report_schema`: the Pydantic class the reasoner returns.
  Cross-sport pipeline plumbing carries `dict[str, BaseModel]`; the
  per-sport director path knows the concrete types.
- `render_extras(event) -> str | None`: per-lens user-message append.
  Currently the structured tennis-stats block feeds only
  `tennis_form_and_surface`; specs that don't need a per-lens append
  leave this `None`.
- `fetcher_sport_hint` / `reasoner_sport_hint`: static, per-lens
  guidance strings that ride on the per-event user message, never the
  cached system block, so slate-wide cache hits on the system stay warm.

A `LensSet` bundles the lenses for one sport plus the director's
sport-specific synthesis tail. The director's cross-sport content lives
in `agents/sports/_director_shared.py`'s `DIRECTOR_SHARED_PREAMBLE` and
is cached as a separate ephemeral block above the sport-specific tail —
two breakpoints per director call, well within the Anthropic 4-cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel

from skimsmarkets.polymarket.models import PolymarketEvent


@dataclass(frozen=True)
class LensSpec:
    """One lens's full contract within a `LensSet`.

    Frozen because the lens-set registry is built once at import time and
    re-instantiating per event would defeat the prompt-cache discipline
    (every cached prompt must be a stable Python object across calls).

    A spec is either LLM-mode (fetcher_system_builder + reasoner_system
    set) or algorithmic-mode (`compute` set). Validated in __post_init__.
    Algorithmic specs skip the Stage A/B chain entirely — the orchestrator
    calls `compute(event)` and synthesizes a placeholder LensNotebook so
    the JSONL persistence shape stays uniform across both lens kinds.
    """

    name: str
    fetcher_system_builder: Callable[[str, str], str] | None
    reasoner_system: str | None
    report_schema: type[BaseModel]
    render_extras: Callable[[PolymarketEvent], str | None] | None = None
    fetcher_sport_hint: str | None = None
    reasoner_sport_hint: str | None = None
    compute: Callable[[PolymarketEvent], BaseModel | None] | None = None

    def __post_init__(self) -> None:
        if self.compute is not None:
            if self.fetcher_system_builder is not None or self.reasoner_system is not None:
                raise ValueError(
                    f"LensSpec {self.name!r}: algorithmic specs (compute set) "
                    f"must leave fetcher_system_builder + reasoner_system as None."
                )
        else:
            if self.fetcher_system_builder is None or self.reasoner_system is None:
                raise ValueError(
                    f"LensSpec {self.name!r}: LLM specs need both "
                    f"fetcher_system_builder and reasoner_system."
                )

    def render_fetcher_hint(self) -> str | None:
        """Format the fetcher sport hint for appending to the user message,
        or `None` when no hint applies. Header mirrors the legacy
        `render_sport_hint` shape so existing JSONL grep / human eyeballs
        recognize the block.
        """
        if self.fetcher_sport_hint is None:
            return None
        return (
            f"--- Sport-specific focus (lens={self.name}) ---\n"
            f"{self.fetcher_sport_hint}"
        )

    def render_reasoner_hint(self) -> str | None:
        """Format the reasoner calibration hint for appending to the user
        message, or `None` when no hint applies.
        """
        if self.reasoner_sport_hint is None:
            return None
        return (
            f"--- Sport-specific calibration (lens={self.name}) ---\n"
            f"{self.reasoner_sport_hint}"
        )


@dataclass(frozen=True)
class LensSet:
    """One sport's ordered lens set + sport-specific director tail.

    `sport` is the canonical `event.sport_type` string the registry keys
    on (`"tennis"`, etc.). Events whose `sport_type` has no matching
    entry drop with `ErrorRecord(stage="lens_dispatch")`.

    `director_system_tail` is the sport-specific director synthesis prompt.
    The director sends `[DIRECTOR_SHARED_PREAMBLE, director_system_tail]`
    as two cached system blocks per call so cross-sport content (market
    anchoring, calibration discipline, UW flow framing, headline format)
    is reused across sports without duplication.

    `judge_system` is reserved for forward compatibility — a sport-specific
    judge prompt could be useful later but is `None` initially; the global
    `JUDGE_SYSTEM` covers every sport today since the judge reads only
    director output, not raw lens reports.

    `lens_specs_by_name` is built lazily so callers can do
    `lens_set.lens_specs_by_name[name]` without an `O(N)` scan over
    `lenses` per dispatch. Computed once per LensSet at first access via
    `__post_init__` (frozen + `object.__setattr__` workaround).
    """

    sport: str
    lenses: tuple[LensSpec, ...]
    director_system_tail: str
    judge_system: str | None = None
    # `lens_specs_by_name` is derived in __post_init__; field hidden from
    # repr so the dataclass stays readable.
    lens_specs_by_name: dict[str, LensSpec] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        # Frozen dataclass workaround — write the derived index via
        # object.__setattr__ once at construction, then never mutate.
        # Cheaper than `cached_property` because it's read on every
        # dispatch and reads should be a plain dict lookup.
        if not self.lens_specs_by_name:
            object.__setattr__(
                self,
                "lens_specs_by_name",
                {spec.name: spec for spec in self.lenses},
            )

    def lens_names(self) -> tuple[str, ...]:
        """Ordered tuple of lens names for iteration and JSONL persistence."""
        return tuple(spec.name for spec in self.lenses)
