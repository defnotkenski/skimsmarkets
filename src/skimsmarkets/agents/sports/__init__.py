"""Sport-keyed lens-set registry — the dispatch entry point for the
per-sport lens architecture.

`SPORT_LENS_SETS` maps `event.sport_type` → `LensSet`. Strict-declaration
posture: events whose `sport_type` has no registration drop with
`ErrorRecord(stage="lens_dispatch")`. Adding a sport requires explicit
registration here, which forces the per-sport prompts/schemas to actually
exist before that sport's events flow through the LLM stack.

Adding a sport:
  1. Build `agents/sports/<sport>/{__init__.py, schemas.py, prompts.py,
     lens_set.py}` with bespoke specs and director tail.
  2. Add `_TOOLS_BY_LENS` entries in `agents/fetchers/grok.py` and
     `agents/fetchers/gemini.py` for the new lens names.
  3. Register here: `SPORT_LENS_SETS["<sport>"] = <SPORT>_LENS_SET`.
"""

from __future__ import annotations

from skimsmarkets.agents.sports._director_shared import DIRECTOR_SHARED_PREAMBLE
from skimsmarkets.agents.sports.base import LensSet, LensSpec
from skimsmarkets.agents.sports.tennis import TENNIS_LENS_SET
from skimsmarkets.polymarket.models import PolymarketEvent

# Strict registry. Sports with no entry drop at lens-dispatch time.
SPORT_LENS_SETS: dict[str, LensSet] = {
    "tennis": TENNIS_LENS_SET,
}


def resolve_lens_set(event: PolymarketEvent) -> LensSet | None:
    """Return the registered `LensSet` for this event's `sport_type`, or
    `None` when no lens set is registered.

    `None` is the dispatch-time signal for "drop with
    ErrorRecord(stage='lens_dispatch')" — the pipeline catches it before
    enrichment so we don't pay HTTP for events we can't process.

    `event.sport_type` is `None` for futures / esports without recognized
    gamma tags / labels missing the sport tag entirely; those also drop.
    """
    if event.sport_type is None:
        return None
    return SPORT_LENS_SETS.get(event.sport_type)


__all__ = [
    "DIRECTOR_SHARED_PREAMBLE",
    "LensSet",
    "LensSpec",
    "SPORT_LENS_SETS",
    "TENNIS_LENS_SET",
    "resolve_lens_set",
]
