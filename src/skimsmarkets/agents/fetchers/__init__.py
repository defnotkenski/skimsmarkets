"""Per-lens fetcher stage — Stage A of the two-stage agent chain.

Each provider's `fetch(event, lens)` runs evidence-gathering with its
native search + code-execution tools and emits a `LensNotebook` (free-form
prose + citations + computed numbers). No probability, no signed shift,
no directional verdict — those land in Stage B (`agents/reasoners.py`).

Public surface:
- `FetcherProvider` — the Protocol every provider implements.
- `build_provider` — name-keyed factory, returns a configured provider.
- `render_context`, `pick_team_a_market` — provider-agnostic helpers also
  used by `agents/reasoners.py` (the reasoner's user message starts from
  the same rendered context the fetcher saw).
"""

from skimsmarkets.agents.fetchers.base import (
    FetcherProvider,
    pick_team_a_market,
    render_context,
    render_lens_extras,
)
from skimsmarkets.agents.fetchers.factory import build_provider

__all__ = [
    "FetcherProvider",
    "build_provider",
    "pick_team_a_market",
    "render_context",
    "render_lens_extras",
]
