from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Default horizon window — markets whose game_start_time sits further out than
# this are left out of the slate. 24h catches "today's slate"; use 48-72 on the
# CLI to pull in tomorrow. Enforced server-side via Polymarket's startTimeMax,
# so events outside the window never hit the matcher/LLM path.
DEFAULT_HORIZON_HOURS = 8

# Max implied probability for the event's favorite. Events whose favorite
# is priced at or above this on the YES mid (`(bid+ask)/2`) are dropped
# from the slate before the LLM path — there's no ranking signal to
# extract from a 99% lock and the LLM spend is wasted. The check picks
# `max` across all markets in the event so it works uniformly: for binary
# head-to-heads `max(YES_mid, NO_mid)` IS the favorite's mid; for 3-way
# soccer `max(home_mid, draw_mid, away_mid)` IS the favorite's mid.
# `--slug X` requests bypass this filter, same posture as the horizon
# filter — explicit slug fetches are user-driven.
MAX_IMPLIED_PROBABILITY = 0.70

# Cap on the number of events sent through the LLM chain from the default
# browse. Survivors of all upstream filters (league + horizon + tradability
# + blowout) are sorted by earliest market tipoff ascending and the top N
# are kept; the rest are dropped before enrichment and LLM spend. Tuned
# for cost containment on heavy days — the umbrella `tag_slug=sports`
# browse can return 150+ events post-filter, which is ~$45 of LLM spend
# at ~$0.30/event. `--slug X` fetches bypass the cap (added on top after
# truncation), same posture as the horizon + blowout filters.
MAX_SLATE_EVENTS = 5

# Concurrency caps. See plan for rationale.
# Each event runs through 4 Grok fetchers (Stage A) → 4 Claude reasoners
# (Stage B) → 1 Claude director, all parallel where possible. The fetcher
# semaphore caps concurrent Grok calls across all events; the reasoner
# semaphore caps concurrent Claude reasoner calls (4 per event vs 1 director
# per event, so reasoner sem is roughly 4× director sem).
FETCHER_SEM = 16
REASONER_SEM = 8
DIRECTOR_SEM = 2
# Per-event Unusual Whales detail fan-out. UW doesn't publish rate limits; a
# conservative cap keeps us safely under whatever they enforce. Each event
# triggers at most 1 gamma-api call + 1 UW detail call (YES side only).
UW_FETCH_SEM = 8
# Gamma-api fan-out (event listing + per-slug detail). Same conservative
# ceiling as UW since both ride the same public gamma host.
GAMMA_FETCH_SEM = 8
# CLOB fetch concurrency (clob.polymarket.com `/book` + `/prices-history`).
# Shared across both endpoints since they hit the same host. Public, unauthed,
# but we hedge against unannounced rate limits. Fires once per unique slug
# per enrichment stage.
CLOB_FETCH_SEM = 8

# Opt-in CLOB price-history enrichment toggle. When True, the pipeline
# fetches ~24h of mid-price points per unique slug from `clob.polymarket.com`
# and attaches a sparkline + recency-windowed scalars (30m/1h/4h/24h) to
# each market for the director's context. Adds one HTTP call per unique
# slug. Flip to True here when you want the enrichment on; no env var is
# read for this, the source-of-truth lives in this file so the setting is
# visible in code review and easily greppable.
CLOB_HISTORY_ENABLED = True


@dataclass(frozen=True)
class Config:
    xai_api_key: str
    anthropic_api_key: str
    # Optional — UW enrichment is a nice-to-have, not a hard dependency. When
    # unset, `resolve_unusual_whales()` is skipped and the pipeline behaves
    # exactly as it did pre-integration.
    unusual_whales_api_key: str | None = None

    @classmethod
    def from_env(cls) -> "Config":
        # Reads .env from the current directory (and parents) if present. Does not
        # override vars that are already set in the shell, so explicit exports win.
        load_dotenv()
        xai = os.environ.get("XAI_API_KEY", "").strip()
        anth = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        uw = os.environ.get("UNUSUAL_WHALES_API_KEY", "").strip() or None
        missing: list[str] = []
        if not xai:
            missing.append("XAI_API_KEY")
        if not anth:
            missing.append("ANTHROPIC_API_KEY")
        if missing:
            raise RuntimeError(
                f"Missing required env var(s): {', '.join(missing)}. "
                "Add them to a .env file at the project root or export them in your shell."
            )
        return cls(xai_api_key=xai, anthropic_api_key=anth, unusual_whales_api_key=uw)
