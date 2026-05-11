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
MAX_IMPLIED_PROBABILITY = 0.60

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
# Tennis stats provider fan-out. Fires at most once per ATP/WTA singles
# event after the sport-gate filters out everything else, so the cap is
# loose — most slates carry only a handful of tennis matches. Conservative
# default mirrors UW_FETCH_SEM since both ride third-party APIs whose
# rate limits we'd rather not probe.
TENNIS_STATS_FETCH_SEM = 8

# Opt-in CLOB price-history enrichment toggle. When True, the pipeline
# fetches ~24h of mid-price points per unique slug from `clob.polymarket.com`
# and attaches a sparkline + recency-windowed scalars (30m/1h/4h/24h) to
# each market for the director's context. Adds one HTTP call per unique
# slug. Flip to True here when you want the enrichment on; no env var is
# read for this, the source-of-truth lives in this file so the setting is
# visible in code review and easily greppable.
CLOB_HISTORY_ENABLED = True


# Per-lens fetcher provider. Source of truth — flip the constant in
# source rather than via env, so the setting is visible in code review
# and easily greppable (same posture as `CLOB_HISTORY_ENABLED` above).
# Validated against `FETCHER_PROVIDERS` at startup so a typo here fails
# loudly rather than later in `build_provider`. Add new providers to the
# tuple when they ship.
FETCHER_PROVIDERS: tuple[str, ...] = ("grok", "gemini")
FETCHER_PROVIDER = "grok"


# ---------------------------------------------------------------------------
# Kalshi execute module — venue config + safety defaults
# ---------------------------------------------------------------------------
#
# Polymarket is the data source (`skims rank` ranks Polymarket events).
# Kalshi is the execution venue (`skims execute` places trades). Public
# read endpoints are unauthed; order placement uses an RSA-signed POST
# (KALSHI_API_KEY_ID + the private key, supplied as either
# KALSHI_PRIVATE_KEY_PATH for local-disk or KALSHI_PRIVATE_KEY_PEM for
# cloud env-var-only deploys like Claude routines / GitHub Actions).
#
# Defaults below are deliberately tiny — the smoke-test blast radius
# (one minimum-size contract at $1) lives here so `--live` without
# override caps a single accidental run at single-digit dollars.
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Tennis match-level series — FALLBACK ONLY.
#
# The trader auto-discovers tennis series at runtime via Kalshi's
# `/series` catalog (filter: KX{ATP|WTA}*MATCH prefix/suffix). This
# tuple is the safety net used only when discovery fails (Kalshi API
# down, schema change, etc.) — keep it as the known-good main-tour
# set so the trader degrades to "trade Rome / Slams only" instead of
# refusing to run.
#
# Discovery catches any future sub-tours Kalshi adds (challenger
# already covered; ITF could be next) without code changes. To force
# a particular set instead of trusting discovery, edit this tuple AND
# bypass the discovery call in `execute/trader.py::_prefetch_events`.
KALSHI_TENNIS_SERIES_TICKERS: tuple[str, ...] = (
    "KXATPMATCH",
    "KXWTAMATCH",
    "KXATPCHALLENGERMATCH",
    "KXWTACHALLENGERMATCH",
)

# --- Spend caps ------------------------------------------------------------
# All money is in CENTS (integer). 100 = $1.00, 2500 = $25.00, 100000 = $1000.
# CLI flags override per-invocation; constants here are the defaults.

# Per-trade spend cap. Passed to Kalshi as `buy_max_cost` on each market
# buy. Kalshi fills as many whole contracts as fit at current asks; if the
# book is thin, actual fill cost may be slightly less than this.
# Common values: 100 ($1) for smoke tests, 2500 ($25) typical, 10000 ($100).
KALSHI_DEFAULT_BET_SIZE_CENTS = 2500

# Slippage buffer for market orders. Kalshi requires a price field on every
# order — even "market" type — which acts as a per-contract ceiling for the
# sweep. We send `yes_price = current_ask_cents + this_buffer`, so the
# order fills any contracts at the current ask up to a few cents above it.
# 0 = limit-at-ask (won't fill if book ticks up between decision and order
# arrival). Higher = more headroom for volatile books, at the cost of
# accepting more slippage per contract. 5¢ is a sensible default for tennis
# books which typically move in 1-2¢ ticks. Capped at 99¢ wire-side
# (Kalshi rejects yes_price > 99).
KALSHI_MARKET_ORDER_SLIPPAGE_CENTS = 5

# Belt-and-suspenders ceiling. `bet_size_cents` must be ≤ this or execute
# refuses to start (fail-fast guard against typos like a stray zero).
# Set this once at your "trades this big or smaller" comfort level and
# leave it pinned; adjust `bet_size_cents` underneath as needed.
KALSHI_DEFAULT_MAX_POSITION_CENTS = 5000

# Calendar-day spend ceiling (UTC). Summed across every `logs/trades/*.jsonl`
# row whose `audit_timestamp` is today. Survives multiple ranker runs in
# one day — if you run rank + execute twice and trade $20 each time, the
# second run will refuse the $21st dollar with cap=4000.
# Sizing rule of thumb: ≥ N × bet_size_cents where N is "max trades I
# expect to place per day". A 25-trade day at $25/trade = 62500.
KALSHI_DEFAULT_MAX_DAILY_SPEND_CENTS = 25000

# --- Filter-flag defaults for `skims execute` ------------------------------
# Empty tuple / None / False here = filter is OFF (every row passes).
# Edit a constant non-empty to lock that filter on for every invocation.
# CLI flags always win when present.

# Which confidence tiers to keep. Valid values: "low", "medium", "high".
# Empty = all tiers pass. Common: ("high", "medium") to skip low-conviction
# rows; ("high",) for the strictest filter.
KALSHI_DEFAULT_CONFIDENCE_TIERS: tuple[str, ...] = ("high", "medium")

# Minimum judge defensibility score. Bar boundaries (see `_defensibility_stars`
# in reporting.py): 0.85 = 5 bars, 0.65 = 4 bars, 0.45 = 3 bars, 0.25 = 2 bars.
# Set to 0.65 to keep only 4-5 bar rows. None = no cutoff.
# IMPORTANT: rows with `defensibility_score=None` (judge failure) always FAIL
# this gate when a cutoff is set — they're treated as "can't verify".
KALSHI_DEFAULT_MIN_DEFENSIBILITY: float | None = 0.45

# Drop predictions where the director agrees with Polymarket's market but
# with lower conviction (= negative expected value vs the market). When the
# flag is True (drop), rows with `negative_edge=None` are also dropped
# (None = "can't verify, safe-default to drop").
# Per-invocation override: `--negative-edge` (allow) or `--no-negative-edge`
# (drop). Set True here if you want the safety filter ON by default.
KALSHI_DEFAULT_NO_NEGATIVE_EDGE: bool = True

# Sport-type allowlist. v1 only supports "tennis" — any other value will
# raise at startup. Set to ("tennis",) to lock the v1 scope at the config
# layer; leave empty to let all rows through the sport gate (the matcher
# will still skip non-tennis rows as `no_kalshi_match`).
KALSHI_DEFAULT_SPORTS: tuple[str, ...] = ()


@dataclass(frozen=True)
class Config:
    # Provider keys are conditionally required based on `fetcher_provider`:
    # only the chosen one is validated at startup so a Grok-only run doesn't
    # need GOOGLE_API_KEY (and vice versa). Anthropic is always required —
    # reasoner / director / judge all use it regardless of fetcher choice.
    anthropic_api_key: str
    fetcher_provider: str = FETCHER_PROVIDER
    xai_api_key: str | None = None
    google_api_key: str | None = None
    # Optional — UW enrichment is a nice-to-have, not a hard dependency. When
    # unset, `resolve_unusual_whales()` is skipped and the pipeline behaves
    # exactly as it did pre-integration.
    unusual_whales_api_key: str | None = None
    # Optional — third-party tennis stats vendor key. Mirrors UW posture:
    # silently absent → the stub provider runs, every event ends up with
    # `tennis_stats=None`, and the rest of the pipeline behaves as before.
    # Concrete provider adapter keyed off this lands in a follow-up.
    tennis_stats_api_key: str | None = None
    # Runtime opt-out from the CLI's `--no-tennis-stats` flag. Kept on the
    # config so the pipeline stage and the provider factory can both see
    # it without the CLI threading a separate argument through every call
    # site. Defaults False so env-only runs still pick up the key.
    tennis_stats_disabled: bool = False
    # Optional — Kalshi execution credentials. Only the order-placement
    # path (`skims execute --live`) requires the key; read-only probes
    # of `/events` / `/markets` work without it. Mirrors the optional-
    # vendor posture: silently absent → `skims execute --live` fails
    # loudly at startup; dry-run is unaffected.
    #
    # The private key can be supplied two ways. Pick one:
    #   - KALSHI_PRIVATE_KEY_PATH: filesystem path to a .pem (local-disk
    #     pattern; works on your laptop / VMs / containers with mounted
    #     volumes).
    #   - KALSHI_PRIVATE_KEY_PEM: inline PEM contents in the env var
    #     itself (cloud-scheduler pattern — Claude routines, GitHub
    #     Actions, Lambda, etc. where the platform only exposes env
    #     vars and there's no persistent disk).
    # If both are set, the inline PEM wins.
    kalshi_api_key_id: str | None = None
    kalshi_private_key_path: str | None = None
    kalshi_private_key_pem: str | None = None

    @classmethod
    def from_env(
        cls,
        *,
        tennis_stats_disabled: bool = False,
        require_llm: bool = True,
    ) -> "Config":
        # Reads .env from the current directory (and parents) if present. Does not
        # override vars that are already set in the shell, so explicit exports win.
        # `fetcher_provider` is hand-edited in this file's `FETCHER_PROVIDER`
        # constant — there is no env var or CLI override. Validated here so a
        # typo in the constant fails loudly at startup.
        #
        # `require_llm=False` is for code paths that don't talk to any LLM
        # (e.g. `skims execute` reading the JSONL → placing Kalshi orders).
        # In that mode the Anthropic / Grok / Gemini key checks are skipped
        # and the placeholder anthropic_api_key is set to an empty string
        # so callers that mistakenly try to reach the LLM later fail loudly.
        load_dotenv()
        provider = FETCHER_PROVIDER
        if provider not in FETCHER_PROVIDERS:
            raise RuntimeError(
                f"Unknown FETCHER_PROVIDER {provider!r} in config.py. "
                f"Valid: {', '.join(FETCHER_PROVIDERS)}."
            )
        xai = os.environ.get("XAI_API_KEY", "").strip() or None
        google = os.environ.get("GOOGLE_API_KEY", "").strip() or None
        anth = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        uw = os.environ.get("UNUSUAL_WHALES_API_KEY", "").strip() or None
        tennis_key = os.environ.get("TENNIS_STATS_API_KEY", "").strip() or None
        kalshi_id = os.environ.get("KALSHI_API_KEY_ID", "").strip() or None
        kalshi_pk = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "").strip() or None
        # Don't .strip() the PEM — leading/trailing newlines around the
        # `-----BEGIN/END-----` markers are legal in PEM-format text and
        # `load_pem_private_key` handles them. We only treat "" as absent.
        kalshi_pem = os.environ.get("KALSHI_PRIVATE_KEY_PEM") or None
        if require_llm:
            missing: list[str] = []
            if not anth:
                missing.append("ANTHROPIC_API_KEY")
            if provider == "grok" and not xai:
                missing.append("XAI_API_KEY (required when fetcher_provider=grok)")
            if provider == "gemini" and not google:
                missing.append("GOOGLE_API_KEY (required when fetcher_provider=gemini)")
            if missing:
                raise RuntimeError(
                    f"Missing required env var(s): {', '.join(missing)}. "
                    "Add them to a .env file at the project root or export them in your shell."
                )
        return cls(
            anthropic_api_key=anth,
            fetcher_provider=provider,
            xai_api_key=xai,
            google_api_key=google,
            unusual_whales_api_key=uw,
            tennis_stats_api_key=tennis_key,
            tennis_stats_disabled=tennis_stats_disabled,
            kalshi_api_key_id=kalshi_id,
            kalshi_private_key_path=kalshi_pk,
            kalshi_private_key_pem=kalshi_pem,
        )
