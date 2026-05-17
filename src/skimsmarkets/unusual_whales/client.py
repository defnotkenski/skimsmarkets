"""Thin async HTTP client for Unusual Whales' prediction endpoints.

Why our own client (not an SDK): UW's Python SDK doesn't cover the
predictions surface, and the REST shape is stable enough that a direct
httpx wrapper is cleaner than a generated-client detour. Responses land
as plain dicts; we compress them to our `UnusualWhalesContext` at the
call site so no UW-specific types leak into downstream modules.

Failures return None on any level — network error, non-2xx, malformed
JSON — so the pipeline can degrade gracefully the same way it does for
Polymarket BBO.
"""

from __future__ import annotations

import asyncio
import logging
from types import TracebackType
from typing import Any, Self

import httpx

from skimsmarkets.unusual_whales.models import (
    UnusualWhalesContext,
    UWInsider,
    UWLiquidity,
    UWMci,
    UWTrade,
    tag_scores_from_list,
)

log = logging.getLogger(__name__)

_BASE_URL = "https://phx.unusualwhales.com/hashdive/api"
# Trades arrays come back with up to 50 entries; keep only the few most
# recent per category to keep prompt context small. The feed is reverse
# chronological so slicing from the head is the right move.
_SMART_TRADE_LIMIT = 5
_WHALE_TRADE_LIMIT = 5
_INSIDER_LIMIT = 3

# Retry configuration for HTTP 429 (rate limited) responses. UW doesn't
# publish rate limits, but empirically the gamma fan-out can hit a
# per-second cap when multiple events resolve their detail endpoints in
# the same millisecond. Three total attempts with exponential backoff
# (1s → 2s) puts us at ~3s worst case before giving up — short enough
# that the enrichment stage doesn't block the run, long enough to outlast
# a typical sliding-window rate limiter.
_RETRY_ATTEMPTS = 3
_RETRY_BASE_S = 1.0


def _parse_retry_after(value: str | None) -> float | None:
    """Parse an HTTP `Retry-After` header value to seconds.

    Only handles the integer-seconds form (UW's known shape); HTTP-date
    form returns None and we fall back to exponential backoff. Negative
    values clamp to 0.
    """
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None


class UnusualWhalesClient:
    """Async context-managed UW client. `async with UnusualWhalesClient(token) as c: ...`.

    Pass `token=None` or an empty string to construct a disabled client —
    every method returns None. Useful so callers don't have to short-circuit.
    """

    def __init__(
        self,
        api_key: str | None,
        *,
        timeout: float = 20.0,
    ) -> None:
        self._api_key = (api_key or "").strip() or None
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def enabled(self) -> bool:
        return self._api_key is not None

    @property
    def http(self) -> httpx.AsyncClient:
        """Underlying httpx client — exposed so `GammaTokenResolver` can share it."""
        if self._client is None:
            raise RuntimeError(
                "UnusualWhalesClient used outside of `async with` context"
            )
        return self._client

    async def __aenter__(self) -> Self:
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        self._client = httpx.AsyncClient(timeout=self._timeout, headers=headers)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get_market_detail(self, asset_id: str) -> UnusualWhalesContext | None:
        """GET /assets/{asset_id}/detail_agg → compact `UnusualWhalesContext`.

        Hits the Hashdive aggregated-detail endpoint (2026-05-17 API
        migration from the prior `api.unusualwhales.com/api/predictions/
        market/{id}` shape — same payload concept, different host/path).
        Retries up to `_RETRY_ATTEMPTS` times on HTTP 429 with exponential
        backoff (honoring `Retry-After` when present). Returns None if UW
        is disabled, the request fails after retries, or the response is
        not a JSON object (the new endpoint returns the asset record
        directly, with no wrapper envelope — old code unwrapped `data`).
        """
        if not self.enabled or self._client is None:
            return None
        url = f"{_BASE_URL}/assets/{asset_id}/detail_agg"

        resp: httpx.Response | None = None
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                resp = await self._client.get(url)
                resp.raise_for_status()
                break
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                # 404 on an asset we haven't seen in UW yet is normal; log
                # quietly and don't retry — the asset isn't going to appear.
                if status == 404:
                    log.debug("uw market %s: 404 (not tracked)", asset_id)
                    return None
                if status == 429 and attempt + 1 < _RETRY_ATTEMPTS:
                    wait = _parse_retry_after(
                        e.response.headers.get("Retry-After")
                    ) or _RETRY_BASE_S * (2 ** attempt)
                    log.debug(
                        "uw market %s: 429, sleeping %.1fs (attempt %d/%d)",
                        asset_id, wait, attempt + 1, _RETRY_ATTEMPTS,
                    )
                    await asyncio.sleep(wait)
                    continue
                # Avoid logging Authorization header contents by never including
                # the request object — httpx's str(response) is body-only.
                if status == 429:
                    log.warning(
                        "uw market %s: HTTP 429 after %d attempts",
                        asset_id, _RETRY_ATTEMPTS,
                    )
                else:
                    log.warning("uw market %s: HTTP %s", asset_id, status)
                return None
            except Exception as e:  # noqa: BLE001
                log.warning("uw market %s: %s", asset_id, type(e).__name__)
                return None

        if resp is None:
            return None

        try:
            payload = resp.json()
        except Exception as e:  # noqa: BLE001
            log.warning("uw market %s: non-json response (%s)", asset_id, e)
            return None

        # Hashdive `/detail_agg` returns the asset record at the top
        # level (no `data` envelope). Earlier `api.unusualwhales.com`
        # shape wrapped it in `{data: {...}}`; we used to unwrap here.
        if not isinstance(payload, dict):
            log.debug("uw asset %s: unexpected payload shape %s",
                      asset_id, type(payload).__name__)
            return None

        return _context_from_detail(asset_id, payload)


def _context_from_detail(
    asset_id: str, data: dict[str, Any]
) -> UnusualWhalesContext | None:
    """Squash the UW detail payload into our compact context model.

    Validation failures on individual trades are dropped silently; the whole
    context still comes through. UW is a best-effort enrichment, not a hard
    dependency, so a single malformed field shouldn't blank the whole record.
    """
    try:
        mci_raw = data.get("mci")
        liquidity_raw = data.get("liquidity")
        return UnusualWhalesContext(
            asset_id=asset_id,
            question=data.get("question"),
            outcome_label=_outcome_label(data),
            unusual_score=_best_unusual_score(data),
            volume=data.get("volume"),
            tag_scores=tag_scores_from_list(data.get("tag_scores")),
            mci=UWMci.model_validate(mci_raw) if isinstance(mci_raw, dict) else None,
            liquidity=(
                UWLiquidity.model_validate(liquidity_raw)
                if isinstance(liquidity_raw, dict)
                else None
            ),
            smart_trades=_trades(data.get("smart_trades"), _SMART_TRADE_LIMIT),
            # Hashdive renamed `contrarian_whale_trades` (the old
            # tag-classified subset) to plain `whale_trades` (all
            # whale-size fills, irrespective of consensus direction).
            # The "contrarian" angle is preserved in
            # `tag_scores.contrarian_whales` if a caller still wants it.
            whale_trades=_trades(
                data.get("whale_trades"), _WHALE_TRADE_LIMIT,
            ),
            insiders=_insiders(data.get("insiders"), _INSIDER_LIMIT),
        )
    except Exception as e:  # noqa: BLE001
        log.warning("uw context build failed for %s: %s", asset_id, e)
        return None


def _outcome_label(data: dict[str, Any]) -> str | None:
    """Resolve `outcomes[outcome_index]` → the team/outcome name for this asset.

    UW returns the asset's outcome list and the index of which one this asset
    represents; the team name is just `outcomes[outcome_index]`. Falls back to
    None on any shape mismatch — the renderer treats `None` as "unknown side."
    """
    outcomes = data.get("outcomes")
    idx = data.get("outcome_index")
    if not isinstance(outcomes, list) or not isinstance(idx, int):
        return None
    if 0 <= idx < len(outcomes):
        label = outcomes[idx]
        return label if isinstance(label, str) else None
    return None


def _best_unusual_score(data: dict[str, Any]) -> Any:
    """Pull the composite unusual-activity score off the asset record.

    Hashdive surfaces it as `tags_score` at the top level (the old
    `api.unusualwhales.com` shape called the same value `unusual_score`).
    Both keys are checked for forward/back compat. Falls back to summing
    `weighted` across `tag_scores[]` when the top-level field is absent
    — that's how UW computes the composite under the hood.
    """
    direct = data.get("tags_score")
    if direct is None:
        direct = data.get("unusual_score")
    if direct is not None:
        return direct
    scores = data.get("tag_scores")
    if not isinstance(scores, list):
        return None
    total = 0.0
    seen = False
    for entry in scores:
        if not isinstance(entry, dict):
            continue
        weighted = entry.get("weighted")
        if weighted is None:
            continue
        try:
            total += float(weighted)
            seen = True
        except (TypeError, ValueError):
            continue
    return total if seen else None


def _trades(raw: Any, limit: int) -> list[UWTrade]:
    if not isinstance(raw, list):
        return []
    out: list[UWTrade] = []
    for entry in raw[:limit]:
        if not isinstance(entry, dict):
            continue
        try:
            out.append(UWTrade.model_validate(entry))
        except Exception:  # noqa: BLE001
            continue
    return out


def _insiders(raw: Any, limit: int) -> list[UWInsider]:
    if not isinstance(raw, list):
        return []
    out: list[UWInsider] = []
    for entry in raw[:limit]:
        if not isinstance(entry, dict):
            continue
        try:
            out.append(UWInsider.model_validate(entry))
        except Exception:  # noqa: BLE001
            continue
    return out
