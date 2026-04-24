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

_BASE_URL = "https://api.unusualwhales.com/api/predictions"
# Trades arrays come back with up to 50 entries; keep only the few most
# recent per category to keep prompt context small. The feed is reverse
# chronological so slicing from the head is the right move.
_SMART_TRADE_LIMIT = 5
_CONTRARIAN_TRADE_LIMIT = 5
_INSIDER_LIMIT = 3


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
        """GET /predictions/market/{asset_id} → compact `UnusualWhalesContext`.

        Returns None if UW is disabled, the request fails, or the response
        is missing the expected `data` envelope.
        """
        if not self.enabled or self._client is None:
            return None
        url = f"{_BASE_URL}/market/{asset_id}"
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            # 404 on an asset we haven't seen in UW yet is normal; log quietly.
            if e.response.status_code == 404:
                log.debug("uw market %s: 404 (not tracked)", asset_id)
            else:
                # Avoid logging Authorization header contents by never including
                # the request object — httpx's str(response) is body-only.
                log.warning(
                    "uw market %s: HTTP %s",
                    asset_id,
                    e.response.status_code,
                )
            return None
        except Exception as e:  # noqa: BLE001
            log.warning("uw market %s: %s", asset_id, type(e).__name__)
            return None

        try:
            payload = resp.json()
        except Exception as e:  # noqa: BLE001
            log.warning("uw market %s: non-json response (%s)", asset_id, e)
            return None

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            log.debug("uw market %s: missing data envelope", asset_id)
            return None

        return _context_from_detail(asset_id, data)


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
            contrarian_whale_trades=_trades(
                data.get("contrarian_whale_trades"), _CONTRARIAN_TRADE_LIMIT
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
    """Detail endpoint doesn't always surface `unusual_score` at top level;
    fall back to summing weighted tag_scores, which is how UW computes it."""
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
