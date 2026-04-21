from __future__ import annotations

import logging
from types import TracebackType
from typing import Any, Self

import httpx

from skimsmarkets.kalshi.models import KalshiEvent, KalshiMarket, KalshiSeries

log = logging.getLogger(__name__)

KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
PAGE_LIMIT = 200


class KalshiClient:
    def __init__(
        self,
        base_url: str = KALSHI_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            transport=httpx.AsyncHTTPTransport(retries=2),
            headers={"Accept": "application/json"},
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._client.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        # httpx drops params whose value is None; keep booleans as lowercase strings
        # because the Kalshi API expects `with_nested_markets=true`, not `True`.
        clean: dict[str, Any] = {}
        for k, v in params.items():
            if v is None:
                continue
            clean[k] = str(v).lower() if isinstance(v, bool) else v
        resp = await self._client.get(path, params=clean)
        resp.raise_for_status()
        return resp.json()

    async def list_sports_series(self) -> list[KalshiSeries]:
        """Dynamically discover all sports series tickers."""
        data = await self._get("/series", {"category": "Sports"})
        series = data.get("series") or data.get("data") or []
        return [KalshiSeries.model_validate(s) for s in series]

    async def list_open_events(
        self,
        series_ticker: str | None = None,
        *,
        with_nested_markets: bool = True,
    ) -> list[KalshiEvent]:
        """Return all open events for a series (or across all series if None), paginated via cursor."""
        events: list[KalshiEvent] = []
        cursor = ""
        page = 0
        while True:
            page += 1
            params: dict[str, Any] = {
                "status": "open",
                "limit": PAGE_LIMIT,
                "with_nested_markets": with_nested_markets,
            }
            if series_ticker:
                params["series_ticker"] = series_ticker
            if cursor:
                params["cursor"] = cursor
            data = await self._get("/events", params)
            batch = data.get("events", [])
            for raw in batch:
                events.append(KalshiEvent.model_validate(raw))
            cursor = data.get("cursor") or ""
            log.debug(
                "kalshi /events page %d: series=%s count=%d cursor=%s",
                page,
                series_ticker,
                len(batch),
                bool(cursor),
            )
            if not cursor:
                break
        return events

    async def list_markets_for_event(self, event_ticker: str) -> list[KalshiMarket]:
        """Fallback when an event response doesn't include nested markets."""
        markets: list[KalshiMarket] = []
        cursor = ""
        while True:
            params: dict[str, Any] = {
                "event_ticker": event_ticker,
                "limit": PAGE_LIMIT,
                "status": "open",
            }
            if cursor:
                params["cursor"] = cursor
            data = await self._get("/markets", params)
            for raw in data.get("markets", []):
                markets.append(KalshiMarket.model_validate(raw))
            cursor = data.get("cursor") or ""
            if not cursor:
                break
        return markets
