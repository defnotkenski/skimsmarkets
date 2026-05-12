"""Async Kalshi v2 trade API client.

Public endpoints (`GET /events`, `GET /markets`) need no auth and are
used by the matcher. `POST /portfolio/orders` requires the RSA-PSS
signature trio:

    KALSHI-ACCESS-KEY        api key UUID
    KALSHI-ACCESS-TIMESTAMP  Unix epoch in milliseconds (as string)
    KALSHI-ACCESS-SIGNATURE  base64(RSA-PSS-SHA256(timestamp + method + path))

The signed message concatenates the three components with no separator
— `f"{ts_ms}{method}{path}"`. Path includes everything from the host
onwards (e.g. `/trade-api/v2/portfolio/orders`), NOT the base URL.

The private key is loaded lazily on the first signed request so a
public-only run (dry-run, matcher probes) works with no Kalshi config.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from pathlib import Path

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from skimsmarkets.kalshi.models import (
    KalshiCandle,
    KalshiEvent,
    KalshiOrderbook,
    OrderRequest,
    OrderResponse,
)

log = logging.getLogger(__name__)


def _parse_book_side(levels: object) -> list[tuple[float, float]]:
    """Parse one side of `/markets/{ticker}/orderbook`'s `orderbook_fp`.

    Wire shape: `[[price_str, size_str], ...]` (e.g. `[["0.0100",
    "9025.00"], ["0.0200", "36372.00"]]`). Drops any level whose
    fields don't parse as floats.
    """
    if not isinstance(levels, list):
        return []
    out: list[tuple[float, float]] = []
    for level in levels:
        if not isinstance(level, list | tuple) or len(level) < 2:
            continue
        try:
            px = float(level[0])
            sz = float(level[1])
        except (TypeError, ValueError):
            continue
        out.append((px, sz))
    return out


class KalshiOrderError(Exception):
    """Non-2xx response from `/portfolio/orders`.

    Carries the response body (parsed JSON when possible, raw text
    otherwise) AND the request body we sent, so the audit row can
    persist both for forensic comparison. Distinct from network /
    timeout errors which propagate as the underlying httpx exceptions.
    """

    def __init__(
        self,
        *,
        status: int,
        detail: object,
        request_body: dict,
    ) -> None:
        self.status = status
        self.detail = detail
        self.request_body = request_body
        super().__init__(f"Kalshi /portfolio/orders → {status}: {detail!r}")


class KalshiClient:
    """Thin async wrapper for the Kalshi v2 trade API."""

    def __init__(
        self,
        *,
        base_url: str,
        http: httpx.AsyncClient,
        api_key_id: str | None = None,
        private_key_path: str | None = None,
        private_key_pem: str | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._http = http
        self._api_key_id = api_key_id
        # Two ways to supply the signing key — file path (local disk) or
        # inline PEM (cloud env-var deploys). Inline wins when both are
        # set, since explicitly setting the PEM env var is the more
        # specific intent.
        self._private_key_path = private_key_path
        self._private_key_pem = private_key_pem
        self._private_key: rsa.RSAPrivateKey | None = None

    # ------------------------------------------------------------------
    # Public, unauthed reads
    # ------------------------------------------------------------------

    async def list_events(
        self,
        *,
        series_ticker: str,
        status: str = "open",
        with_nested_markets: bool = True,
        limit: int = 200,
    ) -> list[KalshiEvent]:
        """Page through `GET /events` for one series, returning every event.

        For the match-level tennis series the open set rarely exceeds
        one page, but the cursor loop is cheap and future-proofs against
        Grand Slam early rounds where 64-128 first-round matches could
        all be open at once.

        Retries up to 3 times on HTTP 429 with exponential backoff
        (1s, 2s, 4s). The slate adapter fans out across all tennis
        series in parallel (6 series after ITF was added), which
        bursts harder than `/events` likes — without backoff, the
        last 1-2 series in the gather call frequently 429.
        """
        path = "/events"
        params: dict[str, str] = {
            "series_ticker": series_ticker,
            "status": status,
            "limit": str(limit),
        }
        if with_nested_markets:
            params["with_nested_markets"] = "true"
        events: list[KalshiEvent] = []
        cursor: str | None = None
        while True:
            if cursor:
                params["cursor"] = cursor
            backoff = 1.0
            for attempt in range(4):
                r = await self._http.get(f"{self._base}{path}", params=params)
                if r.status_code == 429 and attempt < 3:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                r.raise_for_status()
                break
            payload = r.json()
            for raw in payload.get("events", []):
                events.append(KalshiEvent.model_validate(raw))
            cursor = payload.get("cursor") or None
            if not cursor:
                break
        return events

    async def fetch_orderbook(
        self,
        *,
        ticker: str,
    ) -> KalshiOrderbook | None:
        """`GET /markets/{ticker}/orderbook` → full bid/ask book.

        Returns `KalshiOrderbook` with `yes_levels` and `no_levels`,
        each `list[(price_dollars, size_contracts)]` ordered top-of-
        book first (best bid descending, best ask ascending — same
        convention as `clob/__init__.py:fetch_book`). Public,
        unauthed. Returns None on any HTTP / shape failure
        (degrade-silently posture; the ranker treats missing book
        as "no depth signal").
        """
        path = f"/markets/{ticker}/orderbook"
        try:
            r = await self._http.get(f"{self._base}{path}")
            r.raise_for_status()
            payload = r.json()
        except Exception as e:  # noqa: BLE001
            log.warning("kalshi orderbook ticker=%s failed: %s", ticker, e)
            return None
        ob = payload.get("orderbook_fp") if isinstance(payload, dict) else None
        if not isinstance(ob, dict):
            return None
        yes = _parse_book_side(ob.get("yes_dollars"))
        no = _parse_book_side(ob.get("no_dollars"))
        # Wire shape lists levels worst-first (lowest bid first, lowest
        # ask first). Reverse the bid side so callers can read `[0]` as
        # top-of-book on both sides without thinking about it.
        yes.reverse()
        no.reverse()
        return KalshiOrderbook(yes_levels=yes, no_levels=no)

    async def fetch_candlesticks(
        self,
        *,
        series_ticker: str,
        ticker: str,
        period_interval: int,
        start_ts: int,
        end_ts: int,
    ) -> list[KalshiCandle] | None:
        """`GET /series/{s}/markets/{t}/candlesticks` → list[KalshiCandle].

        `period_interval` is in minutes. **Only `1` and `60` are legal**
        — verified 2026-05-11; 5/15/30 all return HTTP 400. `start_ts`
        and `end_ts` are Unix epoch seconds. Returns the bucket list
        (each carrying OHLC for `yes_bid`, `yes_ask`, `price`, plus
        `volume_fp` and `open_interest_fp`). Public, unauthed.

        Retries up to 3 times on HTTP 429 with exponential backoff
        (1s, 2s, 4s). Kalshi's `/candlesticks` endpoint is much more
        rate-limited than `/orderbook` and `/events` — even small
        bursts trip a per-IP cooldown. Degrade-silently if all
        retries fail.
        """
        path = f"/series/{series_ticker}/markets/{ticker}/candlesticks"
        params = {
            "period_interval": str(period_interval),
            "start_ts": str(start_ts),
            "end_ts": str(end_ts),
        }
        backoff = 1.0
        for attempt in range(4):
            try:
                r = await self._http.get(f"{self._base}{path}", params=params)
                if r.status_code == 429 and attempt < 3:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                r.raise_for_status()
                payload = r.json()
                break
            except Exception as e:  # noqa: BLE001
                if attempt < 3:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                log.warning(
                    "kalshi candlesticks ticker=%s failed: %s", ticker, e,
                )
                return None
        sticks = payload.get("candlesticks") if isinstance(payload, dict) else None
        if not isinstance(sticks, list):
            return None
        out: list[KalshiCandle] = []
        for raw in sticks:
            if not isinstance(raw, dict):
                continue
            try:
                out.append(KalshiCandle.model_validate(raw))
            except Exception:  # noqa: BLE001
                continue
        return out

    async def list_tennis_match_series(self) -> list[str]:
        """Discover all per-match-winner tennis series tickers at runtime.

        Two-layer filter:
          1. Ticker pattern: starts with `KXATP` / `KXWTA` / `KXITF`,
             ends with `MATCH`. Catches `KXATPMATCH`, `KXWTAMATCH`,
             `KXATPCHALLENGERMATCH`, `KXWTACHALLENGERMATCH`,
             `KXITFMATCH` (men's futures), `KXITFWMATCH` (women's
             futures), and any future sub-tour Kalshi adds.
          2. Substring blocklist: tokens that mark a *different market
             structure* on the same per-event surface (not a winner
             market). Currently blocks `EXACT` — series like
             `KXATPEXACTMATCH` use the same event-title format
             ("X vs Y") but the YES markets are exact-score
             predictions ("Tiafoe wins 2-0", "Tiafoe wins 2-1"), so
             the matcher would find duplicate "Tiafoe wins ..." markets
             and ambiguous-match. Add other tokens here as Kalshi
             introduces them (e.g. SET, SPREAD, OVERUNDER).

        Returns a sorted list for deterministic ordering. Caller is
        responsible for falling back to a hardcoded list if this
        returns empty or raises.

        ITF note: per the slate adapter's tour mapping, `KXITFMATCH`
        (men's M-tier futures) is treated as `tour="atp"` for
        downstream MatchStats lookups, and `KXITFWMATCH` as
        `tour="wta"`. MatchStats classifies M15/M25/M35 events under
        their `/atp/fixtures/{date}` endpoint and W15/W25/W35/W75
        under `/wta/fixtures/{date}` — same convention.
        """
        # Tokens that mark a non-winner-format series sharing the same
        # ticker-name surface. Compared case-insensitively against the
        # substring of the ticker BETWEEN the ATP/WTA/ITF prefix and
        # the MATCH suffix — `KXATPCHALLENGERMATCH` is allowed
        # (CHALLENGER is a tour qualifier), `KXATPEXACTMATCH` is not
        # (EXACT is a market-type qualifier).
        non_winner_tokens = ("EXACT", "SPREAD", "SET", "GAME")
        out: set[str] = set()
        cursor: str | None = None
        while True:
            params: dict[str, str] = {"category": "Sports", "limit": "200"}
            if cursor:
                params["cursor"] = cursor
            r = await self._http.get(f"{self._base}/series", params=params)
            r.raise_for_status()
            payload = r.json()
            for s in payload.get("series", []):
                ticker = (s.get("ticker") or "").upper()
                if not ticker:
                    continue
                if not ticker.endswith("MATCH"):
                    continue
                # `KXITF` covers both `KXITFMATCH` (men) and
                # `KXITFWMATCH` (women) — gender split is decoded at
                # the slate-adapter layer via `_TOUR_BY_PREFIX`.
                if not (
                    ticker.startswith("KXATP")
                    or ticker.startswith("KXWTA")
                    or ticker.startswith("KXITF")
                ):
                    continue
                if any(tok in ticker for tok in non_winner_tokens):
                    continue
                out.add(ticker)
            cursor = payload.get("cursor") or None
            if not cursor:
                break
        return sorted(out)

    # ------------------------------------------------------------------
    # Signed, authed writes
    # ------------------------------------------------------------------

    async def place_order(self, order: OrderRequest) -> tuple[OrderResponse, dict]:
        """`POST /portfolio/orders` with the RSA-PSS signature trio.

        Returns `(parsed_response, raw_json)` so the audit row can
        persist the raw payload for forensics (Kalshi's field set has
        shifted between SDK versions).
        """
        has_key_material = bool(
            self._private_key_path or self._private_key_pem
        )
        if not self._api_key_id or not has_key_material:
            raise RuntimeError(
                "Kalshi credentials missing — set KALSHI_API_KEY_ID and "
                "EITHER KALSHI_PRIVATE_KEY_PATH (file path) or "
                "KALSHI_PRIVATE_KEY_PEM (inline PEM contents) to enable "
                "`skims execute --live`."
            )
        endpoint_path = "/portfolio/orders"
        signed_path = f"/trade-api/v2{endpoint_path}"
        body = order.model_dump(exclude_none=True)
        headers = self._sign(method="POST", path=signed_path)
        headers["Content-Type"] = "application/json"
        r = await self._http.post(
            f"{self._base}{endpoint_path}",
            json=body,
            headers=headers,
            timeout=20.0,
        )
        # Don't use raise_for_status() here — it throws an HTTPError whose
        # str() omits the response body, swallowing Kalshi's actual error
        # message (e.g. "invalid field name X", "ticker not tradable").
        # Surface the body verbatim so the trader's audit row captures it.
        if r.status_code >= 400:
            try:
                detail: object = r.json()
            except ValueError:
                detail = r.text
            raise KalshiOrderError(
                status=r.status_code,
                detail=detail,
                request_body=body,
            )
        raw = r.json()
        # Kalshi has wrapped the order in {"order": {...}} in some
        # responses and returned a bare order in others. Accept either.
        order_payload = raw.get("order", raw) if isinstance(raw, dict) else raw
        return OrderResponse.model_validate(order_payload), raw

    # ------------------------------------------------------------------
    # Signing
    # ------------------------------------------------------------------

    def _load_private_key(self) -> rsa.RSAPrivateKey:
        if self._private_key is not None:
            return self._private_key
        # Inline PEM (cloud env-var deploys) takes precedence over a file
        # path. If neither is set, the caller already errored at the
        # credentials check in `place_order` — this is defensive.
        if self._private_key_pem is not None:
            pem_bytes = self._private_key_pem.encode()
            source = "KALSHI_PRIVATE_KEY_PEM"
        elif self._private_key_path is not None:
            pem_bytes = Path(self._private_key_path).read_bytes()
            source = self._private_key_path
        else:
            raise RuntimeError(
                "No Kalshi private key configured — set either "
                "KALSHI_PRIVATE_KEY_PATH or KALSHI_PRIVATE_KEY_PEM."
            )
        key = serialization.load_pem_private_key(pem_bytes, password=None)
        if not isinstance(key, rsa.RSAPrivateKey):
            raise RuntimeError(
                f"{source}: expected an RSA private key, "
                f"got {type(key).__name__}"
            )
        self._private_key = key
        return key

    def _sign(self, *, method: str, path: str) -> dict[str, str]:
        """Build the three-header Kalshi signature.

        Salt length is DIGEST_LENGTH (= 32 for SHA256) per Kalshi's
        official Python SDK — using MAX_LENGTH here would still verify
        but isn't what their server expects to see in test vectors.
        """
        ts_ms = str(int(time.time() * 1000))
        message = f"{ts_ms}{method}{path}".encode()
        key = self._load_private_key()
        sig = key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self._api_key_id or "",
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        }
