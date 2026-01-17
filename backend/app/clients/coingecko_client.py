from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import httpx

try:
    from config import get_settings
except ImportError:
    from config import get_settings


class CoinGeckoClient:
    """Typed, minimal CoinGecko client with API key support and retries.

    Notes:
    - Uses httpx.AsyncClient for efficiency.
    - Injects `x-cg-pro-api-key` header if `COINGECKO_API_KEY` is present.
    - Respects base URL from settings.
    - Provides simple retry/backoff for transient errors and 429s.
    """

    def __init__(self, timeout_seconds: float = 8.0) -> None:
        self._settings = get_settings()
        headers = {}
        if self._settings.coingecko_api_key:
            headers["x-cg-pro-api-key"] = self._settings.coingecko_api_key

        self._client = httpx.AsyncClient(
            base_url=self._settings.coingecko_base_url,
            headers=headers,
            timeout=httpx.Timeout(
                connect=5.0,  # 5 second connection timeout
                read=timeout_seconds,  # 8 second read timeout
                write=5.0,  # 5 second write timeout
                pool=5.0  # 5 second pool timeout
            ),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _request_with_retries(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        max_attempts: int = 1,  # Single attempt for faster response
        initial_backoff_seconds: float = 0.2,  # Very fast backoff
    ) -> httpx.Response:
        backoff = initial_backoff_seconds
        last_exc: Optional[Exception] = None
        
        for attempt in range(1, max_attempts + 1):
            try:
                response = await self._client.request(method, url, params=params)
                
                # Handle rate limiting (429) with exponential backoff
                if response.status_code == 429:
                    if attempt < max_attempts:
                        # Check for Retry-After header
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            wait_time = float(retry_after)
                        else:
                            wait_time = backoff
                        
                        await asyncio.sleep(wait_time)
                        backoff *= 2
                        continue
                
                # Retry on 5xx server errors
                elif response.status_code in {500, 502, 503, 504}:
                    if attempt < max_attempts:
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue
                
                return response
                
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                if attempt < max_attempts:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                raise
                
        if last_exc:
            raise last_exc
        raise RuntimeError("Unexpected retry loop termination")

    async def get_simple_price(self, ids: List[str], vs_currencies: List[str] = None) -> Dict[str, Any]:
        """GET /simple/price

        Example: ids=["bitcoin", "ethereum"] vs_currencies=["usd"]
        """
        if vs_currencies is None:
            vs_currencies = ["usd"]
            
        response = await self._request_with_retries(
            "GET",
            "/simple/price",
            params={"ids": ",".join(ids), "vs_currencies": ",".join(vs_currencies)},
        )
        response.raise_for_status()
        return response.json()

    async def get_coins_markets(
        self,
        vs_currency: List[str] = None,
        ids: Optional[List[str]] = None,
        order: str = "market_cap_desc",
        per_page: int = 100,
        page: int = 1,
        sparkline: bool = False,
        price_change_percentage: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """GET /coins/markets

        Get market data for coins.
        """
        if vs_currency is None:
            vs_currency = ["usd"]
            
        params = {
            "vs_currency": ",".join(vs_currency),
            "order": order,
            "per_page": per_page,
            "page": page,
            "sparkline": str(sparkline).lower(),
        }
        
        if ids:
            params["ids"] = ",".join(ids)
        if price_change_percentage:
            params["price_change_percentage"] = ",".join(price_change_percentage)
            
        response = await self._request_with_retries("GET", "/coins/markets", params=params)
        response.raise_for_status()
        return response.json()

    async def get_coin_ohlc_by_id(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: int = 1
    ) -> List[List[float]]:
        """GET /coins/{id}/ohlc

        Get historical OHLC data for a coin.
        """
        params = {
            "vs_currency": vs_currency,
            "days": days,
        }
        
        response = await self._request_with_retries("GET", f"/coins/{coin_id}/ohlc", params=params)
        response.raise_for_status()
        return response.json()


