from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import httpx

try:
    from config import get_settings  # type: ignore
except Exception:  # pragma: no cover
    try:
        from config import get_settings  # type: ignore
    except ImportError:
        from config import get_settings  # type: ignore


class BinanceClient:
    """Binance API client for real-time crypto data.
    
    Features:
    - Real-time price data
    - Order book data
    - Historical klines/candlestick data
    - 24hr ticker statistics
    - High rate limits (1200 requests/minute)
    """

    def __init__(self, timeout_seconds: float = 15.0) -> None:
        self._settings = get_settings()
        self._client = httpx.AsyncClient(
            base_url="https://api.binance.com/api/v3",
            timeout=httpx.Timeout(timeout_seconds),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _request_with_retries(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        max_attempts: int = 3,
        initial_backoff_seconds: float = 1.0,
    ) -> httpx.Response:
        backoff = initial_backoff_seconds
        last_exc: Optional[Exception] = None
        
        for attempt in range(1, max_attempts + 1):
            try:
                response = await self._client.request(method, url, params=params)
                # Retry on 429 or 5xx
                if response.status_code in {429, 500, 502, 503, 504}:
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

    async def get_ticker_24hr(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """GET /ticker/24hr
        
        Get 24hr ticker price change statistics.
        """
        params = {"symbol": symbol}
        response = await self._request_with_retries("GET", "/ticker/24hr", params=params)
        response.raise_for_status()
        return response.json()

    async def get_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[List[Any]]:
        """GET /klines
        
        Get kline/candlestick data for a symbol.
        
        Intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
            
        response = await self._request_with_retries("GET", "/klines", params=params)
        response.raise_for_status()
        return response.json()

    async def get_order_book(
        self, symbol: str = "BTCUSDT", limit: int = 100
    ) -> Dict[str, Any]:
        """GET /depth
        
        Get order book for a symbol.
        """
        params = {"symbol": symbol, "limit": limit}
        response = await self._request_with_retries("GET", "/depth", params=params)
        response.raise_for_status()
        return response.json()

    async def get_recent_trades(
        self, symbol: str = "BTCUSDT", limit: int = 500
    ) -> List[Dict[str, Any]]:
        """GET /trades
        
        Get recent trades for a symbol.
        """
        params = {"symbol": symbol, "limit": limit}
        response = await self._request_with_retries("GET", "/trades", params=params)
        response.raise_for_status()
        return response.json()

    async def get_exchange_info(self) -> Dict[str, Any]:
        """GET /exchangeInfo
        
        Get exchange information including trading rules and symbol information.
        """
        response = await self._request_with_retries("GET", "/exchangeInfo")
        response.raise_for_status()
        return response.json()

    def klines_to_ohlcv(self, klines: List[List[Any]]) -> List[Dict[str, Any]]:
        """Convert Binance klines format to OHLCV format.
        
        Binance klines format:
        [
            [
                1499040000000,      // Open time
                "0.01634790",       // Open
                "0.80000000",       // High
                "0.01575800",       // Low
                "0.01577100",       // Close
                "148976.11427815",  // Volume
                1499644799999,      // Close time
                "2434.19055334",    // Quote asset volume
                308,                // Number of trades
                "1756.87402397",    // Taker buy base asset volume
                "28.46694368",      // Taker buy quote asset volume
                "0"                 // Ignore
            ]
        ]
        """
        ohlcv_data = []
        for kline in klines:
            ohlcv_data.append({
                "open_time": kline[0],
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5]),
                "close_time": kline[6],
                "quote_volume": float(kline[7]),
                "trades": int(kline[8]),
                "taker_buy_base": float(kline[9]),
                "taker_buy_quote": float(kline[10]),
            })
        return ohlcv_data

