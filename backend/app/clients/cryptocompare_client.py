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


class CryptoCompareClient:
    """CryptoCompare API client for historical data and social sentiment.
    
    Features:
    - Historical OHLCV data
    - Social sentiment data
    - News aggregation
    - Price conversion
    - High rate limits (100,000 calls/month free tier)
    """

    def __init__(self, timeout_seconds: float = 15.0) -> None:
        self._settings = get_settings()
        self._client = httpx.AsyncClient(
            base_url="https://min-api.cryptocompare.com",
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

    async def get_historical_ohlcv(
        self,
        fsym: str = "BTC",
        tsym: str = "USD",
        limit: int = 2000,
        aggregate: int = 1,
        to_ts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """GET /data/v2/histohour
        
        Get historical OHLCV data.
        
        Args:
            fsym: From symbol (e.g., BTC)
            tsym: To symbol (e.g., USD)
            limit: Number of data points (max 2000)
            aggregate: Time period in minutes (1, 2, 5, 10, 15, 30, 60, 120, 240, 360, 720, 1440, 4320, 10080)
            to_ts: Timestamp to get data up to
        """
        params = {
            "fsym": fsym,
            "tsym": tsym,
            "limit": limit,
            "aggregate": aggregate,
        }
        if to_ts:
            params["toTs"] = to_ts
            
        response = await self._request_with_retries("GET", "/data/v2/histohour", params=params)
        response.raise_for_status()
        return response.json()

    async def get_historical_daily(
        self,
        fsym: str = "BTC",
        tsym: str = "USD",
        limit: int = 2000,
        to_ts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """GET /data/v2/histoday
        
        Get historical daily OHLCV data.
        """
        params = {
            "fsym": fsym,
            "tsym": tsym,
            "limit": limit,
        }
        if to_ts:
            params["toTs"] = to_ts
            
        response = await self._request_with_retries("GET", "/data/v2/histoday", params=params)
        response.raise_for_status()
        return response.json()

    async def get_price(
        self, fsyms: List[str], tsyms: List[str]
    ) -> Dict[str, Any]:
        """GET /data/pricemulti
        
        Get current price for multiple symbols.
        """
        params = {
            "fsyms": ",".join(fsyms),
            "tsyms": ",".join(tsyms),
        }
        response = await self._request_with_retries("GET", "/data/pricemulti", params=params)
        response.raise_for_status()
        return response.json()

    async def get_social_sentiment(
        self, coin_id: str = "1182"  # Bitcoin's coin ID
    ) -> Dict[str, Any]:
        """GET /data/social/coin/latest
        
        Get social sentiment data for a coin.
        """
        params = {"coinId": coin_id}
        response = await self._request_with_retries("GET", "/data/social/coin/latest", params=params)
        response.raise_for_status()
        return response.json()

    async def get_news_latest(
        self, categories: Optional[str] = None, lang: str = "EN"
    ) -> Dict[str, Any]:
        """GET /data/v2/news/
        
        Get latest news articles.
        """
        params = {"lang": lang}
        if categories:
            params["categories"] = categories
            
        response = await self._request_with_retries("GET", "/data/v2/news/", params=params)
        response.raise_for_status()
        return response.json()

    async def get_top_pairs_by_volume(
        self, fsym: str = "BTC", limit: int = 20
    ) -> Dict[str, Any]:
        """GET /data/top/pairs
        
        Get top trading pairs by volume for a symbol.
        """
        params = {"fsym": fsym, "limit": limit}
        response = await self._request_with_retries("GET", "/data/top/pairs", params=params)
        response.raise_for_status()
        return response.json()

    async def get_exchange_list(self) -> Dict[str, Any]:
        """GET /data/exchanges/general
        
        Get list of exchanges.
        """
        response = await self._request_with_retries("GET", "/data/exchanges/general")
        response.raise_for_status()
        return response.json()

    def historical_to_ohlcv(self, historical_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert CryptoCompare historical format to OHLCV format.
        
        CryptoCompare format:
        {
            "Data": {
                "Data": [
                    {
                        "time": 1614556800,
                        "high": 50000.0,
                        "low": 48000.0,
                        "open": 49000.0,
                        "volumefrom": 1000.0,
                        "volumeto": 49000000.0,
                        "close": 49500.0
                    }
                ]
            }
        }
        """
        ohlcv_data = []
        if "Data" in historical_data and "Data" in historical_data["Data"]:
            for candle in historical_data["Data"]["Data"]:
                ohlcv_data.append({
                    "time": candle["time"],
                    "open": candle["open"],
                    "high": candle["high"],
                    "low": candle["low"],
                    "close": candle["close"],
                    "volume": candle["volumefrom"],
                    "volume_to": candle["volumeto"],
                })
        return ohlcv_data

    def get_coin_id_mapping(self) -> Dict[str, str]:
        """Get mapping of common symbols to CryptoCompare coin IDs."""
        return {
            "BTC": "1182",  # Bitcoin
            "ETH": "7605",  # Ethereum
            "SOL": "48543",  # Solana
            "ADA": "1027",  # Cardano
            "DOT": "6636",  # Polkadot
            "LINK": "1975",  # Chainlink
            "UNI": "7083",  # Uniswap
            "AVAX": "5805",  # Avalanche
            "MATIC": "3890",  # Polygon
            "ATOM": "3794",  # Cosmos
        }

