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


class CryptoPanicClient:
    """CryptoPanic API client for crypto news with sentiment analysis.
    
    Features:
    - Crypto news aggregation
    - Sentiment analysis
    - Filtering by currencies, regions, kinds
    - Public and authenticated endpoints
    - Rate limit: 100 requests/hour (free tier)
    """

    def __init__(self, timeout_seconds: float = 15.0) -> None:
        self._settings = get_settings()
        self._client = httpx.AsyncClient(
            base_url="https://cryptopanic.com/api/v1",
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

    async def get_news(
        self,
        *,
        auth_token: Optional[str] = None,
        currencies: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        kinds: Optional[List[str]] = None,
        filter: Optional[str] = None,
        public: bool = True,
        metadata: bool = False,
    ) -> Dict[str, Any]:
        """GET /posts
        
        Get news posts with optional filtering.
        """
        params = {}
        
        if auth_token:
            params["auth_token"] = auth_token
        else:
            # Use public endpoint without auth
            params["public"] = "true"
        if currencies:
            params["currencies"] = ",".join(currencies)
        if regions:
            params["regions"] = ",".join(regions)
        if kinds:
            params["kinds"] = ",".join(kinds)
        if filter:
            params["filter"] = filter
        if not public:
            params["public"] = "false"
        if metadata:
            params["metadata"] = "true"
            
        response = await self._request_with_retries("GET", "/posts", params=params)
        response.raise_for_status()
        return response.json()

    async def get_news_by_currency(
        self, currency: str, auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get news for a specific currency."""
        return await self.get_news(currencies=[currency], auth_token=auth_token)

    async def get_currencies(self) -> Dict[str, Any]:
        """Get list of supported currencies."""
        response = await self._request_with_retries("GET", "/currencies")
        response.raise_for_status()
        return response.json()

    def parse_news_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse and normalize news response."""
        if "results" not in response:
            return []
        
        normalized_news = []
        for post in response["results"]:
            # Calculate sentiment score from votes
            votes = post.get("votes", {})
            positive = votes.get("positive", 0)
            negative = votes.get("negative", 0)
            total_votes = positive + negative
            
            sentiment_score = 0.0
            if total_votes > 0:
                sentiment_score = (positive - negative) / total_votes
            
            normalized_post = {
                "id": post["id"],
                "title": post["title"],
                "url": post["url"],
                "published_at": post["published_at"],
                "source": post.get("source", {}).get("title", "Unknown"),
                "domain": post.get("domain", ""),
                "currencies": [curr["code"] for curr in post.get("currencies", [])],
                "kind": post.get("kind", "news"),
                "sentiment_score": sentiment_score,
                "votes": votes,
                "metadata": post.get("metadata", {}),
            }
            normalized_news.append(normalized_post)
        
        return normalized_news

    def get_sentiment_label(self, sentiment_score: float) -> str:
        """Convert sentiment score to label."""
        if sentiment_score >= 0.3:
            return "bullish"
        elif sentiment_score <= -0.3:
            return "bearish"
        else:
            return "neutral"
