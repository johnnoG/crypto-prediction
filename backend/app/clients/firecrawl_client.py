from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

try:
    from config import get_settings  # type: ignore
except Exception:  # pragma: no cover
    try:
        from config import get_settings  # type: ignore
    except ImportError:
        from config import get_settings  # type: ignore


class FirecrawlClient:
    """Typed Firecrawl client with retries/backoff and API key support.

    Notes:
    - Uses httpx.AsyncClient for efficiency.
    - Injects Bearer token if FIRECRAWL_API_KEY is present.
    - Respects base URL from settings.
    - Provides simple retry/backoff for transient errors and 429s.
    """

    def __init__(self, timeout_seconds: float = 15.0) -> None:
        self._settings = get_settings()
        headers = {}
        if self._settings.firecrawl_api_key:
            headers["Authorization"] = f"Bearer {self._settings.firecrawl_api_key}"

        self._client = httpx.AsyncClient(
            base_url="https://api.firecrawl.dev/v1",
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _request_with_retries(
        self,
        method: str,
        url: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        max_attempts: int = 3,
        initial_backoff_seconds: float = 1.0,
    ) -> httpx.Response:
        backoff = initial_backoff_seconds
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = await self._client.request(method, url, json=json)
                # Retry on 429 or 5xx, but handle 402 (Payment Required) specially
                if response.status_code == 402:
                    # Payment required - don't retry, raise specific exception
                    raise HTTPException(
                        status_code=402, 
                        detail="Firecrawl API quota exceeded. Please upgrade your plan or use cached data."
                    )
                elif response.status_code in {429, 500, 502, 503, 504}:
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

    async def crawl_url(self, url: str, *, max_content_bytes: Optional[int] = None) -> Dict[str, Any]:
        """POST /crawl

        Crawl a single URL and extract readable content.
        """
        payload = {"url": url}
        if max_content_bytes:
            payload["maxContentBytes"] = max_content_bytes

        response = await self._request_with_retries("POST", "/crawl", json=payload)
        response.raise_for_status()
        return response.json()

    async def crawl_sitemap(self, url: str) -> Dict[str, Any]:
        """POST /crawl/sitemap

        Crawl a sitemap and extract all URLs.
        """
        payload = {"url": url}
        response = await self._request_with_retries("POST", "/crawl/sitemap", json=payload)
        response.raise_for_status()
        return response.json()
