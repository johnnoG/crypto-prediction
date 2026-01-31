#!/usr/bin/env python3
"""Targeted API tests that run fully in-process (no external services)."""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from backend.app.main import app
from backend.app.services import smart_cache_service


@pytest.fixture
def stubbed_services(monkeypatch):
    """Patch external integrations so tests can run offline."""

    async def async_noop(*args, **kwargs):
        return None

    async def fake_get_prices(self, ids, vs_currencies=None):
        vs = vs_currencies or ["usd"]
        return {coin: {curr: 123.45 for curr in vs} for coin in ids}

    async def fake_get_market_data(self, ids, vs_currency="usd"):
        return {
            coin: {
                "price": 100.0,
                "price_change_24h": 0.5,
                "market_cap": 1_000_000,
                "volume_24h": 50_000,
                "symbol": coin[:3].upper(),
                "name": coin.title(),
            }
            for coin in ids
        }

    async def fake_get_news_articles(**kwargs):
        return {
            "items": [
                {
                    "id": "demo",
                    "title": "Demo Article",
                    "url": "https://example.com",
                    "published_at": None,
                    "source": "Example",
                }
            ],
            "page": 1,
            "limit": 20,
            "total": 1,
            "total_pages": 1,
        }

    monkeypatch.setattr(
        smart_cache_service.SmartCacheService, "initialize", async_noop, raising=False
    )
    monkeypatch.setattr(
        smart_cache_service.SmartCacheService, "cleanup", async_noop, raising=False
    )
    monkeypatch.setattr(
        smart_cache_service.SmartCacheService, "get_prices", fake_get_prices, raising=False
    )
    monkeypatch.setattr(
        smart_cache_service.SmartCacheService, "get_market_data", fake_get_market_data, raising=False
    )
    monkeypatch.setattr(
        "backend.app.services.news_service.get_news_articles", fake_get_news_articles
    )

    yield


@pytest_asyncio.fixture
async def client(stubbed_services):
    """Provide an in-process ASGI test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_quick_health(client):
    resp = await client.get("/health/quick")
    data = resp.json()
    assert resp.status_code == 200
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_prices_endpoint_uses_stub(client):
    resp = await client.get("/prices?ids=bitcoin,ethereum&vs_currencies=usd")
    data = resp.json()
    assert resp.status_code == 200, resp.text
    assert data["bitcoin"]["usd"] == 123.45
    assert data["ethereum"]["usd"] == 123.45


@pytest.mark.asyncio
async def test_market_endpoint_uses_stub(client):
    resp = await client.get("/prices/market?ids=bitcoin,solana&vs_currency=usd")
    data = resp.json()
    assert resp.status_code == 200, resp.text
    assert set(data.keys()) == {"bitcoin", "solana"}
    assert data["bitcoin"]["symbol"] == "BIT"


@pytest.mark.asyncio
async def test_news_endpoint_with_realtime_disabled(client):
    resp = await client.get("/news?realtime=false")
    data = resp.json()
    assert resp.status_code == 200, resp.text
    assert data["total"] == 1
    assert data["items"][0]["title"] == "Demo Article"


@pytest.mark.asyncio
async def test_cache_status_endpoint(client):
    resp = await client.get("/cache/status")
    data = resp.json()
    assert resp.status_code == 200
    assert data["backend"] in {"memory", "redis"}


@pytest.mark.asyncio
async def test_not_found_route(client):
    resp = await client.get("/nope")
    assert resp.status_code == 404

