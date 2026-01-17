#!/usr/bin/env python3
"""
Test script for crypto APIs.

Tests CoinGecko, Binance, CryptoCompare, and CryptoPanic APIs.
"""

import asyncio
import json
import os
from pathlib import Path
import sys

import pytest

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from backend.app.clients.coingecko_client import CoinGeckoClient
    from backend.app.clients.binance_client import BinanceClient
    from backend.app.clients.cryptocompare_client import CryptoCompareClient
    from backend.app.clients.cryptopanic_client import CryptoPanicClient
    from backend.app.config import get_settings
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


if os.environ.get("RUN_EXTERNAL_API_TESTS") != "1":
    pytest.skip(
        "Skipping external crypto API integration tests. "
        "Set RUN_EXTERNAL_API_TESTS=1 to enable.",
        allow_module_level=True,
    )


async def test_coingecko():
    """Test CoinGecko API."""
    print("🧪 Testing CoinGecko API...")
    
    try:
        client = CoinGeckoClient()
        
        # Test simple price endpoint
        prices = await client.get_simple_price(["bitcoin", "ethereum"], ["usd"])
        print(f"✅ CoinGecko prices: {prices}")
        
        # Test market data
        market_data = await client.get_coins_markets(vs_currency=["usd"], ids=["bitcoin"], per_page=1, page=1)
        print(f"✅ CoinGecko market data: {len(market_data)} coins")
        
        await client.close()
        return True
        
    except Exception as e:
        print(f"❌ CoinGecko test failed: {e}")
        return False


async def test_binance():
    """Test Binance API."""
    print("\n🧪 Testing Binance API...")
    
    try:
        client = BinanceClient()
        
        # Test 24hr ticker
        ticker = await client.get_ticker_24hr("BTCUSDT")
        print(f"✅ Binance BTC ticker: ${ticker.get('lastPrice', 'N/A')}")
        
        # Test recent klines
        klines = await client.get_klines("BTCUSDT", "1h", 5)
        print(f"✅ Binance klines: {len(klines)} candles")
        
        await client.close()
        return True
        
    except Exception as e:
        print(f"❌ Binance test failed: {e}")
        return False


async def test_cryptocompare():
    """Test CryptoCompare API."""
    print("\n🧪 Testing CryptoCompare API...")
    
    try:
        client = CryptoCompareClient()
        
        # Test price endpoint
        prices = await client.get_price(["BTC", "ETH"], ["USD"])
        print(f"✅ CryptoCompare prices: {prices}")
        
        # Test historical data
        historical = await client.get_historical_daily("BTC", "USD", 5)
        ohlcv_data = client.historical_to_ohlcv(historical)
        print(f"✅ CryptoCompare historical: {len(ohlcv_data)} days")
        
        await client.close()
        return True
        
    except Exception as e:
        print(f"❌ CryptoCompare test failed: {e}")
        return False


async def test_cryptopanic():
    """Test CryptoPanic API."""
    print("\n🧪 Testing CryptoPanic API...")
    
    try:
        client = CryptoPanicClient()
        settings = get_settings()
        
        # Test public news endpoint
        news = await client.get_news(public=True)
        normalized_news = client.parse_news_response(news)
        print(f"✅ CryptoPanic public news: {len(normalized_news)} articles")
        
        # Test with API key if available
        if settings.cryptopanic_api_key:
            auth_news = await client.get_news(
                auth_token=settings.cryptopanic_api_key,
                currencies=["BTC"],
                public=True
            )
            auth_normalized = client.parse_news_response(auth_news)
            print(f"✅ CryptoPanic authenticated news: {len(auth_normalized)} BTC articles")
        else:
            print("⚠️  No CryptoPanic API key provided, skipping authenticated test")
        
        # Test currencies endpoint
        currencies = await client.get_currencies()
        print(f"✅ CryptoPanic currencies: {len(currencies.get('results', []))} supported")
        
        await client.close()
        return True
        
    except Exception as e:
        print(f"❌ CryptoPanic test failed: {e}")
        return False


async def test_firecrawl():
    """Test Firecrawl API."""
    print("\n🧪 Testing Firecrawl API...")
    
    try:
        from backend.app.clients.firecrawl_client import FirecrawlClient
        
        client = FirecrawlClient()
        settings = get_settings()
        
        if not settings.firecrawl_api_key:
            print("⚠️  No Firecrawl API key provided, skipping test")
            return False
        
        # Test crawling a simple URL
        result = await client.crawl_url(
            "https://coindesk.com",
            max_content_bytes=10000
        )
        
        print(f"✅ Firecrawl crawl: {result.get('status', 'unknown')} status")
        print(f"   Title: {result.get('data', {}).get('title', 'N/A')}")
        
        await client.close()
        return True
        
    except Exception as e:
        print(f"❌ Firecrawl test failed: {e}")
        return False


async def main():
    """Run all API tests."""
    print("🚀 Crypto API Testing Suite")
    print("=" * 50)
    
    # Set environment variables for testing
    os.environ["FIRECRAWL_API_KEY"] = "fc-a04fcc8d98f64bcd8fb27d8dbd00eee1"
    os.environ["CRYPTOPANIC_API_KEY"] = "8e4c69d9e095290d6a57b86ecfb0b5cd1ef69161"
    
    results = {}
    
    # Test each API
    results["coingecko"] = await test_coingecko()
    results["binance"] = await test_binance()
    results["cryptocompare"] = await test_cryptocompare()
    results["cryptopanic"] = await test_cryptopanic()
    results["firecrawl"] = await test_firecrawl()
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for api, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{api.upper():15} {status}")
    
    print(f"\nOverall: {passed}/{total} APIs working")
    
    if passed == total:
        print("🎉 All APIs are working correctly!")
    else:
        print("⚠️  Some APIs need attention")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
