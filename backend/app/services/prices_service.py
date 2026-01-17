from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from fastapi import BackgroundTasks, HTTPException

try:
    from clients.coingecko_client import CoinGeckoClient
    from cache import AsyncCache
    from services.smart_cache_service import smart_cache
except ImportError:
    from clients.coingecko_client import CoinGeckoClient
    from cache import AsyncCache
    from services.smart_cache_service import smart_cache


FRESH_TTL_SECONDS = 30
LAST_SUCCESS_TTL_SECONDS = 3600  # keep for 1 hour unless evicted


def _cache_key(ids: str, vs_currencies: str) -> str:
    return f"simple_price:{ids}:{vs_currencies}"


async def _refresh_cache(ids: str, vs_currencies: str, cache: AsyncCache) -> None:
    client = CoinGeckoClient()
    try:
        # Convert string parameters to lists for the client
        ids_list = [id.strip() for id in ids.split(",") if id.strip()]
        vs_currencies_list = [curr.strip() for curr in vs_currencies.split(",") if curr.strip()]
        
        data = await client.get_simple_price(ids=ids_list, vs_currencies=vs_currencies_list)
        await cache.set(_cache_key(ids, vs_currencies), data, ttl_seconds=FRESH_TTL_SECONDS)
        await cache.set(_cache_key(ids, vs_currencies) + ":last_success", data, ttl_seconds=LAST_SUCCESS_TTL_SECONDS)
    finally:
        await client.close()


async def get_simple_price_with_cache(
    *,
    ids: str,
    vs_currencies: str,
    background_tasks: Optional[BackgroundTasks] = None,
) -> Dict[str, Any]:
    # Convert string parameters to lists
    ids_list = [id.strip() for id in ids.split(",") if id.strip()]
    vs_currencies_list = [curr.strip() for curr in vs_currencies.split(",") if curr.strip()]
    
    print(f"[DEBUG] get_simple_price_with_cache called for {ids_list}")
    
    try:
        # Use smart cache service for better data management
        print(f"[DEBUG] Calling smart_cache.get_prices...")
        import asyncio
        data = await asyncio.wait_for(
            smart_cache.get_prices(ids_list, vs_currencies_list),
            timeout=5.0  # 5 second timeout to prevent hanging
        )
        print(f"[DEBUG] smart_cache.get_prices returned {len(data) if data else 0} items")
        
        # Remove cache metadata before returning
        if "_cached_at" in data:
            data.pop("_cached_at")
        if "_data_type" in data:
            data.pop("_data_type")
        if "_cache_key" in data:
            data.pop("_cache_key")
        if "_cache_status" in data:
            cache_status = data.pop("_cache_status")
            if cache_status == "stale":
                print("[INFO] Returning stale cached data (background update in progress)")
            elif cache_status == "empty":
                raise HTTPException(
                    status_code=503,
                    detail="Price cache is empty and live data is unavailable.",
                )
        
        print(f"[SUCCESS] Retrieved prices for {len(data)} cryptocurrencies")
        return data
        
    except asyncio.TimeoutError:
        print(f"[ERROR] Smart cache timed out after 5 seconds, using fallback")
        # Fallback to old method if smart cache times out
        return await _fallback_get_simple_price(ids, vs_currencies, background_tasks)
    except Exception as e:
        print(f"[ERROR] Smart cache error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to old method if smart cache fails
        return await _fallback_get_simple_price(ids, vs_currencies, background_tasks)


async def _fallback_get_simple_price(
    ids: str,
    vs_currencies: str,
    background_tasks: Optional[BackgroundTasks] = None,
) -> Dict[str, Any]:
    """Fallback method using the old cache system."""
    cache = AsyncCache()
    await cache.initialize()

    key = _cache_key(ids, vs_currencies)
    cached = await cache.get(key)
    if cached is not None:
        return cached

    # Convert string parameters to lists for the client
    ids_list = [id.strip() for id in ids.split(",") if id.strip()]
    vs_currencies_list = [curr.strip() for curr in vs_currencies.split(",") if curr.strip()]

    # No fresh cache; try live fetch with better error handling
    client = CoinGeckoClient(timeout_seconds=6.0)
    try:
        print(f"[INFO] Fallback: Fetching live data for: {ids_list}")
        data = await client.get_simple_price(ids=ids_list, vs_currencies=vs_currencies_list)
        print(f"[SUCCESS] Fallback: Successfully fetched data")
        await cache.set(key, data, ttl_seconds=FRESH_TTL_SECONDS)
        await cache.set(key + ":last_success", data, ttl_seconds=LAST_SUCCESS_TTL_SECONDS)
        
        # IMMEDIATELY save to smart cache (disk) - this ensures we always have the latest successful API call
        try:
            from services.smart_cache_service import smart_cache
            # Check if this is major cryptos, save to major_cryptos cache
            major_cryptos = ["bitcoin", "ethereum", "solana", "cardano", "polkadot", 
                            "chainlink", "uniswap", "avalanche-2", "matic-network", "cosmos",
                            "binancecoin"]
            if all(id in major_cryptos for id in ids_list):
                # Use the new immediate save method
                await smart_cache.save_successful_api_response("prices", data, "major_cryptos")
                print("[SUCCESS] Immediately saved successful price data to smart cache (disk)")
        except Exception as cache_error:
            print(f"[WARNING] Failed to save to smart cache: {cache_error}")
        
        return data
    except Exception as e:
        print(f"[ERROR] Fallback: CoinGecko API error: {e}")
        # Always try to return cached data first
        last = await cache.get(key + ":last_success")
        if last is not None:
            print("[INFO] Fallback: Returning cached data")
            return last

        # Try fallback sources like CoinCap to keep real data flowing
        fallback_prices = await _fetch_fallback_prices(ids_list, vs_currencies_list)
        if fallback_prices:
            print("[WARNING] Using CoinCap fallback data for prices")
            await cache.set(key, fallback_prices, ttl_seconds=FRESH_TTL_SECONDS)
            await cache.set(key + ":last_success", fallback_prices, ttl_seconds=LAST_SUCCESS_TTL_SECONDS)
            # Persist fallback data into smart cache for downstream consumers
            fallback_key = f"{','.join(ids_list)}_{','.join(vs_currencies_list)}"
            await smart_cache.save_external_cache("prices", fallback_key, fallback_prices)
            return fallback_prices

        raise HTTPException(
            status_code=503,
            detail="Unable to fetch CoinGecko prices and no cached data is available.",
        )
    finally:
        await client.close()




async def get_market_data_with_cache(
    *,
    ids: str,
    vs_currency: str,
    background_tasks: Optional[BackgroundTasks] = None,
) -> Dict[str, Any]:
    """Get market data including 24h percentage changes from CoinGecko markets API."""
    # Convert string parameters to lists
    ids_list = [id.strip() for id in ids.split(",") if id.strip()]
    
    print(f"[DEBUG] get_market_data_with_cache called for {ids_list}")
    
    try:
        # Use smart cache service for better data management
        print(f"[DEBUG] Calling smart_cache.get_market_data...")
        import asyncio
        data = await asyncio.wait_for(
            smart_cache.get_market_data(ids_list, vs_currency),
            timeout=5.0  # 5 second timeout to prevent hanging
        )
        print(f"[DEBUG] smart_cache.get_market_data returned {len(data) if data else 0} items")
        
        # Remove cache metadata before returning
        if "_cached_at" in data:
            data.pop("_cached_at")
        if "_data_type" in data:
            data.pop("_data_type")
        if "_cache_key" in data:
            data.pop("_cache_key")
        if "_cache_status" in data:
            cache_status = data.pop("_cache_status")
            if cache_status == "stale":
                print("[INFO] Returning stale market data (background update in progress)")
            elif cache_status == "empty":
                raise HTTPException(
                    status_code=503,
                    detail="Market cache is empty and live data is unavailable.",
                )
        
        print(f"[SUCCESS] Retrieved market data for {len(data)} cryptocurrencies")
        return data
        
    except asyncio.TimeoutError:
        print(f"[ERROR] Smart cache timed out after 5 seconds, using fallback")
        # Fallback to old method if smart cache times out
        return await _fallback_get_market_data(ids, vs_currency, background_tasks)
    except Exception as e:
        print(f"[ERROR] Smart cache error for market data: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to old method if smart cache fails
        return await _fallback_get_market_data(ids, vs_currency, background_tasks)


async def _fallback_get_market_data(
    ids: str,
    vs_currency: str,
    background_tasks: Optional[BackgroundTasks] = None,
) -> Dict[str, Any]:
    """Fallback method for market data using the old cache system."""
    cache = AsyncCache()
    await cache.initialize()

    key = f"market_data:{ids}:{vs_currency}"
    cached = await cache.get(key)
    if cached is not None:
        return cached

    # Convert string parameters to lists for the client
    ids_list = [id.strip() for id in ids.split(",") if id.strip()]

    # Try to get market data with 24h changes
    client = CoinGeckoClient(timeout_seconds=6.0)
    try:
        print(f"[INFO] Fallback: Fetching market data for: {ids_list}")
        data = await client.get_coins_markets(
            vs_currency=[vs_currency],
            ids=ids_list,
            price_change_percentage=["24h"]
        )
        
        # Transform to match our expected format
        result = {}
        for coin in data:
            coin_id = coin.get("id")
            if coin_id:
                result[coin_id] = {
                    "price": coin.get("current_price", 0.0),
                    "price_change_24h": coin.get("price_change_percentage_24h", 0.0),
                    "market_cap": coin.get("market_cap", 0.0),
                    "volume_24h": coin.get("total_volume", 0.0),
                    "symbol": coin.get("symbol", "").upper(),
                    "name": coin.get("name", ""),
                }
        
        print(f"[SUCCESS] Fallback: Market data fetched: {len(result)} coins")
        await cache.set(key, result, ttl_seconds=FRESH_TTL_SECONDS)
        await cache.set(key + ":last_success", result, ttl_seconds=LAST_SUCCESS_TTL_SECONDS)
        
        # IMMEDIATELY save to smart cache (disk) - this ensures we always have the latest successful API call
        try:
            from services.smart_cache_service import smart_cache
            # Check if this is major cryptos, save to major_cryptos cache
            major_cryptos = ["bitcoin", "ethereum", "solana", "cardano", "polkadot", 
                            "chainlink", "uniswap", "avalanche-2", "matic-network", "cosmos",
                            "binancecoin"]
            if all(id in major_cryptos for id in ids_list):
                # Use the new immediate save method
                await smart_cache.save_successful_api_response("market_data", result, "major_cryptos")
                print("[SUCCESS] Immediately saved successful market data to smart cache (disk)")
        except Exception as cache_error:
            print(f"[WARNING] Failed to save to smart cache: {cache_error}")
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Fallback: Market data API error: {e}")
        # Fallback to cached data
        last = await cache.get(key + ":last_success")
        if last is not None:
            print("[INFO] Fallback: Returning cached market data")
            return last

        fallback_market = await _fetch_fallback_market_data(ids_list)
        if fallback_market:
            print("[WARNING] Using CoinCap fallback data for market metrics")
            await cache.set(key, fallback_market, ttl_seconds=FRESH_TTL_SECONDS)
            await cache.set(key + ":last_success", fallback_market, ttl_seconds=LAST_SUCCESS_TTL_SECONDS)
            fallback_key = f"{','.join(ids_list)}_{vs_currency}"
            await smart_cache.save_external_cache("market_data", fallback_key, fallback_market)
            return fallback_market
        
        raise HTTPException(
            status_code=503,
            detail="Unable to fetch CoinGecko market data and no cached data is available.",
        )
    finally:
        await client.close()


async def _refresh_market_cache(ids: str, vs_currency: str, cache: AsyncCache) -> None:
    """Background task to refresh market data cache."""
    client = CoinGeckoClient()
    try:
        ids_list = [id.strip() for id in ids.split(",") if id.strip()]
        data = await client.get_coins_markets(
            vs_currency=[vs_currency],
            ids=ids_list,
            price_change_percentage=["24h"]
        )
        
        result = {}
        for coin in data:
            coin_id = coin.get("id")
            if coin_id:
                result[coin_id] = {
                    "price": coin.get("current_price", 0.0),
                    "price_change_24h": coin.get("price_change_percentage_24h", 0.0),
                    "market_cap": coin.get("market_cap", 0.0),
                    "volume_24h": coin.get("total_volume", 0.0),
                    "symbol": coin.get("symbol", "").upper(),
                    "name": coin.get("name", ""),
                }
        
        key = f"market_data:{ids}:{vs_currency}"
        await cache.set(key, result, ttl_seconds=FRESH_TTL_SECONDS)
        await cache.set(key + ":last_success", result, ttl_seconds=LAST_SUCCESS_TTL_SECONDS)
    finally:
        await client.close()


async def _fetch_fallback_prices(ids_list: list[str], vs_currencies_list: list[str]) -> Optional[Dict[str, Any]]:
    """Fetch price data from CoinCap as a fallback when CoinGecko is unavailable."""
    if not ids_list or vs_currencies_list != ["usd"]:
        return None

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(
                "https://api.coincap.io/v2/assets",
                params={"ids": ",".join(ids_list)},
                headers={"User-Agent": "CryptoForecast/1.0"},
            )
            response.raise_for_status()
            payload = response.json()
    except Exception as error:
        print(f"[ERROR] CoinCap fallback price fetch failed: {error}")
        return None

    data = {}
    for asset in payload.get("data", []):
        asset_id = asset.get("id")
        price = asset.get("priceUsd")
        if not asset_id or price is None:
            continue
        try:
            data[asset_id] = {"usd": round(float(price), 6)}
        except (TypeError, ValueError):
            continue
    return data or None


async def _fetch_fallback_market_data(ids_list: list[str]) -> Optional[Dict[str, Any]]:
    """Fetch market metrics (price, change, cap, volume) from CoinCap when CoinGecko is down."""
    if not ids_list:
        return None

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(
                "https://api.coincap.io/v2/assets",
                params={"ids": ",".join(ids_list)},
                headers={"User-Agent": "CryptoForecast/1.0"},
            )
            response.raise_for_status()
            payload = response.json()
    except Exception as error:
        print(f"[ERROR] CoinCap fallback market fetch failed: {error}")
        return None

    market_snapshot: Dict[str, Any] = {}
    for asset in payload.get("data", []):
        asset_id = asset.get("id")
        if not asset_id:
            continue
        try:
            market_snapshot[asset_id] = {
                "price": round(float(asset.get("priceUsd", 0.0)), 6),
                "price_change_24h": round(float(asset.get("changePercent24Hr", 0.0)), 4),
                "market_cap": round(float(asset.get("marketCapUsd", 0.0))),
                "volume_24h": round(float(asset.get("volumeUsd24Hr", 0.0))),
                "symbol": asset.get("symbol", asset_id[:5]).upper(),
                "name": asset.get("name", asset_id.title()),
            }
        except (TypeError, ValueError):
            continue

    return market_snapshot or None


