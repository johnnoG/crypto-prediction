from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import aiofiles
import httpx

try:
    from clients.rate_limited_coingecko_client import RateLimitedCoinGeckoClient
    from services.rate_limit_manager import RequestPriority
    from config import get_settings
except ImportError:
    from clients.rate_limited_coingecko_client import RateLimitedCoinGeckoClient
    from services.rate_limit_manager import RequestPriority
    from config import get_settings


class SmartCacheService:
    """
    Smart persistent cache service that:
    1. Saves last good API data to disk
    2. Intelligently updates cached data in background
    3. Provides fresh data when possible, fallback when needed
    4. Tracks data freshness and age
    """
    
    def __init__(self):
        self.settings = get_settings()
        # Use absolute path relative to backend/app directory
        # __file__ is at backend/app/services/smart_cache_service.py
        # So parent.parent = backend/app/
        backend_app_dir = Path(__file__).parent.parent.resolve()
        self.cache_dir = (backend_app_dir / "data" / "cache").resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Smart cache directory: {self.cache_dir}")

        # Cache TTL settings (in seconds)
        self.ttl_settings = {
            "prices": 300,      # 5 minutes for prices
            "market_data": 600, # 10 minutes for market data
            "news": 1800,       # 30 minutes for news
            "forecasts": 3600,  # 1 hour for forecasts
        }
        
        # Background update intervals (in seconds)
        self.update_intervals = {
            "prices": 120,      # Try to update every 2 minutes
            "market_data": 300, # Try to update every 5 minutes
            "news": 900,        # Try to update every 15 minutes
            "forecasts": 1800,  # Try to update every 30 minutes
        }
        
        self._background_tasks: Dict[str, asyncio.Task] = {}
        self._last_update_attempts: Dict[str, datetime] = {}
        self._initialized = False

    @staticmethod
    def _strip_metadata(cache_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Remove metadata keys (_cached_at, etc.) from cached payloads."""
        if not cache_payload or not isinstance(cache_payload, dict):
            return {}
        return {
            key: value
            for key, value in cache_payload.items()
            if not key.startswith("_")
        }
    
    async def initialize(self):
        """Initialize the cache service and start background tasks."""
        if self._initialized:
            return
        
        self._initialized = True
        
        print("[INFO] Smart cache initialized, skipping initial data fetch (will use existing cache)")
        
        # Start background update tasks only
        for data_type in self.ttl_settings.keys():
            if data_type not in self._background_tasks:
                self._background_tasks[data_type] = asyncio.create_task(
                    self._background_update_loop(data_type)
                )
    
    async def cleanup(self):
        """Cleanup background tasks."""
        for task in self._background_tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._background_tasks.values(), return_exceptions=True)
        self._background_tasks.clear()
        self._last_update_attempts.clear()
        self._initialized = False
    
    def _get_cache_file_path(self, data_type: str, key: str = "default") -> Path:
        """Get the file path for cached data."""
        return self.cache_dir / f"{data_type}_{key}.json"
    
    async def _load_cached_data(self, data_type: str, key: str = "default") -> Optional[Dict[str, Any]]:
        """Load cached data from disk."""
        cache_file = self._get_cache_file_path(data_type, key)
        
        if not cache_file.exists():
            print(f"[INFO] No cache file found: {cache_file}")
            return None
        
        try:
            # Use timeout to prevent hanging on slow disk I/O
            async with aiofiles.open(cache_file, 'r') as f:
                content = await asyncio.wait_for(f.read(), timeout=2.0)
                data = json.loads(content)
                
                # Check if data is still fresh
                if self._is_data_fresh(data, data_type):
                    print(f"[SUCCESS] Loaded fresh cached {data_type} from disk")
                    return data
                else:
                    # Data is stale, but we'll return it as fallback
                    data["_cache_status"] = "stale"
                    print(f"[INFO] Loaded stale cached {data_type} from disk")
                    return data
                    
        except asyncio.TimeoutError:
            print(f"[ERROR] Timeout loading cached {data_type} from disk")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to load cached {data_type}: {e}")
            return None
    
    async def _save_cached_data(self, data_type: str, data: Dict[str, Any], key: str = "default"):
        """Save data to disk cache."""
        cache_file = self._get_cache_file_path(data_type, key)
        
        # Add metadata
        cache_data = {
            "_cached_at": datetime.utcnow().isoformat(),
            "_data_type": data_type,
            "_cache_key": key,
            **data
        }
        
        try:
            async with aiofiles.open(cache_file, 'w') as f:
                await f.write(json.dumps(cache_data, indent=2))
            print(f"[SUCCESS] Cached {data_type} data to disk")
        except Exception as e:
            print(f"[ERROR] Failed to save cached {data_type}: {e}")
    
    async def save_external_cache(self, data_type: str, key: str, data: Dict[str, Any]):
        """Public helper to persist externally fetched data into the smart cache."""
        await self._save_cached_data(data_type, data, key)

    def _is_data_fresh(self, data: Dict[str, Any], data_type: str) -> bool:
        """Check if cached data is still fresh."""
        if "_cached_at" not in data:
            return False
        
        try:
            cached_at = datetime.fromisoformat(data["_cached_at"])
            ttl = self.ttl_settings.get(data_type, 300)
            return datetime.utcnow() - cached_at < timedelta(seconds=ttl)
        except:
            return False
    
    def _should_attempt_update(self, data_type: str) -> bool:
        """Check if we should attempt to update cached data."""
        if data_type not in self._last_update_attempts:
            return True
        
        last_attempt = self._last_update_attempts[data_type]
        interval = self.update_intervals.get(data_type, 300)
        return datetime.utcnow() - last_attempt > timedelta(seconds=interval)
    
    async def _background_update_loop(self, data_type: str):
        """Background loop to keep cached data updated."""
        while True:
            try:
                if self._should_attempt_update(data_type):
                    await self._update_cached_data(data_type)
                    self._last_update_attempts[data_type] = datetime.utcnow()
                
                # Wait before next update attempt
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[ERROR] Background update error for {data_type}: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _update_cached_data(self, data_type: str):
        """Update cached data for a specific type."""
        try:
            if data_type == "prices":
                await self._update_prices_cache()
            elif data_type == "market_data":
                await self._update_market_data_cache()
            elif data_type == "news":
                await self._update_news_cache()
            elif data_type == "forecasts":
                await self._update_forecasts_cache()
        except Exception as e:
            print(f"[ERROR] Failed to update {data_type} cache: {e}")
    
    async def _update_prices_cache(self):
        """Update prices cache with fresh data. ALWAYS saves successful API responses."""
        client = RateLimitedCoinGeckoClient(timeout_seconds=6.0)
        major_ids = [
            "bitcoin",
            "ethereum",
            "solana",
            "cardano",
            "polkadot",
            "chainlink",
            "uniswap",
            "avalanche-2",
            "matic-network",
            "cosmos",
            "binancecoin",
        ]
        existing = self._strip_metadata(await self._load_cached_data("prices", "major_cryptos"))
        
        # Start with existing cache as base
        data = existing.copy() if existing else {}
        
        try:
            # Fetch fresh data (may be partial due to rate limits)
            fresh_data = await self._fetch_prices_in_chunks(
                major_ids,
                vs_currencies=["usd"],
                client=client,
            )
            
            # ALWAYS merge successful API responses into cache, even if partial
            if fresh_data:
                data.update(fresh_data)
                print(f"[SUCCESS] Fetched fresh price data for {len(fresh_data)} cryptocurrencies")
            
            # Fill missing coins from existing cache
            if existing:
                missing = [coin for coin in major_ids if coin not in data]
                if missing:
                    print(f"[INFO] Using cached data for {len(missing)} missing coins: {missing}")
                    for coin in missing:
                        if coin in existing:
                            data[coin] = existing[coin]
            
            # ALWAYS save the cache, even if partial - this ensures we have the latest successful API call
            if data:
                await self._save_cached_data("prices", data, "major_cryptos")
                print(f"[SUCCESS] Updated prices cache with {len(data)} cryptocurrencies (fresh: {len(fresh_data) if fresh_data else 0}, cached: {len(data) - len(fresh_data) if fresh_data else len(data)})")
            else:
                print("[WARNING] No price data to cache (all API calls failed and no existing cache)")
                
        except Exception as e:
            print(f"[ERROR] Exception during prices cache update: {e}")
            # Even on error, save what we have (existing cache or partial fresh data)
            if data:
                await self._save_cached_data("prices", data, "major_cryptos")
                print(f"[INFO] Saved existing/partial prices cache ({len(data)} coins) after error")
            else:
                # Last resort: try to load and preserve existing cache
                existing_raw = await self._load_cached_data("prices", "major_cryptos")
                if existing_raw and not existing_raw.get("error"):
                    existing_clean = self._strip_metadata(existing_raw)
                    if existing_clean:
                        await self._save_cached_data("prices", existing_clean, "major_cryptos")
                        print("[INFO] Preserved existing prices cache after error")
        finally:
            await client.close()
    
    async def _update_market_data_cache(self):
        """Update market data cache with fresh data. ALWAYS saves successful API responses."""
        client = RateLimitedCoinGeckoClient(timeout_seconds=6.0)
        major_ids = [
            "bitcoin",
            "ethereum",
            "solana",
            "cardano",
            "polkadot",
            "chainlink",
            "uniswap",
            "avalanche-2",
            "matic-network",
            "cosmos",
            "binancecoin",
        ]
        existing = self._strip_metadata(await self._load_cached_data("market_data", "major_cryptos"))
        
        # Start with existing cache as base
        result = existing.copy() if existing else {}
        
        try:
            # Fetch fresh data (may be partial due to rate limits)
            fresh_data = await self._fetch_market_data_in_chunks(
                major_ids,
                client=client,
            )
            
            # ALWAYS merge successful API responses into cache, even if partial
            if fresh_data:
                result.update(fresh_data)
                print(f"[SUCCESS] Fetched fresh market data for {len(fresh_data)} cryptocurrencies")
            
            # Fill missing coins from existing cache
            if existing:
                missing = [coin for coin in major_ids if coin not in result]
                if missing:
                    print(f"[INFO] Using cached data for {len(missing)} missing coins: {missing}")
                    for coin in missing:
                        if coin in existing:
                            result[coin] = existing[coin]
            
            # ALWAYS save the cache, even if partial - this ensures we have the latest successful API call
            if result:
                await self._save_cached_data("market_data", result, "major_cryptos")
                print(f"[SUCCESS] Updated market data cache with {len(result)} cryptocurrencies (fresh: {len(fresh_data) if fresh_data else 0}, cached: {len(result) - len(fresh_data) if fresh_data else len(result)})")
            else:
                print("[WARNING] No market data to cache (all API calls failed and no existing cache)")
                
        except Exception as e:
            print(f"[ERROR] Exception during market data cache update: {e}")
            # Even on error, save what we have (existing cache or partial fresh data)
            if result:
                await self._save_cached_data("market_data", result, "major_cryptos")
                print(f"[INFO] Saved existing/partial market data cache ({len(result)} coins) after error")
            else:
                # Last resort: try to load and preserve existing cache
                existing_raw = await self._load_cached_data("market_data", "major_cryptos")
                if existing_raw and not existing_raw.get("error"):
                    existing_clean = self._strip_metadata(existing_raw)
                    if existing_clean:
                        await self._save_cached_data("market_data", existing_clean, "major_cryptos")
                        print("[INFO] Preserved existing market data cache after error")
        finally:
            await client.close()

    async def _fetch_prices_in_chunks(
        self,
        ids: list[str],
        vs_currencies: list[str] | None = None,
        chunk_size: int = 3,
        client: Optional[RateLimitedCoinGeckoClient] = None,
    ) -> Dict[str, Any]:
        if vs_currencies is None:
            vs_currencies = ["usd"]
        results: Dict[str, Any] = {}
        chunk_size = max(1, min(chunk_size, 3, len(ids)))
        chunks = list(self._chunk_list(ids, chunk_size))

        if client is not None:
            for chunk in chunks:
                for attempt in range(3):
                    try:
                        chunk_data = await client.get_simple_price(
                            ids=chunk,
                            vs_currencies=vs_currencies,
                            priority=RequestPriority.HIGH,
                        )
                        results.update(chunk_data)
                        
                        # IMMEDIATELY save successful chunk to cache (even if partial)
                        if chunk_data:
                            # Check if this is major cryptos
                            major_cryptos = ["bitcoin", "ethereum", "solana", "cardano", "polkadot", 
                                            "chainlink", "uniswap", "avalanche-2", "matic-network", "cosmos",
                                            "binancecoin"]
                            if all(cid in major_cryptos for cid in chunk):
                                await self.save_successful_api_response("prices", chunk_data, "major_cryptos")
                        
                        await asyncio.sleep(1.2)
                        break
                    except Exception as chunk_error:
                        message = str(chunk_error).lower()
                        if "429" in message or "too many requests" in message:
                            wait_time = 5 * (attempt + 1)
                            print(f"[RATE LIMIT] Price chunk {chunk} attempt {attempt+1}/3 waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        print(f"[ERROR] Rate-limited price fetch failed for {chunk}: {chunk_error}")
                        await asyncio.sleep(1.0)
                        break
            return results

        async with httpx.AsyncClient(timeout=10.0) as direct_client:
            for chunk in chunks:
                for attempt in range(3):
                    try:
                        response = await direct_client.get(
                            "https://api.coingecko.com/api/v3/simple/price",
                            params={
                                "ids": ",".join(chunk),
                                "vs_currencies": ",".join(vs_currencies),
                            },
                        )
                        response.raise_for_status()
                        chunk_data = response.json()
                        results.update(chunk_data)
                        
                        # IMMEDIATELY save successful chunk to cache (even if partial)
                        if chunk_data:
                            # Check if this is major cryptos
                            major_cryptos = ["bitcoin", "ethereum", "solana", "cardano", "polkadot", 
                                            "chainlink", "uniswap", "avalanche-2", "matic-network", "cosmos",
                                            "binancecoin"]
                            if all(cid in major_cryptos for cid in chunk):
                                await self.save_successful_api_response("prices", chunk_data, "major_cryptos")
                        
                        await asyncio.sleep(1.2)
                        break
                    except Exception as chunk_error:
                        message = str(chunk_error).lower()
                        if "429" in message or "too many requests" in message:
                            wait_time = 5 * (attempt + 1)
                            print(f"[RATE LIMIT] Direct price chunk {chunk} attempt {attempt+1}/3 waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        print(f"[ERROR] Chunk price fetch failed for {chunk}: {chunk_error}")
                        await asyncio.sleep(1.0)
                        break
        return results

    async def _fetch_market_data_in_chunks(
        self,
        ids: list[str],
        chunk_size: int = 3,
        client: Optional[RateLimitedCoinGeckoClient] = None,
        vs_currency: str = "usd",
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        chunk_size = max(1, min(chunk_size, 3, len(ids)))
        chunks = list(self._chunk_list(ids, chunk_size))

        if client is not None:
            for chunk in chunks:
                for attempt in range(3):
                    try:
                        chunk_data = await client.get_coins_markets(
                            vs_currency=vs_currency,
                            ids=chunk,
                            price_change_percentage=["24h"],
                            priority=RequestPriority.HIGH,
                        )
                        transformed = self._transform_market_data(chunk_data)
                        results.update(transformed)
                        
                        # IMMEDIATELY save successful chunk to cache (even if partial)
                        if transformed:
                            # Check if this is major cryptos
                            major_cryptos = ["bitcoin", "ethereum", "solana", "cardano", "polkadot", 
                                            "chainlink", "uniswap", "avalanche-2", "matic-network", "cosmos",
                                            "binancecoin"]
                            if all(cid in major_cryptos for cid in chunk):
                                await self.save_successful_api_response("market_data", transformed, "major_cryptos")
                        
                        await asyncio.sleep(1.2)
                        break
                    except Exception as chunk_error:
                        message = str(chunk_error).lower()
                        if "429" in message or "too many requests" in message:
                            wait_time = 5 * (attempt + 1)
                            print(f"[RATE LIMIT] Market chunk {chunk} attempt {attempt+1}/3 waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        print(f"[ERROR] Rate-limited market data fetch failed for {chunk}: {chunk_error}")
                        await asyncio.sleep(1.0)
                        break
            return results

        async with httpx.AsyncClient(timeout=10.0) as direct_client:
            for chunk in chunks:
                for attempt in range(3):
                    try:
                        response = await direct_client.get(
                            "https://api.coingecko.com/api/v3/coins/markets",
                            params={
                                "vs_currency": vs_currency,
                                "ids": ",".join(chunk),
                                "price_change_percentage": "24h",
                                "order": "market_cap_desc",
                                "per_page": len(chunk),
                                "page": 1,
                                "sparkline": "false",
                            },
                        )
                        response.raise_for_status()
                        chunk_data = response.json()
                        transformed = self._transform_market_data(chunk_data)
                        results.update(transformed)
                        
                        # IMMEDIATELY save successful chunk to cache (even if partial)
                        if transformed:
                            # Check if this is major cryptos
                            major_cryptos = ["bitcoin", "ethereum", "solana", "cardano", "polkadot", 
                                            "chainlink", "uniswap", "avalanche-2", "matic-network", "cosmos",
                                            "binancecoin"]
                            if all(cid in major_cryptos for cid in chunk):
                                await self.save_successful_api_response("market_data", transformed, "major_cryptos")
                        
                        await asyncio.sleep(1.2)
                        break
                    except Exception as chunk_error:
                        message = str(chunk_error).lower()
                        if "429" in message or "too many requests" in message:
                            wait_time = 5 * (attempt + 1)
                            print(f"[RATE LIMIT] Direct market chunk {chunk} attempt {attempt+1}/3 waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        print(f"[ERROR] Chunk market data fetch failed for {chunk}: {chunk_error}")
                        await asyncio.sleep(1.0)
                        break
        return results

    @staticmethod
    def _chunk_list(items: list[str], size: int):
        for i in range(0, len(items), size):
            yield items[i:i + size]

    @staticmethod
    def _transform_market_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
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
        return result
    
    async def _update_news_cache(self):
        """Update news cache with fresh data."""
        print("[INFO] News cache update skipped until real news ingestion is implemented.")
    
    async def _update_forecasts_cache(self):
        """Update forecasts cache with fresh data."""
        print("[INFO] Forecast cache update skipped until real model outputs are available.")
    
    async def get_smart_data(self, data_type: str, key: str = "default", 
                           fallback_func: Optional[callable] = None) -> Dict[str, Any]:
        """
        Get data with smart fallback strategy:
        1. Try to get fresh cached data
        2. If stale, try to update immediately (not just background)
        3. Return best available data
        4. ALWAYS save successful API responses immediately
        """
        # First, try to get cached data
        cached_data = await self._load_cached_data(data_type, key)
        
        if cached_data and cached_data.get("_cache_status") != "stale":
            # We have fresh cached data
            return cached_data
        
        # If we have stale data, try to update immediately (not just background)
        if cached_data:
            # Try immediate update, but don't block on it
            asyncio.create_task(self._update_cached_data(data_type))
            # Return stale data immediately
            return cached_data
        
        # No cached data, try to get fresh data immediately
        if fallback_func:
            try:
                fresh_data = await fallback_func()
                if fresh_data:
                    # IMMEDIATELY save successful API response
                    await self._save_cached_data(data_type, fresh_data, key)
                    print(f"[SUCCESS] Saved fresh {data_type} data to cache immediately")
                    return fresh_data
            except Exception as e:
                print(f"[ERROR] Fallback function failed for {data_type}: {e}")
        
        # Last resort: return empty data
        return {"error": f"No data available for {data_type}", "_cache_status": "empty"}
    
    async def save_successful_api_response(self, data_type: str, data: Dict[str, Any], key: str = "default"):
        """
        IMMEDIATELY save a successful API response to cache.
        This is called whenever an API call succeeds, even if partial.
        
        Args:
            data_type: Type of data (prices, market_data, etc.)
            data: The successful API response data
            key: Cache key (e.g., "major_cryptos")
        """
        if not data:
            return
        
        try:
            # Load existing cache to merge with
            existing = await self._load_cached_data(data_type, key)
            existing_clean = self._strip_metadata(existing) if existing else {}
            
            # Merge fresh data with existing (fresh data takes priority)
            if isinstance(data, dict) and isinstance(existing_clean, dict):
                # If data is a dict of coins, merge them
                if any(isinstance(v, dict) for v in data.values()):
                    existing_clean.update(data)
                    merged_data = existing_clean
                else:
                    # Single item, just use fresh data
                    merged_data = data
            else:
                merged_data = data
            
            # IMMEDIATELY save to disk
            await self._save_cached_data(data_type, merged_data, key)
            print(f"[SUCCESS] Immediately saved successful {data_type} API response to cache ({len(merged_data) if isinstance(merged_data, dict) else 1} items)")
            
        except Exception as e:
            print(f"[ERROR] Failed to save successful API response: {e}")
    
    async def get_prices(self, ids: list, vs_currencies: list = None) -> Dict[str, Any]:
        """Get prices with smart caching."""
        if vs_currencies is None:
            vs_currencies = ["usd"]
        
        # Check if this is a subset of major cryptos, use major_cryptos cache
        major_cryptos = ["bitcoin", "ethereum", "solana", "cardano", "polkadot", 
                        "chainlink", "uniswap", "avalanche-2", "matic-network", "cosmos",
                        "binancecoin"]
        
        if all(id in major_cryptos for id in ids):
            # Use major cryptos cache - PRIORITIZE SPEED
            data = await self._load_cached_data("prices", "major_cryptos")
            
            # If we have ANY cached data (even if stale), use it immediately
            if data and isinstance(data, dict):
                clean_data = self._strip_metadata(data)
                if clean_data and not clean_data.get("error"):
                    # Filter to only requested IDs
                    filtered_data = {}
                    for id in ids:
                        if id in clean_data and isinstance(clean_data[id], dict):
                            filtered_data[id] = clean_data[id]
                        else:
                            # Default to zero if missing
                            filtered_data[id] = {curr: 0.0 for curr in vs_currencies}
                    
                    # Trigger background update if data is stale (non-blocking)
                    if data.get("_cache_status") == "stale":
                        asyncio.create_task(self._update_prices_cache())
                    
                    print(f"[SUCCESS] Returned cached prices for {len(filtered_data)} cryptos instantly")
                    return filtered_data
            
            # No cache available, try to fetch (but with timeout)
            print("[INFO] No cached prices available, attempting fresh fetch...")
            try:
                async def fetch_prices_quick():
                    return await asyncio.wait_for(
                        self._fetch_prices_in_chunks(ids, vs_currencies),
                        timeout=5.0  # 5 second timeout
                    )
                return await fetch_prices_quick()
            except asyncio.TimeoutError:
                print("[WARNING] Fresh price fetch timed out, returning empty data")
                return {id: {curr: 0.0 for curr in vs_currencies} for id in ids}
        
        # For other combinations, use specific cache key
        async def fetch_prices():
            return await self._fetch_prices_in_chunks(ids, vs_currencies)
        
        return await self.get_smart_data("prices", f"{','.join(ids)}_{','.join(vs_currencies)}", fetch_prices)
    
    async def get_market_data(self, ids: list, vs_currency: str = "usd") -> Dict[str, Any]:
        """Get market data with smart caching."""
        # Check if this is a subset of major cryptos, use major_cryptos cache
        major_cryptos = ["bitcoin", "ethereum", "solana", "cardano", "polkadot", 
                        "chainlink", "uniswap", "avalanche-2", "matic-network", "cosmos",
                        "binancecoin"]
        
        if all(id in major_cryptos for id in ids):
            # Use major cryptos cache - PRIORITIZE SPEED
            data = await self._load_cached_data("market_data", "major_cryptos")
            
            # If we have ANY cached data (even if stale), use it immediately
            if data and isinstance(data, dict):
                clean_data = self._strip_metadata(data)
                if clean_data and not clean_data.get("error"):
                    # Filter to only requested IDs, preserving existing data structure
                    filtered_data = {}
                    for id in ids:
                        if id in clean_data and isinstance(clean_data[id], dict):
                            # Use cached data if available
                            filtered_data[id] = clean_data[id]
                        else:
                            # Default structure if missing
                            filtered_data[id] = {
                                "price": 0.0,
                                "price_change_24h": 0.0,
                                "market_cap": 0.0,
                                "volume_24h": 0.0,
                                "symbol": id.upper(),
                                "name": id.title(),
                            }
                    
                    # Trigger background update if data is stale (non-blocking)
                    if data.get("_cache_status") == "stale":
                        asyncio.create_task(self._update_market_data_cache())
                    
                    print(f"[SUCCESS] Returned cached market data for {len(filtered_data)} cryptos instantly")
                    return filtered_data
            
            # No cache available, try to fetch (but with timeout)
            print("[INFO] No cached market data available, attempting fresh fetch...")
            try:
                async def fetch_market_quick():
                    return await asyncio.wait_for(
                        self._fetch_market_data_in_chunks(ids, vs_currency=vs_currency),
                        timeout=5.0  # 5 second timeout
                    )
                return await fetch_market_quick()
            except asyncio.TimeoutError:
                print("[WARNING] Fresh market data fetch timed out, returning default data")
                return {
                    id: {
                        "price": 0.0,
                        "price_change_24h": 0.0,
                        "market_cap": 0.0,
                        "volume_24h": 0.0,
                        "symbol": id.upper(),
                        "name": id.title(),
                    }
                    for id in ids
                }
        
        # For other combinations, use specific cache key
        async def fetch_market_data():
            return await self._fetch_market_data_in_chunks(ids, vs_currency=vs_currency)
        
        return await self.get_smart_data("market_data", f"{','.join(ids)}_{vs_currency}", fetch_market_data)
    
    async def get_news(self, page: int = 1, limit: int = 24) -> Dict[str, Any]:
        """Get news with smart caching."""
        async def fetch_news():
            # In production, integrate with your news service
            return {
                "items": [],
                "page": page,
                "limit": limit,
                "total": 0,
                "total_pages": 0,
                "realtime": True
            }
        
        return await self.get_smart_data("news", f"page_{page}_limit_{limit}", fetch_news)
    
    async def get_forecasts(self, ids: list, days: int = 7, model: str = "baseline") -> Dict[str, Any]:
        """Get forecasts with smart caching."""
        async def fetch_forecasts():
            # In production, run your ML models
            return {
                "forecasts": {},
                "metadata": {
                    "generated_at": datetime.utcnow().isoformat(),
                    "model": model,
                    "forecast_horizon": days,
                    "total_assets": len(ids)
                }
            }
        
        return await self.get_smart_data("forecasts", f"{','.join(ids)}_{days}d_{model}", fetch_forecasts)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "cache_dir": str(self.cache_dir),
            "cache_files": [],
            "background_tasks": len(self._background_tasks),
            "last_update_attempts": {}
        }
        
        # Check cache files
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    stats["cache_files"].append({
                        "file": cache_file.name,
                        "cached_at": data.get("_cached_at"),
                        "data_type": data.get("_data_type"),
                        "size_kb": round(cache_file.stat().st_size / 1024, 2)
                    })
            except:
                pass
        
        # Add last update attempts
        for data_type, last_attempt in self._last_update_attempts.items():
            stats["last_update_attempts"][data_type] = last_attempt.isoformat()
        
        return stats
    
    async def _ensure_initial_data(self):
        """Ensure we have initial data available for all data types."""
        print("[INFO] Ensuring initial data is available...")
        
        # Check if we have cached data, if not create initial fallback data
        # Use timeouts to prevent hanging on API calls
        for data_type in ["prices", "market_data"]:
            cache_file = self._get_cache_file_path(data_type, "major_cryptos")
            if not cache_file.exists():
                print(f"[INFO] No cached {data_type} found, attempting to fetch...")
                try:
                    # Use timeout to prevent hanging
                    if data_type == "prices":
                        await asyncio.wait_for(self._update_prices_cache(), timeout=10.0)
                    elif data_type == "market_data":
                        await asyncio.wait_for(self._update_market_data_cache(), timeout=10.0)
                except asyncio.TimeoutError:
                    print(f"[WARNING] Initial {data_type} fetch timed out, will use empty cache")
                except Exception as e:
                    print(f"[WARNING] Failed to create initial {data_type}: {e}, will use empty cache")
            else:
                print(f"[INFO] Found existing cache for {data_type}, skipping fetch")
        
        print("[SUCCESS] Initial data check complete")


# Global instance
smart_cache = SmartCacheService()
