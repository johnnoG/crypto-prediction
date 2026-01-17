"""
Rate-Limited CoinGecko Client

Wraps the standard CoinGecko client with intelligent rate limiting,
request batching, and priority queuing.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Union
import time

try:
    from .coingecko_client import CoinGeckoClient
    from ..services.rate_limit_manager import rate_limit_manager, RequestPriority
    from ..services.request_batcher import request_batcher
except ImportError:
    from clients.coingecko_client import CoinGeckoClient
    from services.rate_limit_manager import rate_limit_manager, RequestPriority
    from services.request_batcher import request_batcher


class RateLimitedCoinGeckoClient:
    """
    CoinGecko client with automatic rate limiting and request batching.
    
    Features:
    - Automatic rate limit management
    - Request batching for bulk operations
    - Priority-based queuing
    - Statistics tracking
    """
    
    def __init__(self, timeout_seconds: float = 8.0):
        self._client = CoinGeckoClient(timeout_seconds=timeout_seconds)
        self._api_name = "coingecko"
    
    async def close(self):
        """Close the underlying client."""
        await self._client.close()
    
    async def get_simple_price(
        self,
        ids: List[str],
        vs_currencies: List[str] = None,
        priority: RequestPriority = RequestPriority.MEDIUM
    ) -> Dict[str, Any]:
        """
        Get current prices with automatic rate limiting and batching.
        
        Args:
            ids: List of coin IDs
            vs_currencies: List of currencies (default: ["usd"])
            priority: Request priority (CRITICAL, HIGH, MEDIUM, LOW)
        """
        if vs_currencies is None:
            vs_currencies = ["usd"]
        
        # Wait for rate limit if needed
        await rate_limit_manager.wait_if_needed(self._api_name, priority)
        
        # Track request timing
        start_time = time.time()
        success = False
        rate_limited = False
        
        try:
            # Execute request
            result = await self._client.get_simple_price(ids, vs_currencies)
            success = True
            return result
            
        except Exception as e:
            # Check if rate limited
            if '429' in str(e) or 'rate limit' in str(e).lower():
                rate_limited = True
            raise
            
        finally:
            # Record metrics
            duration_ms = (time.time() - start_time) * 1000
            rate_limit_manager.record_request(
                api_name=self._api_name,
                duration_ms=duration_ms,
                success=success,
                priority=priority,
                rate_limited=rate_limited
            )
    
    async def get_simple_price_batched(
        self,
        ids: List[str],
        vs_currencies: List[str] = None,
        priority: RequestPriority = RequestPriority.MEDIUM
    ) -> Dict[str, Any]:
        """
        Get prices with automatic batching - multiple calls are merged.
        
        This is ideal for scenarios where multiple concurrent requests
        are made for different coins. The batcher will automatically
        merge them into a single API call.
        """
        if vs_currencies is None:
            vs_currencies = ["usd"]
        
        async def executor(batch_params: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Execute the batched request."""
            # Merge all IDs from batch
            all_ids = []
            for params in batch_params:
                all_ids.extend(params["ids"])
            
            # Remove duplicates while preserving order
            seen = set()
            unique_ids = []
            for id in all_ids:
                if id not in seen:
                    seen.add(id)
                    unique_ids.append(id)
            
            # Execute single batched request
            return await self.get_simple_price(
                ids=unique_ids,
                vs_currencies=vs_currencies,
                priority=priority
            )
        
        # Use request batcher
        return await request_batcher.batch_request(
            endpoint="coingecko_prices",
            params={"ids": ids, "vs_currencies": vs_currencies},
            executor=executor
        )
    
    async def get_coins_markets(
        self,
        vs_currency: Union[str, List[str]] = "usd",
        ids: Optional[List[str]] = None,
        order: str = "market_cap_desc",
        per_page: int = 100,
        page: int = 1,
        price_change_percentage: Optional[List[str]] = None,
        priority: RequestPriority = RequestPriority.MEDIUM
    ) -> List[Dict[str, Any]]:
        """Get market data with rate limiting."""
        # Wait for rate limit
        await rate_limit_manager.wait_if_needed(self._api_name, priority)
        
        start_time = time.time()
        success = False
        rate_limited = False
        
        try:
            vs_currency_param = [vs_currency] if isinstance(vs_currency, str) else vs_currency
            result = await self._client.get_coins_markets(
                vs_currency=vs_currency_param,
                ids=ids,
                order=order,
                per_page=per_page,
                page=page,
                price_change_percentage=price_change_percentage,
            )
            success = True
            return result
            
        except Exception as e:
            if '429' in str(e) or 'rate limit' in str(e).lower():
                rate_limited = True
            raise
            
        finally:
            duration_ms = (time.time() - start_time) * 1000
            rate_limit_manager.record_request(
                api_name=self._api_name,
                duration_ms=duration_ms,
                success=success,
                priority=priority,
                rate_limited=rate_limited
            )
    
    async def get_coin_history(
        self,
        coin_id: str,
        date: str,
        priority: RequestPriority = RequestPriority.LOW
    ) -> Dict[str, Any]:
        """Get historical data with rate limiting (lower priority)."""
        await rate_limit_manager.wait_if_needed(self._api_name, priority)
        
        start_time = time.time()
        success = False
        rate_limited = False
        
        try:
            result = await self._client.get_coin_history(coin_id, date)
            success = True
            return result
            
        except Exception as e:
            if '429' in str(e) or 'rate limit' in str(e).lower():
                rate_limited = True
            raise
            
        finally:
            duration_ms = (time.time() - start_time) * 1000
            rate_limit_manager.record_request(
                api_name=self._api_name,
                duration_ms=duration_ms,
                success=success,
                priority=priority,
                rate_limited=rate_limited
            )
    
    async def get_coin_ohlc(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: int = 7,
        priority: RequestPriority = RequestPriority.MEDIUM
    ) -> List[List[float]]:
        """Get OHLC data with rate limiting."""
        await rate_limit_manager.wait_if_needed(self._api_name, priority)
        
        start_time = time.time()
        success = False
        rate_limited = False
        
        try:
            result = await self._client.get_coin_ohlc(coin_id, vs_currency, days)
            success = True
            return result
            
        except Exception as e:
            if '429' in str(e) or 'rate limit' in str(e).lower():
                rate_limited = True
            raise
            
        finally:
            duration_ms = (time.time() - start_time) * 1000
            rate_limit_manager.record_request(
                api_name=self._api_name,
                duration_ms=duration_ms,
                success=success,
                priority=priority,
                rate_limited=rate_limited
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiting statistics for CoinGecko API."""
        return rate_limit_manager.get_statistics(self._api_name)


# Convenience decorators for existing code
def rate_limited(priority: RequestPriority = RequestPriority.MEDIUM):
    """
    Decorator to add rate limiting to any CoinGecko API function.
    
    Usage:
        @rate_limited(RequestPriority.HIGH)
        async def get_urgent_prices():
            client = CoinGeckoClient()
            return await client.get_simple_price(["bitcoin"], ["usd"])
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            await rate_limit_manager.wait_if_needed("coingecko", priority)
            
            start_time = time.time()
            success = False
            
            try:
                result = await func(*args, **kwargs)
                success = True
                return result
            finally:
                duration_ms = (time.time() - start_time) * 1000
                rate_limit_manager.record_request(
                    api_name="coingecko",
                    duration_ms=duration_ms,
                    success=success,
                    priority=priority
                )
        
        return wrapper
    return decorator


# Global instance
rate_limited_coingecko_client = RateLimitedCoinGeckoClient()

