"""
Request Batching Service

Batches multiple API requests to reduce external API calls and improve performance.
Particularly useful for CoinGecko API which has rate limits.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
from collections import defaultdict
import time


class RequestBatcher:
    """
    Batches requests to external APIs to reduce call volume.
    
    Instead of making 10 separate API calls for 10 cryptos,
    make 1 batch call for all 10.
    """
    
    def __init__(
        self,
        batch_window_ms: int = 100,
        max_batch_size: int = 50
    ):
        """
        Args:
            batch_window_ms: Time window to collect requests (milliseconds)
            max_batch_size: Maximum items per batch
        """
        self.batch_window_ms = batch_window_ms
        self.max_batch_size = max_batch_size
        
        # Pending requests
        self._pending_requests: Dict[str, List[asyncio.Future]] = defaultdict(list)
        self._batch_timers: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
    
    async def batch_request(
        self,
        request_type: str,
        item_id: str,
        fetch_fn: callable
    ) -> Any:
        """
        Add a request to the batch queue.
        
        Args:
            request_type: Type of request (e.g., 'prices', 'market_data')
            item_id: ID of item to fetch (e.g., crypto ID)
            fetch_fn: Async function to fetch all batched items
                      Signature: async fn(item_ids: List[str]) -> Dict[str, Any]
            
        Returns:
            Result for the requested item
        """
        async with self._lock:
            # Create future for this request
            future = asyncio.Future()
            self._pending_requests[request_type].append((item_id, future))
            
            # Start batch timer if not already running
            if request_type not in self._batch_timers:
                self._batch_timers[request_type] = asyncio.create_task(
                    self._process_batch_after_delay(request_type, fetch_fn)
                )
        
        # Wait for batch to complete
        return await future
    
    async def _process_batch_after_delay(
        self,
        request_type: str,
        fetch_fn: callable
    ) -> None:
        """
        Wait for batch window, then process batch.
        
        Args:
            request_type: Type of request
            fetch_fn: Function to fetch batch
        """
        # Wait for batch window
        await asyncio.sleep(self.batch_window_ms / 1000.0)
        
        async with self._lock:
            # Get pending requests
            pending = self._pending_requests.get(request_type, [])
            
            if len(pending) == 0:
                # Clean up timer
                if request_type in self._batch_timers:
                    del self._batch_timers[request_type]
                return
            
            # Extract IDs and futures
            item_ids = [item_id for item_id, _ in pending]
            futures = [future for _, future in pending]
            
            # Clear pending requests
            self._pending_requests[request_type] = []
            del self._batch_timers[request_type]
        
        # Process batch
        try:
            # Remove duplicates while preserving order
            unique_ids = list(dict.fromkeys(item_ids))
            
            # Fetch all items in batch
            print(f"[BATCH] Fetching {len(unique_ids)} {request_type} in batch")
            results = await fetch_fn(unique_ids)
            
            # Resolve futures with results
            for item_id, future in zip(item_ids, futures):
                if not future.done():
                    result = results.get(item_id)
                    future.set_result(result)
        
        except Exception as e:
            # On error, reject all futures
            print(f"[BATCH] Error processing {request_type} batch: {e}")
            for future in futures:
                if not future.done():
                    future.set_exception(e)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get batching statistics.
        
        Returns:
            Dictionary with batching stats
        """
        return {
            'pending_batches': len(self._batch_timers),
            'pending_requests_by_type': {
                req_type: len(requests)
                for req_type, requests in self._pending_requests.items()
            },
            'batch_window_ms': self.batch_window_ms,
            'max_batch_size': self.max_batch_size
        }


class PriceBatchingService:
    """
    Specialized batching for cryptocurrency price requests.
    
    Combines multiple price requests into single API calls.
    """
    
    def __init__(self):
        self.batcher = RequestBatcher(
            batch_window_ms=100,  # 100ms window
            max_batch_size=50
        )
        
        self.stats = {
            'total_requests': 0,
            'batched_requests': 0,
            'api_calls_saved': 0
        }
    
    async def get_prices_batched(
        self,
        crypto_ids: List[str],
        vs_currency: str = 'usd'
    ) -> Dict[str, Any]:
        """
        Get prices for multiple cryptos, batching requests.
        
        Args:
            crypto_ids: List of CoinGecko IDs
            vs_currency: Currency for pricing
            
        Returns:
            Dictionary mapping crypto_id to price data
        """
        from services.prices_service import get_simple_price_with_cache
        
        # For single crypto, don't batch
        if len(crypto_ids) == 1:
            self.stats['total_requests'] += 1
            return await get_simple_price_with_cache(
                ids=crypto_ids[0],
                vs_currencies=vs_currency
            )
        
        # Define fetch function for batch
        async def fetch_batch(ids: List[str]) -> Dict[str, Any]:
            self.stats['batched_requests'] += len(ids)
            self.stats['api_calls_saved'] += len(ids) - 1  # Saved N-1 API calls
            
            ids_str = ','.join(ids)
            return await get_simple_price_with_cache(
                ids=ids_str,
                vs_currencies=vs_currency
            )
        
        # Batch requests
        results = {}
        tasks = []
        
        for crypto_id in crypto_ids:
            self.stats['total_requests'] += 1
            task = self.batcher.batch_request('prices', crypto_id, fetch_batch)
            tasks.append((crypto_id, task))
        
        # Wait for all batched requests
        for crypto_id, task in tasks:
            result = await task
            if result:
                results.update(result)
        
        return results
    
    def get_batching_efficiency(self) -> Dict[str, Any]:
        """
        Calculate batching efficiency metrics.
        
        Returns:
            Efficiency statistics
        """
        total = self.stats['total_requests']
        
        if total == 0:
            return {
                'total_requests': 0,
                'efficiency': 0,
                'api_calls_saved': 0
            }
        
        return {
            'total_requests': total,
            'batched_requests': self.stats['batched_requests'],
            'api_calls_saved': self.stats['api_calls_saved'],
            'efficiency_pct': (self.stats['api_calls_saved'] / total * 100) if total > 0 else 0
        }


# Global batching service
price_batching_service = PriceBatchingService()

