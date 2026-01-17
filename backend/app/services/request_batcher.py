"""
Intelligent Request Batching System

Automatically batches similar API requests to reduce total API calls.
Supports automatic debouncing, deduplication, and smart grouping.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar
from functools import wraps
import json


T = TypeVar('T')


@dataclass
class BatchRequest:
    """A single request in a batch."""
    request_id: str
    params: Dict[str, Any]
    future: asyncio.Future
    timestamp: float


@dataclass
class BatchConfig:
    """Configuration for request batching."""
    max_batch_size: int = 100  # Max requests per batch
    max_wait_ms: int = 50  # Max time to wait for more requests (ms)
    min_batch_size: int = 2  # Min requests to trigger batch
    deduplicate: bool = True  # Remove duplicate requests
    combine_strategy: str = "merge"  # How to combine params: "merge", "union", "individual"


class RequestBatcher:
    """
    Intelligent request batcher that automatically groups similar requests.
    
    Features:
    - Automatic deduplication
    - Smart debouncing
    - Multiple batching strategies
    - Per-endpoint configuration
    """
    
    def __init__(self):
        # Pending requests by endpoint and batch key
        self.pending_batches: Dict[str, Dict[str, List[BatchRequest]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Batch configurations per endpoint
        self.configs: Dict[str, BatchConfig] = {}
        
        # Active batch tasks
        self.batch_tasks: Dict[str, asyncio.Task] = {}
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "batched_requests": 0,
            "api_calls_saved": 0,
            "average_batch_size": 0.0,
        }
        
        print("[REQUEST BATCHER] Initialized request batching system")
    
    def configure_endpoint(self, endpoint: str, config: BatchConfig):
        """Configure batching for a specific endpoint."""
        self.configs[endpoint] = config
        print(f"[REQUEST BATCHER] Configured endpoint: {endpoint} "
              f"(max_batch={config.max_batch_size}, wait={config.max_wait_ms}ms)")
    
    def _get_batch_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate a key for grouping similar requests."""
        config = self.configs.get(endpoint, BatchConfig())
        
        if config.combine_strategy == "individual":
            # Each request is its own batch
            return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
        
        elif config.combine_strategy == "union":
            # Group by request type only (all similar requests batched together)
            # Extract the common parameters
            common_params = {
                k: v for k, v in params.items()
                if not isinstance(v, (list, dict))
            }
            return hashlib.md5(json.dumps(common_params, sort_keys=True).encode()).hexdigest()[:8]
        
        else:  # merge strategy
            # Group requests that can be merged (e.g., multiple IDs)
            if "ids" in params or "symbols" in params or "coins" in params:
                # Requests with list parameters can be merged
                return "batchable"
            else:
                # Other requests grouped individually
                return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
    
    def _get_request_signature(self, params: Dict[str, Any]) -> str:
        """Generate unique signature for deduplication."""
        return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
    
    async def batch_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        executor: Callable[[List[Dict[str, Any]]], Any],
    ) -> Any:
        """
        Submit a request for batching.
        
        Args:
            endpoint: API endpoint name
            params: Request parameters
            executor: Async function that executes the actual API call with batched params
            
        Returns:
            The result for this specific request
        """
        self.stats["total_requests"] += 1
        
        config = self.configs.get(endpoint, BatchConfig())
        batch_key = self._get_batch_key(endpoint, params)
        
        # Create request
        request_id = self._get_request_signature(params)
        future = asyncio.Future()
        
        batch_request = BatchRequest(
            request_id=request_id,
            params=params,
            future=future,
            timestamp=asyncio.get_event_loop().time()
        )
        
        # Check for duplicate requests if deduplication is enabled
        if config.deduplicate:
            existing_requests = self.pending_batches[endpoint][batch_key]
            for existing in existing_requests:
                if existing.request_id == request_id:
                    # Reuse existing future
                    print(f"[REQUEST BATCHER] Deduplicated request: {endpoint}")
                    self.stats["api_calls_saved"] += 1
                    return await existing.future
        
        # Add to pending batch
        self.pending_batches[endpoint][batch_key].append(batch_request)
        
        # Start batch timer if not already started
        task_key = f"{endpoint}:{batch_key}"
        if task_key not in self.batch_tasks or self.batch_tasks[task_key].done():
            self.batch_tasks[task_key] = asyncio.create_task(
                self._execute_batch_after_delay(endpoint, batch_key, executor, config)
            )
        
        # Wait for result
        return await future
    
    async def _execute_batch_after_delay(
        self,
        endpoint: str,
        batch_key: str,
        executor: Callable,
        config: BatchConfig
    ):
        """Execute batch after waiting for more requests."""
        try:
            # Wait for more requests
            await asyncio.sleep(config.max_wait_ms / 1000)
            
            # Get all pending requests for this batch
            requests = self.pending_batches[endpoint][batch_key]
            
            if not requests:
                return
            
            # Check if we should batch
            if len(requests) < config.min_batch_size:
                # Execute individually
                for req in requests:
                    try:
                        result = await executor([req.params])
                        req.future.set_result(result)
                    except Exception as e:
                        req.future.set_exception(e)
            else:
                # Execute as batch
                self.stats["batched_requests"] += len(requests)
                self.stats["api_calls_saved"] += len(requests) - 1
                
                # Update average batch size
                total_batched = self.stats["batched_requests"]
                new_avg = (
                    self.stats["average_batch_size"] * (total_batched - len(requests)) + len(requests)
                ) / total_batched if total_batched > 0 else len(requests)
                self.stats["average_batch_size"] = new_avg
                
                print(f"[REQUEST BATCHER] Executing batch: {endpoint} "
                      f"(size={len(requests)}, saved={len(requests)-1} API calls)")
                
                try:
                    # Merge parameters for batch request
                    batch_params = self._merge_params(requests, config)
                    
                    # Execute batch API call
                    batch_result = await executor(batch_params)
                    
                    # Distribute results to individual futures
                    self._distribute_results(requests, batch_result, batch_params)
                    
                except Exception as e:
                    print(f"[ERROR] Batch execution failed: {e}")
                    # Fail all requests in batch
                    for req in requests:
                        if not req.future.done():
                            req.future.set_exception(e)
            
            # Clean up
            del self.pending_batches[endpoint][batch_key]
            
        except Exception as e:
            print(f"[ERROR] Batch execution error: {e}")
    
    def _merge_params(self, requests: List[BatchRequest], config: BatchConfig) -> List[Dict[str, Any]]:
        """Merge parameters from multiple requests."""
        if config.combine_strategy == "individual":
            return [req.params for req in requests]
        
        elif config.combine_strategy == "union":
            # Merge list parameters (e.g., ids, symbols)
            merged = {}
            list_keys = set()
            
            for req in requests:
                for key, value in req.params.items():
                    if isinstance(value, list):
                        list_keys.add(key)
                        if key not in merged:
                            merged[key] = []
                        merged[key].extend(value)
                    else:
                        merged[key] = value
            
            # Remove duplicates from lists
            for key in list_keys:
                if key in merged:
                    merged[key] = list(set(merged[key]))
            
            return [merged]
        
        else:  # merge strategy
            # Smart merging based on parameter types
            if any("ids" in req.params for req in requests):
                # Merge IDs
                all_ids = []
                base_params = {}
                
                for req in requests:
                    if "ids" in req.params:
                        ids = req.params["ids"]
                        if isinstance(ids, str):
                            all_ids.append(ids)
                        else:
                            all_ids.extend(ids)
                    
                    # Copy other params from first request
                    if not base_params:
                        base_params = {k: v for k, v in req.params.items() if k != "ids"}
                
                base_params["ids"] = list(set(all_ids))
                return [base_params]
            
            else:
                # Return individual params
                return [req.params for req in requests]
    
    def _distribute_results(
        self,
        requests: List[BatchRequest],
        batch_result: Any,
        batch_params: List[Dict[str, Any]]
    ):
        """Distribute batch results to individual request futures."""
        if len(batch_params) == 1:
            # Single merged request - parse and distribute
            # Assume batch_result is a dict with keys matching requested IDs
            if isinstance(batch_result, dict):
                for req in requests:
                    # Extract relevant data for this request
                    if "ids" in req.params:
                        req_ids = req.params["ids"]
                        if isinstance(req_ids, str):
                            req_ids = [req_ids]
                        
                        # Filter result for this request's IDs
                        filtered_result = {
                            k: v for k, v in batch_result.items()
                            if k in req_ids
                        }
                        req.future.set_result(filtered_result)
                    else:
                        # Return full result
                        req.future.set_result(batch_result)
            else:
                # Return same result to all
                for req in requests:
                    req.future.set_result(batch_result)
        else:
            # Individual requests - match results 1:1
            for req, result in zip(requests, batch_result if isinstance(batch_result, list) else [batch_result] * len(requests)):
                req.future.set_result(result)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get batching statistics."""
        if self.stats["total_requests"] == 0:
            efficiency = 0
        else:
            efficiency = (self.stats["api_calls_saved"] / self.stats["total_requests"]) * 100
        
        return {
            **self.stats,
            "efficiency_percent": efficiency,
            "current_pending_batches": sum(
                len(batches) for batches in self.pending_batches.values()
            ),
        }
    
    def batch_decorator(
        self,
        endpoint: str,
        config: Optional[BatchConfig] = None
    ):
        """
        Decorator to automatically batch requests to a function.
        
        Usage:
            @batcher.batch_decorator("get_prices")
            async def get_prices(ids: List[str]):
                # This will be called with batched IDs
                return await api.get_prices(ids)
            
            # Individual calls will be automatically batched:
            price1 = await get_prices(["bitcoin"])
            price2 = await get_prices(["ethereum"])  # Batched with above
        """
        if config:
            self.configure_endpoint(endpoint, config)
        
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(**kwargs):
                # Use batching
                return await self.batch_request(
                    endpoint=endpoint,
                    params=kwargs,
                    executor=lambda batch_params: func(**batch_params[0])
                )
            
            return wrapper
        return decorator


# Global request batcher instance
request_batcher = RequestBatcher()


# Pre-configure common endpoints
def initialize_batching():
    """Initialize batching for common API endpoints."""
    
    # CoinGecko price fetching - highly batchable
    request_batcher.configure_endpoint(
        "coingecko_prices",
        BatchConfig(
            max_batch_size=250,  # CoinGecko supports up to 250 IDs
            max_wait_ms=100,
            min_batch_size=2,
            deduplicate=True,
            combine_strategy="union"
        )
    )
    
    # CoinGecko market data
    request_batcher.configure_endpoint(
        "coingecko_markets",
        BatchConfig(
            max_batch_size=100,
            max_wait_ms=150,
            min_batch_size=3,
            deduplicate=True,
            combine_strategy="union"
        )
    )
    
    # Historical data - less batchable
    request_batcher.configure_endpoint(
        "coingecko_historical",
        BatchConfig(
            max_batch_size=10,
            max_wait_ms=50,
            min_batch_size=2,
            deduplicate=True,
            combine_strategy="individual"
        )
    )
    
    print("[REQUEST BATCHER] Configured batching for common endpoints")


# Initialize on import
initialize_batching()

