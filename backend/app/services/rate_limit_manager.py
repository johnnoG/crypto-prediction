"""
Advanced API Rate Limit Manager

Features:
- Intelligent request batching
- Priority queue (real-time > background)
- Rate limit prediction and forecasting
- Per-API rate limit tracking
- Automatic throttling and backoff
- Statistics and monitoring
"""

from __future__ import annotations

import asyncio
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from functools import wraps
import json


class RequestPriority(Enum):
    """Request priority levels."""
    CRITICAL = 0   # User-facing, blocking requests
    HIGH = 1       # Real-time data updates
    MEDIUM = 2     # Background updates
    LOW = 3        # Prefetch, cache warming


@dataclass
class RateLimitConfig:
    """Rate limit configuration for an API."""
    name: str
    requests_per_minute: int = 50
    requests_per_hour: int = 500
    requests_per_day: int = 10000
    burst_limit: int = 10  # Max concurrent requests
    min_interval_ms: int = 100  # Minimum time between requests (ms)
    
    # Pro tier settings (if available)
    pro_requests_per_minute: Optional[int] = None
    pro_requests_per_hour: Optional[int] = None
    pro_requests_per_day: Optional[int] = None
    
    # Cost estimation (for auto-upgrade decisions)
    free_tier: bool = True
    pro_cost_per_month: float = 0.0


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    timestamp: float
    priority: RequestPriority
    duration_ms: float
    success: bool
    status_code: Optional[int] = None
    rate_limited: bool = False


@dataclass
class APIStatistics:
    """Statistics for an API."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    total_duration_ms: float = 0
    
    # Rate limit tracking
    requests_last_minute: int = 0
    requests_last_hour: int = 0
    requests_last_day: int = 0
    
    # Predictions
    predicted_rpm: float = 0.0
    predicted_time_to_limit: float = float('inf')
    
    def average_duration_ms(self) -> float:
        """Calculate average request duration."""
        if self.successful_requests == 0:
            return 0
        return self.total_duration_ms / self.successful_requests
    
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests


class RateLimitManager:
    """
    Advanced rate limit manager with intelligent batching,
    priority queuing, and predictive rate limiting.
    """
    
    def __init__(self):
        # API configurations
        self.configs: Dict[str, RateLimitConfig] = {}
        
        # Request tracking
        self.request_history: Dict[str, deque[RequestMetrics]] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        
        # Statistics
        self.statistics: Dict[str, APIStatistics] = defaultdict(APIStatistics)
        
        # Priority queues for each API
        self.request_queues: Dict[str, Dict[RequestPriority, asyncio.Queue]] = {}
        
        # Rate limit state
        self.last_request_time: Dict[str, float] = {}
        self.current_burst: Dict[str, int] = defaultdict(int)
        
        # Locks for thread-safety
        self.locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._started = False
    
    def register_api(self, config: RateLimitConfig):
        """Register an API with its rate limit configuration."""
        self.configs[config.name] = config
        
        # Initialize priority queues
        self.request_queues[config.name] = {
            priority: asyncio.Queue() for priority in RequestPriority
        }
        
        print(f"[RATE LIMITER] Registered API: {config.name} "
              f"({config.requests_per_minute} req/min)")
    
    async def start(self):
        """Start background tasks."""
        if self._started:
            return
        
        self._started = True
        self._cleanup_task = asyncio.create_task(self._cleanup_old_metrics())
        print("[RATE LIMITER] Started background cleanup task")
    
    async def stop(self):
        """Stop background tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self._started = False
    
    async def _cleanup_old_metrics(self):
        """Periodically clean up old metrics."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                cutoff_time = time.time() - 86400  # Keep last 24 hours
                
                for api_name, history in self.request_history.items():
                    # Remove old metrics
                    while history and history[0].timestamp < cutoff_time:
                        history.popleft()
                
                print("[RATE LIMITER] Cleaned up old metrics")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[ERROR] Cleanup task error: {e}")
    
    def _update_statistics(self, api_name: str):
        """Update statistics for an API based on recent history."""
        history = self.request_history[api_name]
        stats = self.statistics[api_name]
        
        if not history:
            return
        
        now = time.time()
        
        # Count requests in different time windows
        minute_ago = now - 60
        hour_ago = now - 3600
        day_ago = now - 86400
        
        stats.requests_last_minute = sum(
            1 for m in history if m.timestamp > minute_ago
        )
        stats.requests_last_hour = sum(
            1 for m in history if m.timestamp > hour_ago
        )
        stats.requests_last_day = sum(
            1 for m in history if m.timestamp > day_ago
        )
        
        # Predict requests per minute based on recent trend
        recent_requests = [m for m in history if m.timestamp > now - 300]  # Last 5 min
        if len(recent_requests) >= 2:
            time_span = recent_requests[-1].timestamp - recent_requests[0].timestamp
            if time_span > 0:
                stats.predicted_rpm = (len(recent_requests) / time_span) * 60
        
        # Predict time until rate limit
        config = self.configs.get(api_name)
        if config and stats.predicted_rpm > 0:
            remaining = config.requests_per_minute - stats.requests_last_minute
            if remaining > 0 and stats.predicted_rpm > 0:
                stats.predicted_time_to_limit = (remaining / stats.predicted_rpm) * 60
            else:
                stats.predicted_time_to_limit = 0
    
    async def wait_if_needed(
        self,
        api_name: str,
        priority: RequestPriority = RequestPriority.MEDIUM
    ) -> float:
        """
        Wait if necessary to respect rate limits.
        Returns the wait time in seconds.
        """
        async with self.locks[api_name]:
            config = self.configs.get(api_name)
            if not config:
                return 0.0
            
            stats = self.statistics[api_name]
            now = time.time()
            
            # Check if we need to wait
            wait_time = 0.0
            
            # 1. Minimum interval between requests
            if api_name in self.last_request_time:
                time_since_last = (now - self.last_request_time[api_name]) * 1000
                if time_since_last < config.min_interval_ms:
                    wait_ms = config.min_interval_ms - time_since_last
                    wait_time = max(wait_time, wait_ms / 1000)
            
            # 2. Per-minute rate limit
            if stats.requests_last_minute >= config.requests_per_minute:
                # Wait until oldest request in current minute expires
                minute_ago = now - 60
                old_requests = [
                    m for m in self.request_history[api_name]
                    if m.timestamp > minute_ago
                ]
                if old_requests:
                    oldest = old_requests[0]
                    wait_until_reset = 60 - (now - oldest.timestamp)
                    wait_time = max(wait_time, wait_until_reset)
            
            # 3. Burst limit
            if self.current_burst[api_name] >= config.burst_limit:
                wait_time = max(wait_time, 1.0)  # Wait at least 1 second
            
            # 4. Priority-based throttling
            # Lower priority requests wait longer if we're approaching limits
            if priority in [RequestPriority.LOW, RequestPriority.MEDIUM]:
                utilization = stats.requests_last_minute / config.requests_per_minute
                if utilization > 0.8:  # Over 80% utilization
                    throttle_factor = 1 + (utilization - 0.8) * 10
                    if priority == RequestPriority.LOW:
                        throttle_factor *= 2
                    wait_time = max(wait_time, throttle_factor)
            
            # Apply wait time
            if wait_time > 0:
                print(f"[RATE LIMITER] {api_name}: Waiting {wait_time:.2f}s "
                      f"(priority={priority.name}, rpm={stats.requests_last_minute})")
                await asyncio.sleep(wait_time)
            
            # Update state
            self.last_request_time[api_name] = time.time()
            self.current_burst[api_name] += 1
            
            # Reset burst counter after delay
            asyncio.create_task(self._reset_burst_counter(api_name, config.min_interval_ms))
            
            return wait_time
    
    async def _reset_burst_counter(self, api_name: str, delay_ms: int):
        """Reset burst counter after delay."""
        await asyncio.sleep(delay_ms / 1000)
        if self.current_burst[api_name] > 0:
            self.current_burst[api_name] -= 1
    
    def record_request(
        self,
        api_name: str,
        duration_ms: float,
        success: bool,
        priority: RequestPriority = RequestPriority.MEDIUM,
        status_code: Optional[int] = None,
        rate_limited: bool = False
    ):
        """Record a completed request."""
        metrics = RequestMetrics(
            timestamp=time.time(),
            priority=priority,
            duration_ms=duration_ms,
            success=success,
            status_code=status_code,
            rate_limited=rate_limited
        )
        
        self.request_history[api_name].append(metrics)
        
        # Update statistics
        stats = self.statistics[api_name]
        stats.total_requests += 1
        stats.total_duration_ms += duration_ms
        
        if success:
            stats.successful_requests += 1
        else:
            stats.failed_requests += 1
        
        if rate_limited:
            stats.rate_limited_requests += 1
        
        self._update_statistics(api_name)
    
    def get_statistics(self, api_name: str) -> APIStatistics:
        """Get current statistics for an API."""
        self._update_statistics(api_name)
        return self.statistics[api_name]
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all APIs."""
        result = {}
        
        for api_name in self.configs:
            stats = self.get_statistics(api_name)
            config = self.configs[api_name]
            
            result[api_name] = {
                "config": {
                    "requests_per_minute": config.requests_per_minute,
                    "requests_per_hour": config.requests_per_hour,
                    "free_tier": config.free_tier,
                },
                "usage": {
                    "requests_last_minute": stats.requests_last_minute,
                    "requests_last_hour": stats.requests_last_hour,
                    "requests_last_day": stats.requests_last_day,
                    "utilization_percent": (
                        stats.requests_last_minute / config.requests_per_minute * 100
                    ),
                },
                "performance": {
                    "total_requests": stats.total_requests,
                    "success_rate": stats.success_rate(),
                    "average_duration_ms": stats.average_duration_ms(),
                    "rate_limited_requests": stats.rate_limited_requests,
                },
                "predictions": {
                    "predicted_rpm": stats.predicted_rpm,
                    "predicted_time_to_limit_seconds": stats.predicted_time_to_limit,
                    "should_upgrade": self._should_upgrade_api(api_name),
                },
            }
        
        return result
    
    def _should_upgrade_api(self, api_name: str) -> bool:
        """Determine if API should be upgraded to Pro tier."""
        config = self.configs.get(api_name)
        stats = self.statistics[api_name]
        
        if not config or not config.free_tier:
            return False
        
        # Upgrade if:
        # 1. Frequently hitting rate limits (>10% of requests)
        if stats.total_requests > 100:
            rate_limit_ratio = stats.rate_limited_requests / stats.total_requests
            if rate_limit_ratio > 0.1:
                return True
        
        # 2. Predicted to hit limit soon (<30 seconds)
        if stats.predicted_time_to_limit < 30:
            return True
        
        # 3. Consistently high utilization (>90% for extended period)
        if stats.requests_last_hour > config.requests_per_hour * 0.9:
            return True
        
        return False
    
    def rate_limit_decorator(
        self,
        api_name: str,
        priority: RequestPriority = RequestPriority.MEDIUM
    ):
        """
        Decorator to automatically apply rate limiting to async functions.
        
        Usage:
            @rate_limiter.rate_limit_decorator("coingecko", RequestPriority.HIGH)
            async def fetch_prices():
                ...
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Wait if needed before making request
                wait_time = await self.wait_if_needed(api_name, priority)
                
                # Execute request and track metrics
                start_time = time.time()
                success = False
                status_code = None
                rate_limited = False
                
                try:
                    result = await func(*args, **kwargs)
                    success = True
                    return result
                    
                except Exception as e:
                    # Check if it's a rate limit error
                    error_str = str(e).lower()
                    if '429' in error_str or 'rate limit' in error_str:
                        rate_limited = True
                        status_code = 429
                    raise
                    
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    self.record_request(
                        api_name=api_name,
                        duration_ms=duration_ms,
                        success=success,
                        priority=priority,
                        status_code=status_code,
                        rate_limited=rate_limited
                    )
            
            return wrapper
        return decorator


# Global rate limit manager instance
rate_limit_manager = RateLimitManager()


# Pre-configure common APIs
def initialize_rate_limiters():
    """Initialize rate limiters for common APIs."""
    
    # CoinGecko Free Tier
    rate_limit_manager.register_api(RateLimitConfig(
        name="coingecko",
        requests_per_minute=50,
        requests_per_hour=500,
        requests_per_day=10000,
        burst_limit=10,
        min_interval_ms=1200,  # 1.2 seconds between requests
        pro_requests_per_minute=500,
        pro_requests_per_hour=10000,
        pro_requests_per_day=100000,
        free_tier=True,
        pro_cost_per_month=129.0
    ))
    
    # CryptoCompare
    rate_limit_manager.register_api(RateLimitConfig(
        name="cryptocompare",
        requests_per_minute=100,
        requests_per_hour=2000,
        requests_per_day=100000,
        burst_limit=20,
        min_interval_ms=600,
        free_tier=True,
    ))
    
    # CryptoPanic
    rate_limit_manager.register_api(RateLimitConfig(
        name="cryptopanic",
        requests_per_minute=10,
        requests_per_hour=100,
        requests_per_day=1000,
        burst_limit=5,
        min_interval_ms=6000,  # 6 seconds
        free_tier=True,
    ))
    
    # Binance
    rate_limit_manager.register_api(RateLimitConfig(
        name="binance",
        requests_per_minute=1200,
        requests_per_hour=72000,
        requests_per_day=1728000,
        burst_limit=100,
        min_interval_ms=50,
        free_tier=True,
    ))
    
    print("[RATE LIMITER] Initialized rate limiters for all APIs")


# Initialize on import
initialize_rate_limiters()

