"""
Circuit Breaker Pattern

Prevents cascading failures by:
- Detecting repeated failures
- Opening circuit to stop requests to failing service
- Half-open state for recovery testing
- Automatic recovery when service is healthy again
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker"""
    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: Optional[datetime]
    last_success_time: Optional[datetime]
    opened_at: Optional[datetime]


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation, requests go through
    - OPEN: Too many failures, requests blocked
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        half_open_timeout: float = 30.0,
        name: str = "circuit_breaker"
    ):
        """
        Args:
            failure_threshold: Number of failures before opening circuit
            success_threshold: Number of successes in half-open to close circuit
            timeout: Seconds to wait before trying half-open
            half_open_timeout: Seconds to wait in half-open before reopening
            name: Name for logging
        """
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.half_open_timeout = half_open_timeout
        self.name = name
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        self.opened_at: Optional[datetime] = None
        
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Async function to execute
            *args, **kwargs: Arguments for function
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        async with self._lock:
            # Check state
            if self.state == CircuitState.OPEN:
                # Check if timeout expired
                if self.opened_at:
                    elapsed = (datetime.now() - self.opened_at).total_seconds()
                    if elapsed >= self.timeout:
                        print(f"[{self.name}] Transitioning to HALF_OPEN")
                        self.state = CircuitState.HALF_OPEN
                        self.success_count = 0
                    else:
                        raise CircuitBreakerOpenError(
                            f"Circuit breaker {self.name} is OPEN. "
                            f"Retry in {self.timeout - elapsed:.0f}s"
                        )
            
            elif self.state == CircuitState.HALF_OPEN:
                # Check if half-open timeout expired
                if self.opened_at:
                    elapsed = (datetime.now() - self.opened_at).total_seconds()
                    if elapsed >= self.timeout + self.half_open_timeout:
                        print(f"[{self.name}] Half-open timeout expired, reopening circuit")
                        self.state = CircuitState.OPEN
                        self.opened_at = datetime.now()
                        raise CircuitBreakerOpenError(f"Circuit breaker {self.name} reopened")
        
        # Execute function
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise
    
    async def _on_success(self) -> None:
        """Handle successful request"""
        async with self._lock:
            self.last_success_time = datetime.now()
            self.failure_count = 0  # Reset failures on success
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                
                if self.success_count >= self.success_threshold:
                    print(f"[{self.name}] Transitioning to CLOSED (recovered)")
                    self.state = CircuitState.CLOSED
                    self.opened_at = None
                    self.success_count = 0
    
    async def _on_failure(self) -> None:
        """Handle failed request"""
        async with self._lock:
            self.last_failure_time = datetime.now()
            self.failure_count += 1
            
            if self.state == CircuitState.HALF_OPEN:
                # Failure in half-open -> reopen circuit
                print(f"[{self.name}] Failure in HALF_OPEN, transitioning to OPEN")
                self.state = CircuitState.OPEN
                self.opened_at = datetime.now()
                self.success_count = 0
            
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    print(f"[{self.name}] Failure threshold reached, transitioning to OPEN")
                    self.state = CircuitState.OPEN
                    self.opened_at = datetime.now()
    
    def get_stats(self) -> CircuitBreakerStats:
        """Get current circuit breaker statistics"""
        return CircuitBreakerStats(
            state=self.state,
            failure_count=self.failure_count,
            success_count=self.success_count,
            last_failure_time=self.last_failure_time,
            last_success_time=self.last_success_time,
            opened_at=self.opened_at
        )
    
    async def reset(self) -> None:
        """Manually reset circuit breaker"""
        async with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.opened_at = None
            print(f"[{self.name}] Circuit breaker reset")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


# Global circuit breakers for different services
coingecko_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    success_threshold=2,
    timeout=60.0,
    name="coingecko_api"
)

forecast_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    success_threshold=2,
    timeout=30.0,
    name="forecast_service"
)

