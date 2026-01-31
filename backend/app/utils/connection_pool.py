"""
Connection Pool Manager

Prevents connection leaks and CLOSE_WAIT states that caused backend crashes.
Implements proper httpx client lifecycle management.
"""

from __future__ import annotations

import httpx
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import asyncio


class ConnectionPoolManager:
    """
    Manages HTTP connection pools for external APIs.
    
    Prevents connection leaks by:
    - Setting max_connections limits
    - Proper connection keepalive management
    - Automatic cleanup of stale connections
    - Connection reuse
    """
    
    def __init__(
        self,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        keepalive_expiry: float = 5.0,
        timeout: float = 30.0
    ):
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.keepalive_expiry = keepalive_expiry
        self.timeout = timeout
        
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()
    
    async def get_client(self) -> httpx.AsyncClient:
        """
        Get or create shared HTTP client.
        
        Returns:
            Configured httpx.AsyncClient
        """
        async with self._lock:
            if self._client is None or self._client.is_closed:
                limits = httpx.Limits(
                    max_connections=self.max_connections,
                    max_keepalive_connections=self.max_keepalive_connections,
                    keepalive_expiry=self.keepalive_expiry
                )
                
                timeout = httpx.Timeout(
                    connect=10.0,
                    read=self.timeout,
                    write=10.0,
                    pool=5.0
                )
                
                self._client = httpx.AsyncClient(
                    limits=limits,
                    timeout=timeout,
                    follow_redirects=True,
                    http2=False  # Disable HTTP/2 to avoid connection issues
                )
                
                print(f"Created new HTTP client with max_connections={self.max_connections}")
            
            return self._client
    
    async def close(self) -> None:
        """Close the HTTP client and cleanup connections"""
        async with self._lock:
            if self._client and not self._client.is_closed:
                await self._client.aclose()
                print("HTTP client closed")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return await self.get_client()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Don't close on exit - keep connection pool alive
        pass
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get current connection pool statistics.
        
        Returns:
            Dictionary with connection stats
        """
        if self._client is None:
            return {
                'status': 'not_initialized',
                'active_connections': 0
            }
        
        return {
            'status': 'active' if not self._client.is_closed else 'closed',
            'max_connections': self.max_connections,
            'max_keepalive': self.max_keepalive_connections,
            'is_closed': self._client.is_closed
        }


# Global connection pool instance
connection_pool = ConnectionPoolManager(
    max_connections=100,
    max_keepalive_connections=20,
    keepalive_expiry=5.0,
    timeout=30.0
)


@asynccontextmanager
async def get_http_client():
    """
    Context manager for getting HTTP client.
    
    Usage:
        async with get_http_client() as client:
            response = await client.get(url)
    """
    client = await connection_pool.get_client()
    try:
        yield client
    finally:
        # Client remains open for reuse
        pass

