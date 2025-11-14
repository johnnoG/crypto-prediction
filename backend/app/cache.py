"""
Redis Cache Manager

Provides a high-level interface for caching operations with Redis.
Implements connection pooling, serialization, and TTL management.
"""

import json
import logging
from typing import Any, Optional
from redis import Redis, ConnectionPool
from redis.exceptions import RedisError, ConnectionError, TimeoutError

from app.config import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Redis cache manager with connection pooling and error handling.

    Provides methods for getting, setting, and managing cached data
    with automatic serialization/deserialization.
    """

    def __init__(self):
        """Initialize Redis connection pool."""
        self._pool: Optional[ConnectionPool] = None
        self._redis: Optional[Redis] = None

    def connect(self) -> None:
        """
        Establish Redis connection with connection pooling.

        Raises:
            RedisError: If connection to Redis fails
        """
        try:
            self._pool = ConnectionPool.from_url(
                settings.redis_url_str,
                max_connections=settings.redis_max_connections,
                socket_timeout=settings.redis_socket_timeout,
                socket_connect_timeout=settings.redis_socket_connect_timeout,
                decode_responses=True,  # Automatically decode bytes to strings
            )
            self._redis = Redis(connection_pool=self._pool)
            # Test connection
            self._redis.ping()
            logger.info("Successfully connected to Redis")
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def disconnect(self) -> None:
        """Close Redis connections and cleanup resources."""
        if self._redis:
            self._redis.close()
        if self._pool:
            self._pool.disconnect()
        logger.info("Disconnected from Redis")

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value (deserialized from JSON) or None if not found

        Raises:
            RedisError: If Redis operation fails
        """
        try:
            value = self._redis.get(key)
            if value is None:
                logger.debug(f"Cache miss: {key}")
                return None
            logger.debug(f"Cache hit: {key}")
            return json.loads(value)
        except (RedisError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to get cache key '{key}': {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Store a value in cache with optional TTL.

        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time to live in seconds (None for no expiration)

        Returns:
            True if successful, False otherwise

        Raises:
            RedisError: If Redis operation fails
        """
        try:
            serialized_value = json.dumps(value)
            if ttl:
                self._redis.setex(key, ttl, serialized_value)
            else:
                self._redis.set(key, serialized_value)
            logger.debug(f"Cached key '{key}' with TTL {ttl}s")
            return True
        except (RedisError, json.JSONEncodeError, TypeError) as e:
            logger.warning(f"Failed to set cache key '{key}': {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete a key from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if key was deleted, False otherwise
        """
        try:
            deleted = self._redis.delete(key)
            logger.debug(f"Deleted cache key '{key}': {deleted > 0}")
            return deleted > 0
        except RedisError as e:
            logger.warning(f"Failed to delete cache key '{key}': {e}")
            return False

    def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.

        Args:
            pattern: Redis key pattern (e.g., "user:*")

        Returns:
            Number of keys deleted
        """
        try:
            keys = self._redis.keys(pattern)
            if keys:
                deleted = self._redis.delete(*keys)
                logger.info(f"Deleted {deleted} keys matching pattern '{pattern}'")
                return deleted
            return 0
        except RedisError as e:
            logger.warning(f"Failed to delete keys with pattern '{pattern}': {e}")
            return 0

    def exists(self, key: str) -> bool:
        """
        Check if a key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise
        """
        try:
            return self._redis.exists(key) > 0
        except RedisError as e:
            logger.warning(f"Failed to check existence of key '{key}': {e}")
            return False

    def get_ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for a key.

        Args:
            key: Cache key

        Returns:
            Remaining TTL in seconds, or None if key doesn't exist
        """
        try:
            ttl = self._redis.ttl(key)
            if ttl < 0:
                return None
            return ttl
        except RedisError as e:
            logger.warning(f"Failed to get TTL for key '{key}': {e}")
            return None

    def flush_all(self) -> bool:
        """
        Clear all keys from current database.

        WARNING: This is a destructive operation. Use with caution!

        Returns:
            True if successful, False otherwise
        """
        try:
            self._redis.flushdb()
            logger.warning("Flushed all keys from Redis database")
            return True
        except RedisError as e:
            logger.error(f"Failed to flush Redis database: {e}")
            return False

    def health_check(self) -> dict:
        """
        Perform health check on Redis connection.

        Returns:
            Dictionary with health status and metrics
        """
        try:
            # Ping Redis
            latency_ms = self._redis.ping()

            # Get basic info
            info = self._redis.info()

            return {
                "status": "healthy",
                "connected": True,
                "latency_ms": latency_ms,
                "used_memory_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
                "connected_clients": info.get("connected_clients", 0),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
            }
        except RedisError as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
            }


# Global cache manager instance
cache_manager = CacheManager()


def get_cache() -> CacheManager:
    """
    Get cache manager instance for dependency injection.

    Returns:
        CacheManager instance
    """
    return cache_manager
