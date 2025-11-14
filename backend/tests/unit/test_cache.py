"""
Tests for Redis Cache Manager

Tests cache operations including:
- Get/Set operations
- TTL management
- Pattern-based deletion
- Health checks
"""

import pytest
import json
from app.cache import CacheManager


class TestCacheManager:
    """Test cases for CacheManager."""

    @pytest.mark.unit
    @pytest.mark.cache
    def test_set_and_get(self, test_cache: CacheManager):
        """Test basic set and get operations."""
        key = "test:key"
        value = {"data": "test_value", "number": 123}

        # Set value
        result = test_cache.set(key, value)
        assert result is True

        # Get value
        retrieved = test_cache.get(key)
        assert retrieved == value

    @pytest.mark.unit
    @pytest.mark.cache
    def test_get_nonexistent_key(self, test_cache: CacheManager):
        """Test getting a non-existent key returns None."""
        result = test_cache.get("nonexistent:key")
        assert result is None

    @pytest.mark.unit
    @pytest.mark.cache
    def test_set_with_ttl(self, test_cache: CacheManager):
        """Test setting value with TTL."""
        key = "test:ttl"
        value = "temporary_value"
        ttl = 60

        result = test_cache.set(key, value, ttl=ttl)
        assert result is True

        # Check TTL
        remaining_ttl = test_cache.get_ttl(key)
        assert remaining_ttl is not None
        assert remaining_ttl <= ttl

    @pytest.mark.unit
    @pytest.mark.cache
    def test_delete_key(self, test_cache: CacheManager):
        """Test deleting a key."""
        key = "test:delete"
        test_cache.set(key, "value")

        # Verify key exists
        assert test_cache.exists(key) is True

        # Delete key
        result = test_cache.delete(key)
        assert result is True

        # Verify key is gone
        assert test_cache.exists(key) is False

    @pytest.mark.unit
    @pytest.mark.cache
    def test_delete_nonexistent_key(self, test_cache: CacheManager):
        """Test deleting non-existent key."""
        result = test_cache.delete("nonexistent:key")
        assert result is False

    @pytest.mark.unit
    @pytest.mark.cache
    def test_exists(self, test_cache: CacheManager):
        """Test checking if key exists."""
        key = "test:exists"

        # Key doesn't exist yet
        assert test_cache.exists(key) is False

        # Set key
        test_cache.set(key, "value")

        # Key exists now
        assert test_cache.exists(key) is True

    @pytest.mark.unit
    @pytest.mark.cache
    def test_delete_pattern(self, test_cache: CacheManager):
        """Test deleting keys by pattern."""
        # Set multiple keys
        test_cache.set("crypto:BTC:price", 45000)
        test_cache.set("crypto:ETH:price", 2500)
        test_cache.set("crypto:BTC:volume", 1000000)
        test_cache.set("other:key", "value")

        # Delete crypto:BTC:* pattern
        deleted = test_cache.delete_pattern("crypto:BTC:*")
        assert deleted == 2

        # Verify correct keys were deleted
        assert test_cache.exists("crypto:BTC:price") is False
        assert test_cache.exists("crypto:BTC:volume") is False
        assert test_cache.exists("crypto:ETH:price") is True
        assert test_cache.exists("other:key") is True

    @pytest.mark.unit
    @pytest.mark.cache
    def test_get_ttl(self, test_cache: CacheManager):
        """Test getting TTL of a key."""
        key = "test:ttl_check"

        # Set with TTL
        test_cache.set(key, "value", ttl=300)
        ttl = test_cache.get_ttl(key)
        assert ttl is not None
        assert 0 < ttl <= 300

        # Non-existent key
        ttl = test_cache.get_ttl("nonexistent:key")
        assert ttl is None

    @pytest.mark.unit
    @pytest.mark.cache
    def test_complex_data_serialization(self, test_cache: CacheManager):
        """Test serialization of complex data structures."""
        key = "test:complex"
        value = {
            "string": "test",
            "number": 123,
            "float": 45.67,
            "bool": True,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
        }

        test_cache.set(key, value)
        retrieved = test_cache.get(key)

        assert retrieved == value
        assert retrieved["string"] == "test"
        assert retrieved["nested"]["key"] == "value"

    @pytest.mark.unit
    @pytest.mark.cache
    def test_health_check(self, test_cache: CacheManager):
        """Test cache health check."""
        health = test_cache.health_check()

        assert "status" in health
        assert health["status"] == "healthy"
        assert health["connected"] is True


class TestCacheErrors:
    """Test cache error handling."""

    @pytest.mark.unit
    @pytest.mark.cache
    def test_set_invalid_json(self, test_cache: CacheManager):
        """Test handling of non-serializable data."""
        key = "test:invalid"

        # Try to cache non-serializable object
        class NonSerializable:
            pass

        value = NonSerializable()
        result = test_cache.set(key, value)

        # Should return False (error handled gracefully)
        assert result is False

    @pytest.mark.unit
    @pytest.mark.cache
    def test_get_corrupted_data(self, test_cache: CacheManager):
        """Test handling of corrupted JSON data."""
        key = "test:corrupted"

        # Manually set invalid JSON
        test_cache._redis.set(key, "{invalid json}")

        # Should return None (error handled gracefully)
        result = test_cache.get(key)
        assert result is None


class TestCachePatterns:
    """Test common caching patterns."""

    @pytest.mark.unit
    @pytest.mark.cache
    def test_cache_aside_pattern(self, test_cache: CacheManager):
        """Test cache-aside (lazy loading) pattern."""

        def get_data_from_db(key: str) -> dict:
            """Simulate database fetch."""
            return {"data": f"value_for_{key}"}

        def get_with_cache(key: str) -> dict:
            """Implement cache-aside pattern."""
            cache_key = f"cache:{key}"

            # Try cache first
            cached = test_cache.get(cache_key)
            if cached:
                return cached

            # Cache miss - fetch from DB
            data = get_data_from_db(key)

            # Store in cache
            test_cache.set(cache_key, data, ttl=300)

            return data

        # First call - cache miss
        result1 = get_with_cache("test")
        assert result1 == {"data": "value_for_test"}

        # Second call - cache hit
        result2 = get_with_cache("test")
        assert result2 == {"data": "value_for_test"}

        # Verify it was cached
        assert test_cache.exists("cache:test") is True

    @pytest.mark.unit
    @pytest.mark.cache
    def test_cache_invalidation(self, test_cache: CacheManager):
        """Test cache invalidation on data update."""
        crypto_id = "BTC"

        # Set initial cache
        test_cache.set(f"crypto:{crypto_id}:price", 45000, ttl=300)
        test_cache.set(f"crypto:{crypto_id}:volume", 1000000, ttl=300)
        test_cache.set(f"crypto:{crypto_id}:metadata", {"name": "Bitcoin"}, ttl=3600)

        # Verify cached
        assert test_cache.exists(f"crypto:{crypto_id}:price") is True

        # Simulate data update - invalidate related cache
        def update_crypto_price(crypto_id: str, new_price: float):
            # Update DB (simulated)
            # ...

            # Invalidate cache
            test_cache.delete_pattern(f"crypto:{crypto_id}:*")

        update_crypto_price(crypto_id, 46000)

        # Verify cache was invalidated
        assert test_cache.exists(f"crypto:{crypto_id}:price") is False
        assert test_cache.exists(f"crypto:{crypto_id}:volume") is False
        assert test_cache.exists(f"crypto:{crypto_id}:metadata") is False

    @pytest.mark.unit
    @pytest.mark.cache
    def test_tiered_ttl_strategy(self, test_cache: CacheManager):
        """Test different TTL tiers for different data types."""
        from app.config import settings

        # Short TTL - frequently updated data
        test_cache.set("price:latest", 45000, ttl=settings.cache_ttl_short)
        ttl_short = test_cache.get_ttl("price:latest")
        assert ttl_short <= settings.cache_ttl_short

        # Medium TTL - moderately stable data
        test_cache.set("stats:daily", {"avg": 45000}, ttl=settings.cache_ttl_medium)
        ttl_medium = test_cache.get_ttl("stats:daily")
        assert ttl_medium <= settings.cache_ttl_medium

        # Long TTL - static data
        test_cache.set("crypto:metadata", {"name": "BTC"}, ttl=settings.cache_ttl_long)
        ttl_long = test_cache.get_ttl("crypto:metadata")
        assert ttl_long <= settings.cache_ttl_long
