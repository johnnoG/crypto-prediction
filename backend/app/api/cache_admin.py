from __future__ import annotations

import time
from typing import Dict, Any

from fastapi import APIRouter

try:
    from cache import AsyncCache
    from config import get_settings
except ImportError:
    from cache import AsyncCache
    from config import get_settings


router = APIRouter(prefix="/cache", tags=["cache"])


@router.get("/status")
async def cache_status() -> Dict[str, Any]:
    """Get comprehensive cache system status."""
    cache = AsyncCache()
    await cache.initialize()
    
    settings = get_settings()
    healthy = await cache.ping()
    
    status = {
        "backend": cache.backend_name(),
        "healthy": healthy,
        "timestamp": int(time.time()),
        "redis_configured": settings.redis_url is not None,
        "redis_url": settings.redis_url if settings.redis_url else None,
    }
    
    # Add Redis-specific info if available
    if cache._redis is not None:
        try:
            info = await cache._redis.info()
            status.update({
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory_human": info.get("used_memory_human"),
                "uptime_in_seconds": info.get("uptime_in_seconds"),
            })
        except Exception:
            status["redis_info_error"] = "Could not retrieve Redis info"
    
    return status


@router.post("/clear")
async def clear_cache() -> Dict[str, str]:
    """Clear all cache entries (admin endpoint)."""
    cache = AsyncCache()
    await cache.initialize()
    
    if cache._redis is not None:
        try:
            await cache._redis.flushdb()
            return {"status": "success", "message": "Redis cache cleared"}
        except Exception as e:
            return {"status": "error", "message": f"Failed to clear Redis cache: {str(e)}"}
    else:
        # Clear in-memory cache
        cache._mem.clear()
        return {"status": "success", "message": "In-memory cache cleared"}


@router.get("/stats")
async def cache_stats() -> Dict[str, Any]:
    """Get cache statistics and metrics."""
    cache = AsyncCache()
    await cache.initialize()
    
    stats = {
        "backend": cache.backend_name(),
        "timestamp": int(time.time()),
    }
    
    if cache._redis is not None:
        try:
            info = await cache._redis.info()
            stats.update({
                "total_keys": await cache._redis.dbsize(),
                "memory_usage": info.get("used_memory"),
                "memory_usage_human": info.get("used_memory_human"),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": info.get("keyspace_hits", 0) / max(1, info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0)) * 100,
            })
        except Exception as e:
            stats["error"] = f"Could not retrieve Redis stats: {str(e)}"
    else:
        # In-memory cache stats
        stats.update({
            "total_keys": len(cache._mem),
            "memory_usage": "N/A (in-memory)",
            "hits": "N/A",
            "misses": "N/A",
            "hit_rate": "N/A",
        })
    
    return stats


