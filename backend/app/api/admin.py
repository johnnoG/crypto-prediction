"""
Admin API Endpoints

Administrative functions for cache management and system control.
"""

from __future__ import annotations

from typing import Dict, Any
from fastapi import APIRouter, Request, Response

try:
    from services.smart_cache_service import smart_cache
    from api.dependencies.rate_limiter import rate_limit
except ImportError:
    from services.smart_cache_service import smart_cache
    from api.dependencies.rate_limiter import rate_limit  # type: ignore


router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/cache/refresh")
@rate_limit("admin")
async def refresh_cache(
    request: Request,
    response: Response,
    data_type: str = "all"  # "all", "prices", "market_data", "forecasts"
) -> Dict[str, Any]:
    """
    Manually trigger cache refresh.
    
    Useful when rate limits have reset and you want fresh data.
    
    Args:
        data_type: Type of data to refresh ("all", "prices", "market_data", "forecasts")
    """
    results = {}
    
    try:
        if data_type in ["all", "prices"]:
            await smart_cache._update_prices_cache()
            results["prices"] = "refreshed"
        
        if data_type in ["all", "market_data"]:
            await smart_cache._update_market_data_cache()
            results["market_data"] = "refreshed"
        
        if data_type in ["all", "forecasts"]:
            await smart_cache._update_forecasts_cache()
            results["forecasts"] = "refreshed"
        
        return {
            "status": "success",
            "refreshed": results,
            "message": f"Cache refresh triggered for {data_type}"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Cache refresh failed"
        }


@router.get("/cache/invalidate")
@rate_limit("admin")
async def invalidate_cache(
    request: Request,
    response: Response
) -> Dict[str, Any]:
    """
    Invalidate all caches to force fresh data fetch.
    
    Use with caution - will cause temporary performance degradation.
    """
    try:
        # Clear cache files
        import os
        from pathlib import Path
        
        cache_dir = Path("backend/app/data/cache")
        files_deleted = 0
        
        if cache_dir.exists():
            for cache_file in cache_dir.glob("*.json"):
                if not cache_file.name.startswith("_"):
                    os.remove(cache_file)
                    files_deleted += 1
        
        return {
            "status": "success",
            "files_deleted": files_deleted,
            "message": "Cache invalidated, will fetch fresh data on next request"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

