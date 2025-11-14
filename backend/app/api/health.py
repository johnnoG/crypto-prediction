"""
Health Check Endpoints

Provides endpoints for monitoring service health, database connectivity,
and cache availability.
"""

import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.database import get_db
from app.cache import get_cache, CacheManager
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.

    Returns:
        Dictionary with service status and timestamp

    Example:
        GET /health
        Response: {"status": "healthy", "timestamp": "2024-01-01T00:00:00"}
    """
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/health/db", status_code=status.HTTP_200_OK)
async def database_health_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Database health check endpoint.

    Verifies PostgreSQL connectivity and query execution.

    Args:
        db: Database session (injected)

    Returns:
        Dictionary with database health status

    Example:
        GET /health/db
        Response: {
            "status": "healthy",
            "database": "connected",
            "latency_ms": 5.2
        }
    """
    try:
        # Measure query latency
        start_time = datetime.utcnow()
        result = db.execute(text("SELECT 1"))
        result.fetchone()
        end_time = datetime.utcnow()

        latency_ms = (end_time - start_time).total_seconds() * 1000

        return {
            "status": "healthy",
            "database": "connected",
            "latency_ms": round(latency_ms, 2),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


@router.get("/health/cache", status_code=status.HTTP_200_OK)
async def cache_health_check(
    cache: CacheManager = Depends(get_cache)
) -> Dict[str, Any]:
    """
    Redis cache health check endpoint.

    Verifies Redis connectivity and retrieves metrics.

    Args:
        cache: Cache manager instance (injected)

    Returns:
        Dictionary with cache health status and metrics

    Example:
        GET /health/cache
        Response: {
            "status": "healthy",
            "connected": true,
            "used_memory_mb": 12.5,
            "connected_clients": 3
        }
    """
    try:
        health_info = cache.health_check()
        health_info["timestamp"] = datetime.utcnow().isoformat()
        return health_info
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        return {
            "status": "unhealthy",
            "connected": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


@router.get("/health/detailed", status_code=status.HTTP_200_OK)
async def detailed_health_check(
    db: Session = Depends(get_db),
    cache: CacheManager = Depends(get_cache),
) -> Dict[str, Any]:
    """
    Comprehensive health check endpoint.

    Checks all service dependencies including database and cache.

    Args:
        db: Database session (injected)
        cache: Cache manager instance (injected)

    Returns:
        Dictionary with comprehensive health status

    Example:
        GET /health/detailed
        Response: {
            "status": "healthy",
            "service": {...},
            "database": {...},
            "cache": {...}
        }
    """
    # Check database
    db_healthy = True
    db_info = {}
    try:
        start_time = datetime.utcnow()
        result = db.execute(text("SELECT 1"))
        result.fetchone()
        end_time = datetime.utcnow()
        latency_ms = (end_time - start_time).total_seconds() * 1000

        db_info = {
            "status": "healthy",
            "connected": True,
            "latency_ms": round(latency_ms, 2),
        }
    except Exception as e:
        db_healthy = False
        db_info = {
            "status": "unhealthy",
            "connected": False,
            "error": str(e),
        }
        logger.error(f"Database health check failed: {e}")

    # Check cache
    cache_healthy = True
    cache_info = {}
    try:
        cache_info = cache.health_check()
        if cache_info["status"] != "healthy":
            cache_healthy = False
    except Exception as e:
        cache_healthy = False
        cache_info = {
            "status": "unhealthy",
            "connected": False,
            "error": str(e),
        }
        logger.error(f"Cache health check failed: {e}")

    # Overall status
    overall_status = "healthy" if (db_healthy and cache_healthy) else "degraded"
    if not db_healthy:
        overall_status = "unhealthy"

    return {
        "status": overall_status,
        "service": {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": "development" if settings.debug else "production",
        },
        "database": db_info,
        "cache": cache_info,
        "timestamp": datetime.utcnow().isoformat(),
    }
