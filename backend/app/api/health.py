"""
Health Monitoring API Endpoints

Provides detailed health information about the backend system.
"""

from __future__ import annotations

from typing import Dict, Any
from fastapi import APIRouter, Request, Response

try:
    from services.health_monitor import health_monitor
    from api.dependencies.rate_limiter import rate_limit
except ImportError:
    from services.health_monitor import health_monitor
    from api.dependencies.rate_limiter import rate_limit  # type: ignore


router = APIRouter(prefix="/health", tags=["health"])


@router.get("/detailed")
@rate_limit("health")
async def get_detailed_health(request: Request, response: Response) -> Dict[str, Any]:
    """
    Get comprehensive health status of all system components.
    
    Returns detailed information about:
    - API performance
    - Cache system
    - Connection pools
    - Circuit breakers
    - System resources
    """
    return await health_monitor.get_system_health()


@router.get("/trends")
@rate_limit("health")
async def get_health_trends(
    request: Request,
    response: Response,
    hours: int = 24
) -> Dict[str, Any]:
    """
    Get health trends over time.
    
    Args:
        hours: Number of hours to analyze (default: 24)
    """
    return health_monitor.get_health_trends(hours=hours)


@router.get("/components/{component_name}")
@rate_limit("health")
async def get_component_health(
    component_name: str,
    request: Request,
    response: Response
) -> Dict[str, Any]:
    """
    Get health status of a specific component.
    
    Args:
        component_name: Name of component (api, cache, connection_pool, etc.)
    """
    all_health = await health_monitor.get_system_health()
    
    component = all_health['components'].get(component_name)
    
    if component:
        return {
            'component': component_name,
            **component
        }
    else:
        return {
            'component': component_name,
            'status': 'not_found',
            'message': f"Component '{component_name}' not found"
        }

