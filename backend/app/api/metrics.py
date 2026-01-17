"""
Metrics API Endpoint

Exposes Prometheus metrics for monitoring.
"""

from fastapi import APIRouter, Response

try:
    from services.prometheus_metrics import prometheus_metrics, PROMETHEUS_AVAILABLE, CONTENT_TYPE_LATEST
except ImportError:
    # Fallback if import fails
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain"
    prometheus_metrics = None  # type: ignore


router = APIRouter(tags=["monitoring"])


@router.get("/metrics")
async def get_metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus exposition format.
    """
    if not PROMETHEUS_AVAILABLE or prometheus_metrics is None:
        return Response(
            content="Prometheus not available",
            media_type=CONTENT_TYPE_LATEST
        )
    
    metrics_data = prometheus_metrics.get_metrics()
    
    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST
    )

