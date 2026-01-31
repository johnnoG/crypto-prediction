"""
Prometheus Metrics for Backend Monitoring

Exposes metrics for:
- HTTP request latency
- API call counts
- Model prediction times
- Cache hit rates
- Error rates
"""

from __future__ import annotations

from typing import Optional
import time

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST  # type: ignore[reportMissingImports]
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain"  # Fallback content type
    print("Warning: prometheus-client not installed")


if PROMETHEUS_AVAILABLE:
    # HTTP Metrics
    http_requests_total = Counter(
        'http_requests_total',
        'Total HTTP requests',
        ['method', 'endpoint', 'status_code']
    )
    
    http_request_duration_seconds = Histogram(
        'http_request_duration_seconds',
        'HTTP request latency',
        ['method', 'endpoint']
    )
    
    # API Metrics
    api_calls_total = Counter(
        'api_calls_total',
        'Total external API calls',
        ['api_name', 'endpoint', 'status']
    )
    
    api_call_duration_seconds = Histogram(
        'api_call_duration_seconds',
        'External API call duration',
        ['api_name', 'endpoint']
    )
    
    # Cache Metrics
    cache_hits_total = Counter(
        'cache_hits_total',
        'Total cache hits',
        ['cache_type']
    )
    
    cache_misses_total = Counter(
        'cache_misses_total',
        'Total cache misses',
        ['cache_type']
    )
    
    cache_hit_rate = Gauge(
        'cache_hit_rate',
        'Cache hit rate percentage',
        ['cache_type']
    )
    
    # Model Metrics
    model_predictions_total = Counter(
        'model_predictions_total',
        'Total model predictions',
        ['model_type', 'crypto_id']
    )
    
    model_prediction_duration_seconds = Histogram(
        'model_prediction_duration_seconds',
        'Model prediction duration',
        ['model_type']
    )
    
    model_prediction_error = Gauge(
        'model_prediction_error',
        'Model prediction error (MAPE)',
        ['model_type', 'crypto_id']
    )
    
    # System Metrics
    active_connections = Gauge(
        'active_connections',
        'Number of active connections'
    )
    
    circuit_breaker_state = Gauge(
        'circuit_breaker_state',
        'Circuit breaker state (0=closed, 1=open, 2=half_open)',
        ['circuit_name']
    )
    
    # Application Info
    app_info = Info(
        'app_info',
        'Application information'
    )
    
    app_info.info({
        'version': '1.0.0',
        'name': 'crypto_forecast_api'
    })


class PrometheusMetrics:
    """Helper class for recording Prometheus metrics"""
    
    @staticmethod
    def record_http_request(
        method: str,
        endpoint: str,
        status_code: int,
        duration_seconds: float
    ) -> None:
        """Record HTTP request metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()
        
        http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration_seconds)
    
    @staticmethod
    def record_api_call(
        api_name: str,
        endpoint: str,
        status: str,
        duration_seconds: float
    ) -> None:
        """Record external API call metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        api_calls_total.labels(
            api_name=api_name,
            endpoint=endpoint,
            status=status
        ).inc()
        
        api_call_duration_seconds.labels(
            api_name=api_name,
            endpoint=endpoint
        ).observe(duration_seconds)
    
    @staticmethod
    def record_cache_access(cache_type: str, hit: bool) -> None:
        """Record cache access (hit or miss)"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        if hit:
            cache_hits_total.labels(cache_type=cache_type).inc()
        else:
            cache_misses_total.labels(cache_type=cache_type).inc()
    
    @staticmethod
    def record_model_prediction(
        model_type: str,
        crypto_id: str,
        duration_seconds: float,
        error: Optional[float] = None
    ) -> None:
        """Record model prediction metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        model_predictions_total.labels(
            model_type=model_type,
            crypto_id=crypto_id
        ).inc()
        
        model_prediction_duration_seconds.labels(
            model_type=model_type
        ).observe(duration_seconds)
        
        if error is not None:
            model_prediction_error.labels(
                model_type=model_type,
                crypto_id=crypto_id
            ).set(error)
    
    @staticmethod
    def update_circuit_breaker_state(circuit_name: str, state: str) -> None:
        """Update circuit breaker state metric"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        state_value = {
            'closed': 0,
            'open': 1,
            'half_open': 2
        }.get(state.lower(), 0)
        
        circuit_breaker_state.labels(circuit_name=circuit_name).set(state_value)
    
    @staticmethod
    def get_metrics() -> bytes:
        """Get Prometheus metrics in exposition format"""
        if not PROMETHEUS_AVAILABLE:
            return b"Prometheus not available"
        
        return generate_latest()


# Global metrics instance
prometheus_metrics = PrometheusMetrics()

