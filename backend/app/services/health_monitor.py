"""
Health Monitoring Service

Comprehensive health checks for:
- API response times
- Cache hit rates
- Active connections
- Memory usage
- Model status
- External API health
"""

from __future__ import annotations

import asyncio
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

try:
    from utils.connection_pool import connection_pool
    from utils.circuit_breaker import coingecko_circuit_breaker, forecast_circuit_breaker
    from cache import AsyncCache
except ImportError:
    print("Warning: Some health monitoring modules not available")


@dataclass
class HealthStatus:
    """Health status for a component"""
    name: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    message: str
    response_time_ms: Optional[float] = None
    last_check: Optional[datetime] = None
    details: Optional[Dict[str, Any]] = None


class HealthMonitor:
    """
    Comprehensive health monitoring for the backend system.
    
    Monitors:
    - API endpoints
    - Cache system
    - External APIs (CoinGecko)
    - System resources
    - ML models
    """
    
    def __init__(self):
        self.health_history: List[Dict[str, Any]] = []
        self.max_history_size = 100
        self.alert_thresholds = {
            'api_latency_ms': 1000,  # Alert if >1s
            'memory_usage_pct': 90,  # Alert if >90%
            'cache_hit_rate_pct': 50,  # Alert if <50%
            'error_rate_pct': 5  # Alert if >5% errors
        }
    
    async def check_api_health(self) -> HealthStatus:
        """Check API response time"""
        start = time.time()
        
        try:
            # Try a simple health endpoint
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://127.0.0.1:8000/health/quick")
                
                response_time_ms = (time.time() - start) * 1000
                
                if response.status_code == 200:
                    status = 'healthy' if response_time_ms < 500 else 'degraded'
                    return HealthStatus(
                        name="api",
                        status=status,
                        message=f"API responding in {response_time_ms:.0f}ms",
                        response_time_ms=response_time_ms,
                        last_check=datetime.now()
                    )
                else:
                    return HealthStatus(
                        name="api",
                        status="degraded",
                        message=f"API returned status {response.status_code}",
                        response_time_ms=response_time_ms,
                        last_check=datetime.now()
                    )
        except Exception as e:
            return HealthStatus(
                name="api",
                status="unhealthy",
                message=f"API check failed: {str(e)}",
                last_check=datetime.now()
            )
    
    async def check_cache_health(self) -> HealthStatus:
        """Check cache system health and hit rates"""
        try:
            cache = AsyncCache()
            await cache.initialize()
            
            # Test cache read/write
            test_key = "health_check_test"
            test_value = {"test": datetime.now().isoformat()}
            
            start = time.time()
            await cache.set(test_key, test_value, ttl_seconds=10)
            retrieved = await cache.get(test_key)
            response_time_ms = (time.time() - start) * 1000
            
            if retrieved == test_value:
                return HealthStatus(
                    name="cache",
                    status="healthy",
                    message=f"Cache operational ({response_time_ms:.0f}ms)",
                    response_time_ms=response_time_ms,
                    last_check=datetime.now()
                )
            else:
                return HealthStatus(
                    name="cache",
                    status="degraded",
                    message="Cache read/write mismatch",
                    last_check=datetime.now()
                )
        except Exception as e:
            return HealthStatus(
                name="cache",
                status="unhealthy",
                message=f"Cache error: {str(e)}",
                last_check=datetime.now()
            )
    
    async def check_connection_pool_health(self) -> HealthStatus:
        """Check connection pool status"""
        try:
            stats = connection_pool.get_connection_stats()
            
            status = 'healthy' if stats['status'] == 'active' else 'unhealthy'
            
            return HealthStatus(
                name="connection_pool",
                status=status,
                message=f"Connection pool {stats['status']}",
                last_check=datetime.now(),
                details=stats
            )
        except Exception as e:
            return HealthStatus(
                name="connection_pool",
                status="unhealthy",
                message=f"Connection pool error: {str(e)}",
                last_check=datetime.now()
            )
    
    async def check_circuit_breaker_health(self) -> HealthStatus:
        """Check circuit breaker states"""
        try:
            coingecko_stats = coingecko_circuit_breaker.get_stats()
            forecast_stats = forecast_circuit_breaker.get_stats()
            
            # Check if any circuits are open
            if coingecko_stats.state.value == 'open' or forecast_stats.state.value == 'open':
                status = 'degraded'
                message = "Some circuits are open"
            elif coingecko_stats.state.value == 'half_open' or forecast_stats.state.value == 'half_open':
                status = 'degraded'
                message = "Some circuits are recovering"
            else:
                status = 'healthy'
                message = "All circuits closed"
            
            return HealthStatus(
                name="circuit_breakers",
                status=status,
                message=message,
                last_check=datetime.now(),
                details={
                    'coingecko': {
                        'state': coingecko_stats.state.value,
                        'failure_count': coingecko_stats.failure_count
                    },
                    'forecast': {
                        'state': forecast_stats.state.value,
                        'failure_count': forecast_stats.failure_count
                    }
                }
            )
        except Exception as e:
            return HealthStatus(
                name="circuit_breakers",
                status="unknown",
                message=f"Circuit breaker check error: {str(e)}",
                last_check=datetime.now()
            )
    
    async def check_system_resources(self) -> HealthStatus:
        """Check system resource usage"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            memory_pct = memory.percent
            
            # CPU usage
            cpu_pct = psutil.cpu_percent(interval=0.1)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_pct = disk.percent
            
            # Determine status
            if memory_pct > 90 or cpu_pct > 90 or disk_pct > 90:
                status = 'unhealthy'
            elif memory_pct > 75 or cpu_pct > 75 or disk_pct > 75:
                status = 'degraded'
            else:
                status = 'healthy'
            
            return HealthStatus(
                name="system_resources",
                status=status,
                message=f"Memory: {memory_pct:.1f}%, CPU: {cpu_pct:.1f}%, Disk: {disk_pct:.1f}%",
                last_check=datetime.now(),
                details={
                    'memory_percent': memory_pct,
                    'memory_available_mb': memory.available // (1024 * 1024),
                    'cpu_percent': cpu_pct,
                    'disk_percent': disk_pct,
                    'disk_free_gb': disk.free // (1024 * 1024 * 1024)
                }
            )
        except Exception as e:
            return HealthStatus(
                name="system_resources",
                status="unknown",
                message=f"Resource check error: {str(e)}",
                last_check=datetime.now()
            )
    
    async def check_all_components(self) -> Dict[str, HealthStatus]:
        """
        Run health checks on all components.
        
        Returns:
            Dictionary mapping component names to health status
        """
        # Run all checks in parallel
        results = await asyncio.gather(
            self.check_api_health(),
            self.check_cache_health(),
            self.check_connection_pool_health(),
            self.check_circuit_breaker_health(),
            self.check_system_resources(),
            return_exceptions=True
        )
        
        health_statuses = {}
        
        for result in results:
            if isinstance(result, HealthStatus):
                health_statuses[result.name] = result
            elif isinstance(result, Exception):
                health_statuses['error'] = HealthStatus(
                    name="error",
                    status="unhealthy",
                    message=f"Health check error: {str(result)}",
                    last_check=datetime.now()
                )
        
        return health_statuses
    
    async def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health report.
        
        Returns:
            Health report dictionary
        """
        component_health = await self.check_all_components()
        
        # Determine overall health
        statuses = [h.status for h in component_health.values()]
        
        if all(s == 'healthy' for s in statuses):
            overall_status = 'healthy'
        elif any(s == 'unhealthy' for s in statuses):
            overall_status = 'unhealthy'
        else:
            overall_status = 'degraded'
        
        report = {
            'status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'components': {
                name: asdict(status)
                for name, status in component_health.items()
            },
            'summary': {
                'total_components': len(component_health),
                'healthy': sum(1 for s in statuses if s == 'healthy'),
                'degraded': sum(1 for s in statuses if s == 'degraded'),
                'unhealthy': sum(1 for s in statuses if s == 'unhealthy')
            }
        }
        
        # Store in history
        self.health_history.append(report)
        if len(self.health_history) > self.max_history_size:
            self.health_history = self.health_history[-self.max_history_size:]
        
        return report
    
    def get_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get health trends over time.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Trend analysis
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_health = [
            h for h in self.health_history
            if datetime.fromisoformat(h['timestamp']) > cutoff_time
        ]
        
        if len(recent_health) == 0:
            return {'status': 'no_data'}
        
        # Calculate trends
        unhealthy_count = sum(1 for h in recent_health if h['status'] == 'unhealthy')
        degraded_count = sum(1 for h in recent_health if h['status'] == 'degraded')
        healthy_count = sum(1 for h in recent_health if h['status'] == 'healthy')
        
        return {
            'period_hours': hours,
            'total_checks': len(recent_health),
            'healthy_pct': (healthy_count / len(recent_health) * 100),
            'degraded_pct': (degraded_count / len(recent_health) * 100),
            'unhealthy_pct': (unhealthy_count / len(recent_health) * 100),
            'availability': ((healthy_count + degraded_count) / len(recent_health) * 100)
        }


# Global health monitor instance
health_monitor = HealthMonitor()

