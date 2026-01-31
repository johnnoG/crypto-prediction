"""
Rate Limit Monitoring API

Provides endpoints to monitor API rate limiting, batching, and performance.
"""

from __future__ import annotations

from fastapi import APIRouter
from typing import Dict, Any

try:
    from ..services.rate_limit_manager import rate_limit_manager
    from ..services.request_batcher import request_batcher
except ImportError:
    from services.rate_limit_manager import rate_limit_manager
    from services.request_batcher import request_batcher


router = APIRouter(prefix="/rate-limit", tags=["rate-limit"])


@router.get("/status")
async def get_rate_limit_status() -> Dict[str, Any]:
    """
    Get current rate limit status for all APIs.
    
    Returns comprehensive statistics including:
    - Current usage vs limits
    - Success rates
    - Rate limit predictions
    - Auto-upgrade recommendations
    """
    try:
        stats = rate_limit_manager.get_all_statistics()
        
        return {
            "success": True,
            "apis": stats,
            "summary": {
                "total_apis": len(stats),
                "apis_near_limit": sum(
                    1 for api in stats.values()
                    if api["usage"]["utilization_percent"] > 80
                ),
                "apis_should_upgrade": sum(
                    1 for api in stats.values()
                    if api["predictions"]["should_upgrade"]
                ),
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@router.get("/status/{api_name}")
async def get_api_rate_limit_status(api_name: str) -> Dict[str, Any]:
    """Get detailed rate limit status for a specific API."""
    try:
        stats = rate_limit_manager.get_statistics(api_name)
        config = rate_limit_manager.configs.get(api_name)
        
        if not config:
            return {
                "success": False,
                "error": f"API '{api_name}' not found"
            }
        
        return {
            "success": True,
            "api_name": api_name,
            "config": {
                "requests_per_minute": config.requests_per_minute,
                "requests_per_hour": config.requests_per_hour,
                "requests_per_day": config.requests_per_day,
                "burst_limit": config.burst_limit,
                "min_interval_ms": config.min_interval_ms,
                "free_tier": config.free_tier,
                "pro_cost_per_month": config.pro_cost_per_month,
            },
            "current_usage": {
                "last_minute": stats.requests_last_minute,
                "last_hour": stats.requests_last_hour,
                "last_day": stats.requests_last_day,
                "utilization_percent": (
                    stats.requests_last_minute / config.requests_per_minute * 100
                ),
            },
            "performance": {
                "total_requests": stats.total_requests,
                "successful_requests": stats.successful_requests,
                "failed_requests": stats.failed_requests,
                "rate_limited_requests": stats.rate_limited_requests,
                "success_rate_percent": stats.success_rate() * 100,
                "average_duration_ms": stats.average_duration_ms(),
            },
            "predictions": {
                "predicted_rpm": stats.predicted_rpm,
                "predicted_time_to_limit_seconds": stats.predicted_time_to_limit,
                "should_upgrade_to_pro": rate_limit_manager._should_upgrade_api(api_name),
            },
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@router.get("/batching/stats")
async def get_batching_statistics() -> Dict[str, Any]:
    """Get request batching statistics."""
    try:
        stats = request_batcher.get_statistics()
        
        return {
            "success": True,
            "batching": stats,
            "insights": {
                "total_api_calls": stats["total_requests"] - stats["api_calls_saved"],
                "efficiency_rating": (
                    "Excellent" if stats["efficiency_percent"] > 50 else
                    "Good" if stats["efficiency_percent"] > 30 else
                    "Fair" if stats["efficiency_percent"] > 10 else
                    "Poor"
                ),
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@router.get("/recommendations")
async def get_optimization_recommendations() -> Dict[str, Any]:
    """
    Get personalized recommendations for API optimization.
    
    Analyzes current usage patterns and suggests:
    - Which APIs should be upgraded
    - Opportunities for better batching
    - Priority queue adjustments
    """
    try:
        all_stats = rate_limit_manager.get_all_statistics()
        batch_stats = request_batcher.get_statistics()
        
        recommendations = []
        
        # Check for APIs that should be upgraded
        for api_name, stats in all_stats.items():
            if stats["predictions"]["should_upgrade"]:
                config = rate_limit_manager.configs.get(api_name)
                if config and config.free_tier:
                    recommendations.append({
                        "type": "upgrade",
                        "api": api_name,
                        "priority": "high",
                        "reason": f"Frequently hitting rate limits ({stats['performance']['rate_limited_requests']} times)",
                        "action": f"Consider upgrading to Pro tier (${config.pro_cost_per_month}/month)",
                        "expected_benefit": f"Increase from {config.requests_per_minute} to {config.pro_requests_per_minute} req/min",
                    })
        
        # Check for APIs approaching limits
        for api_name, stats in all_stats.items():
            util = stats["usage"]["utilization_percent"]
            if util > 80 and not stats["predictions"]["should_upgrade"]:
                recommendations.append({
                    "type": "optimization",
                    "api": api_name,
                    "priority": "medium",
                    "reason": f"High utilization ({util:.1f}%)",
                    "action": "Enable request batching or implement caching",
                    "expected_benefit": "Reduce API calls by 20-50%",
                })
        
        # Check batching efficiency
        if batch_stats["total_requests"] > 100:
            efficiency = batch_stats["efficiency_percent"]
            if efficiency < 20:
                recommendations.append({
                    "type": "batching",
                    "api": "all",
                    "priority": "low",
                    "reason": f"Low batching efficiency ({efficiency:.1f}%)",
                    "action": "Increase batch window or review request patterns",
                    "expected_benefit": "Potential to reduce API calls by additional 30%",
                })
        
        # Check success rates
        for api_name, stats in all_stats.items():
            success_rate = stats["performance"]["success_rate"]
            if success_rate < 95 and stats["performance"]["total_requests"] > 10:
                recommendations.append({
                    "type": "reliability",
                    "api": api_name,
                    "priority": "high",
                    "reason": f"Low success rate ({success_rate:.1f}%)",
                    "action": "Investigate errors or increase retry attempts",
                    "expected_benefit": "Improve reliability and user experience",
                })
        
        return {
            "success": True,
            "total_recommendations": len(recommendations),
            "recommendations": sorted(
                recommendations,
                key=lambda x: {"high": 0, "medium": 1, "low": 2}[x["priority"]]
            ),
            "overall_health": (
                "Excellent" if len([r for r in recommendations if r["priority"] == "high"]) == 0 else
                "Good" if len([r for r in recommendations if r["priority"] == "high"]) <= 1 else
                "Needs Attention"
            ),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@router.post("/reset-stats")
async def reset_statistics() -> Dict[str, Any]:
    """Reset all rate limiting statistics (admin only)."""
    try:
        # Reset rate limiter stats
        for api_name in rate_limit_manager.configs:
            rate_limit_manager.request_history[api_name].clear()
            rate_limit_manager.statistics[api_name] = type(
                rate_limit_manager.statistics[api_name]
            )()
        
        # Reset batcher stats
        request_batcher.stats = {
            "total_requests": 0,
            "batched_requests": 0,
            "api_calls_saved": 0,
            "average_batch_size": 0.0,
        }
        
        return {
            "success": True,
            "message": "Statistics reset successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

