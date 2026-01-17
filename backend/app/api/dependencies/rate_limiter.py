from __future__ import annotations

from typing import Callable

from slowapi import Limiter
from slowapi.util import get_remote_address

try:
    from ...config import get_settings
except ImportError:  # pragma: no cover - fallback for direct execution
    from config import get_settings  # type: ignore


settings = get_settings()

# Global limiter configured once using application settings
limiter = Limiter(
    key_func=get_remote_address,
    headers_enabled=True,
    default_limits=[settings.rate_limit_default],
)

# Human-friendly keys mapped to actual rate limit strings
RATE_LIMITS: dict[str, str] = {
    "prices_default": settings.rate_limit_prices,
    "prices_market": settings.rate_limit_market,
    "prices_history": settings.rate_limit_history,
    "forecasts": settings.rate_limit_forecasts,
    "news_list": settings.rate_limit_news_list,
    "news_refresh": settings.rate_limit_news_refresh,
    "cache_status": settings.rate_limit_cache_status,
}


def rate_limit(name: str) -> Callable:
    """Return a slowapi decorator for the configured rate limit key."""
    limit_value = RATE_LIMITS.get(name, settings.rate_limit_default)
    return limiter.limit(limit_value)


__all__ = ["limiter", "rate_limit", "RATE_LIMITS"]

