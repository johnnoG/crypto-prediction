from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Any
import asyncio

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

# Note: We use absolute imports for better compatibility
import sys
import os
from pathlib import Path

# Add the backend/app directory to Python path
backend_app_dir = Path(__file__).parent
if str(backend_app_dir) not in sys.path:
    sys.path.insert(0, str(backend_app_dir))

try:
    from .config import get_settings, Settings
    from .api import prices_router, cache_router, db_router, news_router, features_router, crypto_data_router, forecasts_router, stream_router, quick_prices_router
    from .api.auth import router as auth_router
    from .api.rate_limit_monitor import router as rate_limit_router
    from .api.health import router as health_router
    from .api.metrics import router as metrics_router
    from .api.admin import router as admin_router
    from .api.dependencies.rate_limiter import limiter
    from .services.smart_cache_service import smart_cache
    from .services.rate_limit_manager import rate_limit_manager
except ImportError:
    # When running as script, use absolute imports
    from config import get_settings, Settings
    from api import prices_router, cache_router, db_router, news_router, features_router, crypto_data_router, forecasts_router, stream_router, quick_prices_router
    from api.auth import router as auth_router
    from api.rate_limit_monitor import router as rate_limit_router
    from api.health import router as health_router
    from api.metrics import router as metrics_router
    from api.admin import router as admin_router
    from api.dependencies.rate_limiter import limiter  # type: ignore
    from services.smart_cache_service import smart_cache
    from services.rate_limit_manager import rate_limit_manager

from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler
from contextlib import asynccontextmanager

# Import connection pool and circuit breaker
try:
    from .utils.connection_pool import connection_pool
    from .utils.circuit_breaker import coingecko_circuit_breaker
except ImportError:
    from utils.connection_pool import connection_pool
    from utils.circuit_breaker import coingecko_circuit_breaker


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app"""
    # Startup - with timeouts to prevent hanging
    print("[STARTUP] Initializing Sentry error tracking...")
    try:
        from services.sentry_config import init_sentry
        init_sentry(environment="production", traces_sample_rate=0.1)
    except Exception as e:
        print(f"[WARNING] Sentry initialization failed: {e}")
    
    print("[STARTUP] Initializing connection pool...")
    try:
        await asyncio.wait_for(connection_pool.get_client(), timeout=5.0)
    except asyncio.TimeoutError:
        print("[WARNING] Connection pool initialization timed out, continuing anyway")
    except Exception as e:
        print(f"[WARNING] Connection pool initialization failed: {e}, continuing anyway")
    
    print("[STARTUP] Initializing smart cache...")
    try:
        # Initialize with timeout - don't block startup if cache init hangs
        await asyncio.wait_for(smart_cache.initialize(), timeout=10.0)
    except asyncio.TimeoutError:
        print("[WARNING] Smart cache initialization timed out, continuing anyway")
    except Exception as e:
        print(f"[WARNING] Smart cache initialization failed: {e}, continuing anyway")
    
    print("[STARTUP] Starting rate limit manager...")
    try:
        await asyncio.wait_for(rate_limit_manager.start(), timeout=5.0)
    except asyncio.TimeoutError:
        print("[WARNING] Rate limit manager startup timed out, continuing anyway")
    except Exception as e:
        print(f"[WARNING] Rate limit manager startup failed: {e}, continuing anyway")
    
    print("[STARTUP] Backend ready - FIX APPLIED: sync endpoints, instant response, no buffering!")
    
    yield
    
    # Shutdown
    print("[SHUTDOWN] Cleaning up connection pool...")
    try:
        await asyncio.wait_for(connection_pool.close(), timeout=5.0)
    except Exception as e:
        print(f"[WARNING] Connection pool cleanup failed: {e}")
    
    print("[SHUTDOWN] Cleaning up smart cache...")
    try:
        await asyncio.wait_for(smart_cache.cleanup(), timeout=5.0)
    except Exception as e:
        print(f"[WARNING] Smart cache cleanup failed: {e}")
    
    print("[SHUTDOWN] Backend shutdown complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings: Settings = settings or get_settings()

    app = FastAPI(
        title="Crypto Forecast & Real‑Time Dashboard API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # Attach rate limiter (slowapi)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    
    # Gzip compression for API responses (reduces bandwidth by ~70%)
    from fastapi.middleware.gzip import GZipMiddleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # CORS - Allow both localhost and 127.0.0.1 with comprehensive headers including WebSocket
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
        allow_headers=[
            "Accept",
            "Accept-Language",
            "Authorization",
            "Content-Type",
            "Origin",
            "X-Requested-With",
            "Sec-WebSocket-Key",
            "Sec-WebSocket-Version",
            "Sec-WebSocket-Extensions",
            "Sec-WebSocket-Protocol", 
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "Origin",
            "Access-Control-Request-Method",
            "Access-Control-Request-Headers",
            "Cache-Control",
            "Pragma",
            "If-Modified-Since",
            "If-None-Match",
        ],
        expose_headers=[
            "Access-Control-Allow-Origin",
            "Access-Control-Allow-Credentials",
            "Access-Control-Allow-Methods",
            "Access-Control-Allow-Headers",
        ],
    )

    # Custom CORS handler for all routes
    @app.middleware("http")
    async def add_cors_headers(request, call_next):
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Accept, Accept-Language, Content-Language, Content-Type, Authorization, X-Requested-With, Origin, Access-Control-Request-Method, Access-Control-Request-Headers, Cache-Control, Pragma, If-Modified-Since, If-None-Match"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response

    @app.get("/health", tags=["system"])
    def healthcheck() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/quick", tags=["system"])
    def quick_healthcheck() -> Dict[str, str]:
        """Quick health check that doesn't require external dependencies."""
        return {"status": "ok", "timestamp": str(datetime.utcnow())}

    @app.get("/health/api", tags=["system"])
    async def api_healthcheck() -> Dict[str, str]:
        """Health check that tests external API connectivity."""
        try:
            from clients.coingecko_client import CoinGeckoClient
            client = CoinGeckoClient(timeout_seconds=3.0)
            try:
                # Quick test with minimal data
                await client.get_simple_price(ids=["bitcoin"], vs_currencies=["usd"])
                return {"status": "ok", "api": "connected", "timestamp": str(datetime.utcnow())}
            finally:
                await client.close()
        except Exception as e:
            return {"status": "degraded", "api": "timeout", "error": str(e), "timestamp": str(datetime.utcnow())}

    @app.get("/cache/smart", tags=["system"])
    @limiter.limit(app_settings.rate_limit_cache_status)
    async def smart_cache_status(request: Request, response: Response) -> Dict[str, Any]:
        """Get smart cache status and statistics."""
        try:
            stats = smart_cache.get_cache_stats()
            return {
                "status": "ok",
                "smart_cache": stats,
                "timestamp": str(datetime.utcnow())
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": str(datetime.utcnow())
            }

    # Routers
    app.include_router(auth_router)  # Auth routes first
    app.include_router(health_router)  # Health monitoring
    app.include_router(metrics_router)  # Prometheus metrics
    app.include_router(admin_router)  # Admin functions
    app.include_router(quick_prices_router)  # EMERGENCY: Fast cache-only endpoints
    app.include_router(prices_router)
    app.include_router(cache_router)
    app.include_router(db_router)
    app.include_router(news_router)
    app.include_router(features_router)
    app.include_router(crypto_data_router)
    app.include_router(forecasts_router)
    app.include_router(stream_router)
    app.include_router(rate_limit_router)

    # Note: Startup/shutdown now handled by lifespan context manager above
    # This provides better cleanup and prevents connection leaks

    return app


app = create_app()


if __name__ == "__main__":
    # Running as a script for Windows-friendly local dev
    import uvicorn

    # Note: reload requires an import string. When running as a script, disable reload.
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)


