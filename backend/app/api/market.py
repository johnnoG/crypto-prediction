from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query, Request, Response

try:
    from services.prices_service import get_market_data_with_cache
except ImportError:
    from services.prices_service import get_market_data_with_cache


router = APIRouter(prefix="/market", tags=["market"])


@router.get("/data")
async def get_market_data_endpoint(
    request: Request,
    response: Response,
    ids: str = Query("bitcoin,ethereum,solana", description="Comma-separated CoinGecko IDs"),
    vs_currency: str = Query("usd", description="Currency for market data"),
) -> Dict[str, Any]:
    """Return market data for requested assets.

    This endpoint exists to match the frontend expectation: `/api/market/data`.
    It prefers cached data but will fall back to live fetches when needed.
    """
    try:
        data = await get_market_data_with_cache(ids=ids, vs_currency=vs_currency)
        return data or {}
    except HTTPException as exc:
        # Avoid surfacing 503 to the UI; return empty data for graceful handling.
        response.headers["X-Data-Status"] = "unavailable"
        response.headers["X-Data-Error"] = str(exc.detail)
        return {}
    except Exception as exc:
        response.headers["X-Data-Status"] = "error"
        response.headers["X-Data-Error"] = str(exc)
        return {}
