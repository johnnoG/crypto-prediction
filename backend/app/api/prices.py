from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request, Response

try:
    from services.prices_service import get_simple_price_with_cache, get_market_data_with_cache
    from clients.coingecko_client import CoinGeckoClient
    from cache import AsyncCache
    from .dependencies.rate_limiter import rate_limit
except ImportError:
    from services.prices_service import get_simple_price_with_cache, get_market_data_with_cache
    from clients.coingecko_client import CoinGeckoClient
    from cache import AsyncCache
    from api.dependencies.rate_limiter import rate_limit  # type: ignore


router = APIRouter(prefix="/prices", tags=["prices"])


@router.get("")
# TEMPORARILY DISABLED RATE LIMITING FOR DEBUGGING
# @rate_limit("prices_default")
def get_prices(  # CHANGED TO SYNC (not async) - fixes response buffering!
    request: Request,
    response: Response,
    ids: str = Query("bitcoin,ethereum,solana", description="Comma-separated CoinGecko IDs, e.g. bitcoin,ethereum"),
    vs_currencies: str = Query("usd", description="Comma-separated vs currencies, e.g. usd,eur"),
) -> Dict[str, Any]:
    """FIX: Sync function with sync I/O - no event loop blocking!"""
    from pathlib import Path
    import json
    
    try:
        cache_file = Path(__file__).parent.parent / "data" / "cache" / "prices_major_cryptos.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Remove metadata
            result = {k: v for k, v in data.items() if not k.startswith('_')}
            
            # Filter to requested IDs  
            ids_list = [id.strip() for id in ids.split(",") if id.strip()]
            filtered = {id: result.get(id, {}) for id in ids_list if id in result}
            
            return filtered if filtered else result
        else:
            return {"error": "Cache file not found"}
    except Exception as e:
        return {"error": str(e)}


@router.get("/market")
# TEMPORARILY DISABLED RATE LIMITING FOR DEBUGGING
# @rate_limit("prices_market")
def get_market_data(  # CHANGED TO SYNC (not async) - fixes response buffering!
    request: Request,
    response: Response,
    ids: str = Query("bitcoin,ethereum,solana", description="Comma-separated CoinGecko IDs"),
    vs_currency: str = Query("usd", description="Currency for market data"),
) -> Dict[str, Any]:
    """FIX: Sync function with sync I/O - no event loop blocking!"""
    from pathlib import Path
    import json
    
    try:
        cache_file = Path(__file__).parent.parent / "data" / "cache" / "market_data_major_cryptos.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Remove metadata
            result = {k: v for k, v in data.items() if not k.startswith('_')}
            
            # Filter to requested IDs
            ids_list = [id.strip() for id in ids.split(",") if id.strip()]
            filtered = {id: result.get(id, {}) for id in ids_list if id in result}
            
            return filtered if filtered else result
        else:
            return {"error": "Cache file not found"}
    except Exception as e:
        return {"error": str(e)}


@router.get("/history/{coin_id}")
@rate_limit("prices_history")
async def get_coin_history(
    request: Request,
    response: Response,
    coin_id: str,
    vs_currency: str = Query("usd", description="Currency for price data"),
    days: int = Query(7, description="Number of days of history (1, 7, 30, 90, 365, max)"),
) -> Dict[str, Any]:
    """Get historical OHLC data for a specific coin."""
    cache = AsyncCache()
    await cache.initialize()
    
    cache_key = f"ohlc:{coin_id}:{vs_currency}:{days}"
    cached_data = await cache.get(cache_key)
    if cached_data and cached_data.get("data"):
        print(f"[SUCCESS] Returning cached OHLC data for {coin_id}")
        return cached_data
    elif cached_data:
        print(f"[WARNING] Cached OHLC data for {coin_id} is empty, refetching")
    
    try:
        client = CoinGeckoClient(timeout_seconds=8.0)
        try:
            ohlc_data = await client.get_coin_ohlc_by_id(
                coin_id=coin_id,
                vs_currency=vs_currency,
                days=days
            )
        finally:
            await client.close()
    except Exception as e:
        print(f"[ERROR] OHLC endpoint error: {e}")
        import traceback
        traceback.print_exc()
        if cached_data and cached_data.get("data"):
            print("[INFO] Returning previously cached OHLC data after failure")
            return cached_data
        raise HTTPException(
            status_code=503,
            detail=f"Unable to fetch historical data for {coin_id}.",
        )
    
    if not ohlc_data:
        raise HTTPException(
            status_code=502,
            detail=f"CoinGecko returned no OHLC data for {coin_id}.",
        )
    
    formatted_data = {
        "coin_id": coin_id,
        "vs_currency": vs_currency,
        "days": days,
        "data": [
            {
                "timestamp": item[0],
                "open": item[1],
                "high": item[2],
                "low": item[3],
                "close": item[4],
            }
            for item in ohlc_data
        ]
    }
    
    await cache.set(cache_key, formatted_data, ttl_seconds=300)
    print(f"[SUCCESS] Fetched OHLC data for {coin_id}: {len(ohlc_data)} data points")
    return formatted_data


