"""Emergency quick prices endpoint - bypasses all caching logic"""
from fastapi import APIRouter
from pathlib import Path
import json

router = APIRouter(prefix="/quick", tags=["quick"])


@router.get("/prices")
def get_quick_prices():
    """Get prices directly from cache file - NO ASYNC, NO DELAYS"""
    try:
        cache_file = Path(__file__).parent.parent / "data" / "cache" / "prices_major_cryptos.json"
        
        if not cache_file.exists():
            return {"error": "Cache file not found"}
        
        with open(cache_file, 'r') as f:
            data = json.load(f)
        
        # Remove metadata
        result = {k: v for k, v in data.items() if not k.startswith('_')}
        
        return result
    except Exception as e:
        return {"error": str(e)}


@router.get("/market")
def get_quick_market():
    """Get market data directly from cache file - NO ASYNC, NO DELAYS"""
    try:
        cache_file = Path(__file__).parent.parent / "data" / "cache" / "market_data_major_cryptos.json"
        
        if not cache_file.exists():
            return {"error": "Cache file not found"}
        
        with open(cache_file, 'r') as f:
            data = json.load(f)
        
        # Remove metadata
        result = {k: v for k, v in data.items() if not k.startswith('_')}
        
        return result
    except Exception as e:
        return {"error": str(e)}
