from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import polars as pl

from clients.coingecko_client import CoinGeckoClient
from cache import AsyncCache
from config import get_settings
from db import get_db
from models.market import Asset, OHLCV


class DataAggregator:
    """Core data aggregation service for crypto data.
    
    Handles:
    - Price data aggregation from multiple sources
    - OHLCV data retrieval and caching
    - Database integration for historical data
    - Real-time data with fallback to cached data
    """

    def __init__(self):
        self.settings = get_settings()
        self.cache = AsyncCache()
        self.coingecko_client = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the data aggregator."""
        if self._initialized:
            return
            
        await self.cache.initialize()
        self.coingecko_client = CoinGeckoClient()
        self._initialized = True

    async def close(self) -> None:
        """Clean up resources."""
        if self.coingecko_client:
            await self.coingecko_client.close()

    async def get_price(self, symbol: str, use_cache: bool = True) -> Optional[float]:
        """Get current price for a symbol."""
        await self.initialize()
        
        # Normalize symbol
        symbol = symbol.upper()
        
        # Try cache first
        if use_cache:
            cache_key = f"price:{symbol}"
            cached_price = await self.cache.get(cache_key)
            if cached_price is not None:
                return cached_price.get("price")
        
        try:
            # Get from CoinGecko
            coingecko_id = await self._get_coingecko_id(symbol)
            if not coingecko_id:
                return None
                
            data = await self.coingecko_client.get_simple_price(
                ids=[coingecko_id], 
                vs_currencies=["usd"]
            )
            
            if coingecko_id in data and "usd" in data[coingecko_id]:
                price = data[coingecko_id]["usd"]
                
                # Cache the result
                if use_cache:
                    await self.cache.set(
                        cache_key, 
                        {"price": price, "timestamp": datetime.now().isoformat()}, 
                        ttl_seconds=30
                    )
                
                return price
                
        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            
        return None

    async def get_multiple_prices(self, symbols: List[str], use_cache: bool = True) -> Dict[str, Optional[float]]:
        """Get current prices for multiple symbols."""
        await self.initialize()
        
        results = {}
        
        # Get all coingecko IDs
        coingecko_ids = []
        symbol_to_id = {}
        
        for symbol in symbols:
            symbol = symbol.upper()
            coingecko_id = await self._get_coingecko_id(symbol)
            if coingecko_id:
                coingecko_ids.append(coingecko_id)
                symbol_to_id[coingecko_id] = symbol
            else:
                results[symbol] = None
        
        if not coingecko_ids:
            return results
        
        try:
            # Get from CoinGecko
            data = await self.coingecko_client.get_simple_price(
                ids=coingecko_ids, 
                vs_currencies=["usd"]
            )
            
            for coingecko_id, price_data in data.items():
                if "usd" in price_data:
                    symbol = symbol_to_id[coingecko_id]
                    price = price_data["usd"]
                    results[symbol] = price
                    
                    # Cache individual prices
                    if use_cache:
                        cache_key = f"price:{symbol}"
                        await self.cache.set(
                            cache_key, 
                            {"price": price, "timestamp": datetime.now().isoformat()}, 
                            ttl_seconds=30
                        )
            
        except Exception as e:
            print(f"Error getting multiple prices: {e}")
        
        return results

    async def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Get OHLCV data for a symbol."""
        await self.initialize()
        
        symbol = symbol.upper()
        cache_key = f"ohlcv:{symbol}:{timeframe}:{limit}"
        
        # Try cache first
        if use_cache:
            cached_data = await self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Try database first
        db_data = await self._get_ohlcv_from_db(symbol, timeframe, limit)
        if db_data:
            # Cache the result
            if use_cache:
                await self.cache.set(cache_key, db_data, ttl_seconds=300)  # 5 minutes
            return db_data
        
        # Fallback to CoinGecko
        try:
            coingecko_id = await self._get_coingecko_id(symbol)
            if not coingecko_id:
                return []
            
            # Get OHLC data from CoinGecko
            days = min(limit // 24, 30)  # CoinGecko limit
            ohlc_data = await self.coingecko_client.get_coin_ohlc_by_id(
                coingecko_id, 
                vs_currency="usd", 
                days=days
            )
            
            # Convert to our format
            ohlcv_data = []
            for i, ohlc in enumerate(ohlc_data[:limit]):
                timestamp = datetime.fromtimestamp(ohlc[0] / 1000)
                ohlcv_data.append({
                    "timestamp": timestamp.isoformat(),
                    "open": ohlc[1],
                    "high": ohlc[2],
                    "low": ohlc[3],
                    "close": ohlc[4],
                    "volume": 0.0  # CoinGecko OHLC doesn't include volume
                })
            
            # Cache the result
            if use_cache:
                await self.cache.set(cache_key, ohlcv_data, ttl_seconds=300)
            
            return ohlcv_data
            
        except Exception as e:
            print(f"Error getting OHLCV for {symbol}: {e}")
            return []

    async def _get_ohlcv_from_db(self, symbol: str, timeframe: str, limit: int) -> List[Dict[str, Any]]:
        """Get OHLCV data from database."""
        try:
            db = next(get_db())
            
            # Get asset
            asset = db.query(Asset).filter(Asset.symbol == symbol).first()
            if not asset:
                return []
            
            # Get OHLCV data
            ohlcv_records = db.query(OHLCV).filter(
                OHLCV.asset_id == asset.id,
                OHLCV.timeframe == timeframe
            ).order_by(OHLCV.ts.desc()).limit(limit).all()
            
            # Convert to dict format
            data = []
            for record in reversed(ohlcv_records):  # Reverse to get chronological order
                data.append({
                    "timestamp": record.ts.isoformat(),
                    "open": record.open,
                    "high": record.high,
                    "low": record.low,
                    "close": record.close,
                    "volume": record.volume
                })
            
            return data
            
        except Exception as e:
            print(f"Error getting OHLCV from database: {e}")
            return []

    async def _get_coingecko_id(self, symbol: str) -> Optional[str]:
        """Get CoinGecko ID for a symbol."""
        # Common symbol to CoinGecko ID mapping
        symbol_mapping = {
            "BTC": "bitcoin",
            "ETH": "ethereum", 
            "SOL": "solana",
            "ADA": "cardano",
            "DOT": "polkadot",
            "LINK": "chainlink",
            "UNI": "uniswap",
            "AVAX": "avalanche-2",
            "MATIC": "matic-network",
            "ATOM": "cosmos",
            "LTC": "litecoin",
            "BCH": "bitcoin-cash",
            "XRP": "ripple",
            "DOGE": "dogecoin",
            "SHIB": "shiba-inu",
            "TRX": "tron",
            "XLM": "stellar",
            "VET": "vechain",
            "FIL": "filecoin",
            "ICP": "internet-computer"
        }
        
        return symbol_mapping.get(symbol.upper())

    def get_source_health(self) -> Dict[str, Any]:
        """Get health status of all data sources."""
        return {
            "coingecko": {
                "status": "healthy" if self.coingecko_client else "not_initialized",
                "last_check": datetime.now().isoformat()
            },
            "cache": {
                "status": "healthy" if self.cache else "not_initialized",
                "backend": self.cache.backend_name() if self.cache else "unknown"
            },
            "database": {
                "status": "healthy",  # Will be checked by db_ping
                "last_check": datetime.now().isoformat()
            }
        }


# Global instance
_aggregator_instance: Optional[DataAggregator] = None


async def get_aggregator() -> DataAggregator:
    """Get or create the global data aggregator instance."""
    global _aggregator_instance
    
    if _aggregator_instance is None:
        _aggregator_instance = DataAggregator()
        await _aggregator_instance.initialize()
    
    return _aggregator_instance