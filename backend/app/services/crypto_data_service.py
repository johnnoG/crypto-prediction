from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from services.data_aggregator import get_aggregator
from services.feature_engineering import get_feature_pipeline
from clients.coingecko_client import CoinGeckoClient
from clients.crypto_news_client import CryptoNewsClient
from clients.crypto_social_aggregator import CryptoSocialAggregator
from clients.twitter_crypto_client import TwitterCryptoClient
from cache import AsyncCache
try:
    from config import get_settings
except ImportError:
    from config import get_settings


class CryptoDataService:
    """Main crypto data service that aggregates data from multiple sources."""
    
    def __init__(self):
        self.settings = get_settings()
        self.cache = AsyncCache()
        self.coingecko_client = None
        self.news_client = None
        self.social_aggregator = None
        self.twitter_client = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the crypto data service."""
        if self._initialized:
            return
            
        await self.cache.initialize()
        
        # Initialize clients
        self.coingecko_client = CoinGeckoClient()
        self.news_client = CryptoNewsClient()
        self.social_aggregator = CryptoSocialAggregator()
        self.twitter_client = TwitterCryptoClient()
        
        self._initialized = True
    
    async def close(self) -> None:
        """Clean up resources."""
        if self.coingecko_client:
            await self.coingecko_client.close()
        if self.news_client:
            await self.news_client.close()
        if self.social_aggregator:
            await self.social_aggregator.close()
        if self.twitter_client:
            await self.twitter_client.close()
    
    async def get_crypto_dashboard_data(self, background_tasks=None) -> Dict[str, Any]:
        """Get comprehensive crypto dashboard data."""
        await self.initialize()
        
        # Get data from multiple sources in parallel
        tasks = [
            self._get_price_data(),
            self._get_market_summary(),
            self._get_real_social_sentiment(),
            self.get_crypto_news(source="all", limit=10)
        ]
        
        try:
            price_data, market_summary, social_data, news_data = await asyncio.gather(*tasks)
        except Exception as e:
            print(f"Error getting dashboard data: {e}")
            # Return partial data if some sources fail
            price_data = await self._get_price_data()
            market_summary = await self._get_market_summary()
            social_data = {"error": str(e)}
            news_data = {"error": str(e)}
        
        return {
            "timestamp": datetime.now().isoformat(),
            "prices": price_data,
            "market_summary": market_summary,
            "social_sentiment": social_data,
            "news": news_data,
            "sources": {
                "prices": "CoinGecko API",
                "market_summary": "CoinGecko API",
                "social_sentiment": "Reddit, CryptoPanic, RSS",
                "news": "CoinDesk, CoinTelegraph, Decrypt, The Block"
            }
        }
    
    async def _get_price_data(self) -> Dict[str, Any]:
        """Get current price data."""
        try:
            # Major cryptocurrencies
            symbols = ["BTC", "ETH", "BNB", "SOL", "ADA", "DOT", "LINK", "UNI", "AVAX", "MATIC", "ATOM"]
            
            # Get prices from aggregator
            aggregator = await get_aggregator()
            prices = await aggregator.get_multiple_prices(symbols)
            
            # Get market data from CoinGecko
            market_data = await self.coingecko_client.get_coins_markets(
                vs_currency=["usd"],
                ids=["bitcoin", "ethereum", "solana", "cardano", "polkadot"],
                per_page=5,
                order="market_cap_desc"
            )
            
            return {
                "prices": prices,
                "market_data": market_data,
                "source": "CoinGecko API",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting price data: {e}")
            return {"error": str(e), "source": "CoinGecko API"}
    
    async def _get_market_summary(self) -> Dict[str, Any]:
        """Get market summary and trends."""
        try:
            # Get global market data
            market_data = await self.coingecko_client.get_coins_markets(
                vs_currency=["usd"],
                per_page=20,
                order="market_cap_desc",
                price_change_percentage=["1h", "24h", "7d"]
            )
            
            # Calculate summary statistics
            total_market_cap = sum(coin.get("market_cap", 0) for coin in market_data)
            total_volume = sum(coin.get("total_volume", 0) for coin in market_data)
            
            # Count gainers/losers
            gainers_24h = sum(1 for coin in market_data if coin.get("price_change_percentage_24h", 0) > 0)
            losers_24h = len(market_data) - gainers_24h
            
            return {
                "total_market_cap": total_market_cap,
                "total_volume": total_volume,
                "gainers_24h": gainers_24h,
                "losers_24h": losers_24h,
                "top_gainers": sorted(market_data, key=lambda x: x.get("price_change_percentage_24h", 0), reverse=True)[:5],
                "top_losers": sorted(market_data, key=lambda x: x.get("price_change_percentage_24h", 0))[:5],
                "source": "CoinGecko API",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting market summary: {e}")
            return {"error": str(e), "source": "CoinGecko API"}
    
    async def _get_real_social_sentiment(self) -> Dict[str, Any]:
        """Get real social media sentiment data."""
        try:
            # Get sentiment from social aggregator
            sentiment_data = await self.social_aggregator.get_crypto_sentiment_summary()
            
            return {
                "summary": sentiment_data,
                "source": "Reddit, CryptoPanic, RSS",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting social sentiment: {e}")
            return {"error": str(e), "source": "Reddit, CryptoPanic, RSS"}
    
    async def get_crypto_news(self, source: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
        """Get crypto news from various sources."""
        await self.initialize()
        
        try:
            if source == "all" or source is None:
                # Get news from all sources
                news_data = await self.news_client.get_all_news(limit=limit)
            else:
                # Get news from specific source
                news_data = await self.news_client.get_news_by_source(source, limit=limit)
            
            return {
                "articles": news_data,
                "source": source or "all",
                "total": len(news_data),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting crypto news: {e}")
            return {"error": str(e), "source": source or "all"}
    
    async def get_crypto_sentiment(self, source: Optional[str] = None) -> Dict[str, Any]:
        """Get crypto sentiment data."""
        await self.initialize()
        
        try:
            if source == "reddit":
                sentiment_data = await self.social_aggregator.crawl_reddit_crypto(max_posts=50)
            elif source == "cryptopanic":
                sentiment_data = await self.social_aggregator.crawl_cryptopanic_news(max_posts=50)
            else:
                # Get from all sources
                sentiment_data = await self.social_aggregator.get_crypto_sentiment_summary()
            
            return {
                "sentiment": sentiment_data,
                "source": source or "all",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting crypto sentiment: {e}")
            return {"error": str(e), "source": source or "all"}
    
    async def _get_twitter_sentiment(self) -> Dict[str, Any]:
        """Get Twitter sentiment data."""
        try:
            # Get Twitter sentiment
            twitter_data = await self.twitter_client.get_crypto_sentiment()
            
            return {
                "twitter_sentiment": twitter_data,
                "source": "Twitter via Nitter",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting Twitter sentiment: {e}")
            return {"error": str(e), "source": "Twitter via Nitter"}


# Global instance
_crypto_data_service_instance: Optional[CryptoDataService] = None


async def get_crypto_data_service() -> CryptoDataService:
    """Get or create the global crypto data service instance."""
    global _crypto_data_service_instance
    
    if _crypto_data_service_instance is None:
        _crypto_data_service_instance = CryptoDataService()
        await _crypto_data_service_instance.initialize()
    
    return _crypto_data_service_instance