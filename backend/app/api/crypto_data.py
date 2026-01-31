from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Query

try:
    from services.crypto_data_service import get_crypto_data_service
except ImportError:
    from services.crypto_data_service import get_crypto_data_service


router = APIRouter(prefix="/crypto", tags=["crypto-data"])


@router.get("/dashboard")
async def get_crypto_dashboard(
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """Get comprehensive crypto dashboard data from all free sources.
    
    This endpoint aggregates data from:
    - CoinGecko (prices, market data)
    - Direct web crawling (CoinDesk, CoinTelegraph, Decrypt, The Block)
    - Reddit and CryptoPanic (sentiment)
    """
    service = await get_crypto_data_service()
    return await service.get_crypto_dashboard_data(background_tasks=background_tasks)


@router.get("/news")
async def get_crypto_news(
    source: Optional[str] = Query(None, description="News source: coindesk, cointelegraph, decrypt, theblock"),
    limit: int = Query(20, description="Maximum number of articles to return"),
) -> Dict[str, Any]:
    """Get crypto news from free sources.
    
    Sources available:
    - coindesk: CoinDesk news
    - cointelegraph: CoinTelegraph news  
    - decrypt: Decrypt news
    - theblock: The Block news
    - all: All sources combined (default)
    """
    service = await get_crypto_data_service()
    return await service.get_crypto_news(source=source, limit=limit)


@router.get("/sentiment")
async def get_crypto_sentiment(
    source: Optional[str] = Query(None, description="Sentiment source: reddit, cryptopanic, all"),
) -> Dict[str, Any]:
    """Get crypto sentiment data from social media.
    
    Sources available:
    - reddit: Reddit crypto discussions
    - cryptopanic: CryptoPanic news sentiment
    - all: All sources combined with summary (default)
    """
    service = await get_crypto_data_service()
    return await service.get_crypto_sentiment(source=source)


@router.get("/twitter")
async def get_crypto_twitter(
    hashtag: Optional[str] = Query(None, description="Specific hashtag to search (e.g., #Bitcoin)"),
    account: Optional[str] = Query(None, description="Specific Twitter account to crawl"),
    max_tweets: int = Query(20, description="Maximum number of tweets to return"),
) -> Dict[str, Any]:
    """Get crypto Twitter sentiment and posts.
    
    Parameters:
    - hashtag: Search for specific crypto hashtags (#Bitcoin, #Ethereum, etc.)
    - account: Crawl specific Twitter accounts (elonmusk, VitalikButerin, etc.)
    - max_tweets: Maximum number of tweets to return
    
    If no parameters provided, returns general crypto Twitter sentiment.
    """
    service = await get_crypto_data_service()
    
    if hashtag:
        tweets = await service.twitter_client.crawl_twitter_hashtag(hashtag, max_tweets)
        return {
            "tweets": tweets,
            "hashtag": hashtag,
            "total": len(tweets),
            "source": "Twitter via Nitter (Free)"
        }
    elif account:
        tweets = await service.twitter_client.crawl_twitter_account(account, max_tweets)
        return {
            "tweets": tweets,
            "account": account,
            "total": len(tweets),
            "source": "Twitter via Nitter (Free)"
        }
    else:
        return await service._get_twitter_sentiment()


@router.get("/prices")
async def get_crypto_prices() -> Dict[str, Any]:
    """Get current crypto prices from CoinGecko free API."""
    service = await get_crypto_data_service()
    return await service._get_price_data()


@router.get("/market-summary")
async def get_market_summary() -> Dict[str, Any]:
    """Get crypto market summary and trends."""
    service = await get_crypto_data_service()
    return await service._get_market_summary()


@router.get("/real-social")
async def get_real_social_data() -> Dict[str, Any]:
    """Get real crypto social media data from multiple sources.
    
    This endpoint aggregates real data from:
    - Reddit crypto discussions (r/cryptocurrency, r/bitcoin, r/ethereum, etc.)
    - CryptoPanic news and sentiment
    - RSS feeds from major crypto news sites (CoinDesk, CoinTelegraph, Decrypt)
    
    All data is real and crawled from live sources.
    """
    service = await get_crypto_data_service()
    return await service._get_real_social_sentiment()


@router.get("/sources")
async def get_available_sources() -> Dict[str, Any]:
    """Get information about available data sources."""
    return {
        "news_sources": {
            "coindesk": "CoinDesk - Major crypto news",
            "cointelegraph": "CoinTelegraph - Crypto news and analysis", 
            "decrypt": "Decrypt - Web3 and crypto news",
            "theblock": "The Block - Crypto and blockchain news"
        },
        "sentiment_sources": {
            "reddit": "Reddit crypto discussions and sentiment",
            "cryptopanic": "CryptoPanic news and community sentiment",
            "twitter": "Twitter crypto discussions via Nitter (free)",
            "real_social": "Real data from Reddit, CryptoPanic, RSS news feeds"
        },
        "price_sources": {
            "coingecko": "CoinGecko free API for prices and market data"
        },
        "all_free": True,
        "no_api_keys_required": True,
        "rate_limits": {
            "coingecko": "10-50 requests/minute (free tier)",
            "reddit": "60 requests/minute",
            "direct_crawling": "Respectful delays between requests"
        }
    }
