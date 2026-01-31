from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, Query, Request, Response

try:
    from ..services.news_service import crawl_and_store_article, get_news_articles, refresh_news_source
    from ..services.realtime_news_service import RealTimeNewsAggregator
    from ..cache import AsyncCache
    from .dependencies.rate_limiter import rate_limit
except ImportError:
    from services.news_service import crawl_and_store_article, get_news_articles, refresh_news_source
    from services.realtime_news_service import RealTimeNewsAggregator
    from cache import AsyncCache
    from api.dependencies.rate_limiter import rate_limit  # type: ignore


router = APIRouter(prefix="/news", tags=["news"])


@router.get("")
@rate_limit("news_list")
async def list_news(
    request: Request,
    response: Response,
    q: str | None = Query(None, description="Search query"),
    source: str | None = Query(None, description="Filter by source name"),
    limit: int = Query(20, ge=1, le=100, description="Number of articles per page"),
    page: int = Query(1, ge=1, description="Page number"),
    realtime: bool = Query(True, description="Use real-time aggregation"),
) -> Dict[str, Any]:
    """Get news articles with optional filtering and pagination."""
    
    if realtime:
        # Use real-time news aggregator
        cache = AsyncCache()
        await cache.initialize()
        
        cache_key = f"realtime_news:{q or 'all'}:{source or 'all'}:{limit}:{page}"
        cached_result = await cache.get(cache_key)
        
        if cached_result:
            return cached_result
        
        try:
            async with RealTimeNewsAggregator() as aggregator:
                all_articles = await aggregator.fetch_all_news(limit * 3)  # Fetch more for filtering
                
                # Apply filters
                filtered_articles = all_articles
                if q:
                    filtered_articles = [a for a in filtered_articles 
                                       if q.lower() in a.title.lower() or q.lower() in a.content.lower()]
                
                if source:
                    filtered_articles = [a for a in filtered_articles 
                                       if source.lower() in a.source.lower()]
                
                # Apply pagination
                total = len(filtered_articles)
                offset = (page - 1) * limit
                paginated_articles = filtered_articles[offset:offset + limit]
                
                # Convert to API format
                items = []
                for article in paginated_articles:
                    items.append({
                        "id": article.id,
                        "title": article.title,
                        "url": article.url,
                        "author": article.author,
                        "published_at": article.published_at.isoformat(),
                        "source": {"name": article.source},
                        "content_text": article.content,
                        "sentiment": article.sentiment,
                        "category": article.category,
                        "image_url": article.image_url
                    })
                
                result = {
                    "items": items,
                    "page": page,
                    "limit": limit,
                    "total": total,
                    "total_pages": (total + limit - 1) // limit,
                    "realtime": True,
                    "last_updated": datetime.utcnow().isoformat()
                }
                
                # Cache for 2 minutes (frequent updates)
                await cache.set(cache_key, result, ttl_seconds=120)
                return result
                
        except Exception as e:
            print(f"Real-time news error: {e}")
            # Fallback to regular news service
            pass
    
    # Fallback to existing news service
    return await get_news_articles(q=q, source=source, limit=limit, page=page)


@router.post("/refresh")
@rate_limit("news_refresh")
async def refresh_news(
    request: Request,
    response: Response,
    url: str = Query(..., description="URL to crawl and store"),
    background_tasks: BackgroundTasks = None,
) -> Dict[str, Any]:
    """Crawl a URL via Firecrawl and store the article."""
    return await crawl_and_store_article(url, background_tasks=background_tasks)


@router.get("/sources")
@rate_limit("news_list")
async def list_sources(request: Request, response: Response) -> Dict[str, Any]:
    """List configured news sources and their status."""
    try:
        from db import get_db  # type: ignore
        from models.news import NewsSource  # type: ignore
        
        db = next(get_db())
        sources = db.query(NewsSource).all()
        
        source_list = []
        for source in sources:
            source_list.append({
                "id": source.id,
                "name": source.name,
                "base_url": source.base_url,
                "enabled": source.enabled,
                "crawl_depth": source.crawl_depth,
                "last_checked_at": source.last_checked_at.isoformat() if source.last_checked_at else None
            })
        
        return {"sources": source_list}
        
    except Exception as e:
        return {"sources": [], "error": str(e)}


@router.get("/trending")
@rate_limit("news_list")
async def get_trending_topics(request: Request, response: Response) -> Dict[str, Any]:
    """Get trending cryptocurrency topics from real-time news analysis."""
    try:
        async with RealTimeNewsAggregator() as aggregator:
            trending = await aggregator.get_trending_topics()
            return {
                "trending_topics": trending,
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        return {
            "trending_topics": [],
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/sentiment")
@rate_limit("news_list")
async def get_market_sentiment(request: Request, response: Response) -> Dict[str, Any]:
    """Get overall market sentiment from recent news."""
    try:
        async with RealTimeNewsAggregator() as aggregator:
            sentiment_score = await aggregator.get_market_sentiment_score()
            
            # Convert to sentiment label
            if sentiment_score > 0.6:
                sentiment_label = "Bullish"
                sentiment_color = "green"
            elif sentiment_score < 0.4:
                sentiment_label = "Bearish" 
                sentiment_color = "red"
            else:
                sentiment_label = "Neutral"
                sentiment_color = "yellow"
            
            return {
                "sentiment_score": round(sentiment_score, 3),
                "sentiment_label": sentiment_label,
                "sentiment_color": sentiment_color,
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        return {
            "sentiment_score": 0.5,
            "sentiment_label": "Neutral",
            "sentiment_color": "yellow",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/stats")
@rate_limit("news_list")
async def news_stats(request: Request, response: Response) -> Dict[str, Any]:
    """Get news pipeline statistics."""
    try:
        # Get real-time stats
        async with RealTimeNewsAggregator() as aggregator:
            articles = await aggregator.fetch_all_news(100)
            
            # Calculate stats
            total_articles = len(articles)
            recent_articles = len([a for a in articles if (datetime.utcnow() - a.published_at).total_seconds() < 86400])
            
            # Sentiment breakdown
            sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
            for article in articles:
                if article.sentiment:
                    sentiment_counts[article.sentiment] = sentiment_counts.get(article.sentiment, 0) + 1
            
            # Source breakdown
            source_counts = {}
            for article in articles:
                source_counts[article.source] = source_counts.get(article.source, 0) + 1
            
            return {
                "total_articles": total_articles,
                "articles_last_24h": recent_articles,
                "sources_count": len(source_counts),
                "sentiment_breakdown": sentiment_counts,
                "top_sources": dict(sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
                "realtime": True,
                "timestamp": datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        # Fallback to database stats
        try:
            from db import get_db  # type: ignore
            from models.news import NewsSource, NewsArticle  # type: ignore
            from datetime import datetime, timedelta
            
            db = next(get_db())
            
            # Get total counts
            total_articles = db.query(NewsArticle).count()
            sources_count = db.query(NewsSource).count()
            
            # Get articles from last 24 hours
            yesterday = datetime.now() - timedelta(days=1)
            articles_last_24h = db.query(NewsArticle).filter(
                NewsArticle.fetched_at >= yesterday
            ).count()
            
            return {
                "total_articles": total_articles,
                "articles_last_24h": articles_last_24h,
                "sources_count": sources_count,
                "realtime": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as db_error:
            return {
                "total_articles": 0,
                "articles_last_24h": 0,
                "sources_count": 0,
                "realtime": False,
                "error": f"Database error: {db_error}",
                "timestamp": datetime.utcnow().isoformat()
            }

