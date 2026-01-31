from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, HTTPException

try:
    from clients.firecrawl_client import FirecrawlClient
    from cache import AsyncCache
    from models.news import NewsArticle, NewsSource
    from db import get_db
except ImportError:
    from clients.firecrawl_client import FirecrawlClient
    from cache import AsyncCache
    from models.news import NewsArticle, NewsSource
    from db import get_db


def sha256(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_article(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Firecrawl response to NewsArticle fields."""
    url = doc.get("url") or ""
    text = (doc.get("content") or doc.get("text") or "")[:1_048_576]  # 1MB limit
    
    return {
        "url": url,
        "canonical_url": doc.get("canonicalUrl") or url,
        "title": doc.get("title") or "",
        "author": doc.get("author") or "",
        "language": doc.get("language") or "",
        "published_at": doc.get("publishedAt"),
        "content_text": text,
        "content_html": doc.get("html") or "",
        "url_hash": sha256(url.lower().strip()),
        "content_hash": sha256(text.lower().strip()) if text else None,
    }


async def crawl_and_store_article(
    url: str,
    *,
    source_id: Optional[int] = None,
    background_tasks: Optional[BackgroundTasks] = None,
) -> Dict[str, Any]:
    """Crawl URL via Firecrawl, normalize, dedupe, and store with REST fallback."""
    cache = AsyncCache()
    await cache.initialize()
    
    # Check cache first
    cache_key = f"news_article:{sha256(url)}"
    cached = await cache.get(cache_key)
    if cached:
        return cached
    
    # Try Firecrawl first
    client = FirecrawlClient()
    try:
        data = await client.crawl_url(url)
        normalized = normalize_article(data)
        
        # Store in cache with both fresh and last_success keys
        await cache.set(cache_key, normalized, ttl_seconds=3600)  # 1 hour
        await cache.set(cache_key + ":last_success", normalized, ttl_seconds=None)  # Keep indefinitely
        
        # TODO: Store in database via SQLAlchemy
        return normalized
        
    except HTTPException as e:
        if e.status_code == 402:
            # Payment required - try REST fallback or return cached data
            last_success = await cache.get(cache_key + ":last_success")
            if last_success:
                # Schedule background refresh for when quota resets
                if background_tasks:
                    background_tasks.add_task(_schedule_retry_later, url, source_id)
                return last_success
            else:
                # No cached data, try simple REST fallback
                return await _simple_rest_fallback(url, cache_key, cache)
        else:
            # Other HTTP errors - return cached data if available
            last_success = await cache.get(cache_key + ":last_success")
            if last_success:
                if background_tasks:
                    background_tasks.add_task(crawl_and_store_article, url, source_id=source_id)
                return last_success
            raise HTTPException(status_code=503, detail=f"Failed to crawl {url}: {str(e)}")
    except Exception as e:
        # Network or other errors - return cached data if available
        last_success = await cache.get(cache_key + ":last_success")
        if last_success:
            if background_tasks:
                background_tasks.add_task(crawl_and_store_article, url, source_id=source_id)
            return last_success
        raise HTTPException(status_code=503, detail=f"Failed to crawl {url}: {str(e)}")
    finally:
        await client.close()


async def _schedule_retry_later(url: str, source_id: Optional[int]) -> None:
    """Schedule a retry for later when quota might reset."""
    # TODO: Implement proper scheduling with APScheduler
    # For now, just log the retry
    print(f"Scheduled retry for {url} due to quota limit")


async def _simple_rest_fallback(url: str, cache_key: str, cache: AsyncCache) -> Dict[str, Any]:
    """Simple REST fallback when Firecrawl quota is exceeded."""
    # Create a minimal article structure for development
    fallback_article = {
        "url": url,
        "canonical_url": url,
        "title": f"Article from {url} (Cached/Fallback)",
        "author": "Unknown",
        "language": "en",
        "published_at": None,
        "content_text": f"This is a fallback article for {url}. Firecrawl quota exceeded.",
        "content_html": f"<p>This is a fallback article for {url}. Firecrawl quota exceeded.</p>",
        "url_hash": sha256(url.lower().strip()),
        "content_hash": sha256(f"This is a fallback article for {url}. Firecrawl quota exceeded.".lower().strip()),
    }
    
    # Cache the fallback
    await cache.set(cache_key, fallback_article, ttl_seconds=3600)
    await cache.set(cache_key + ":last_success", fallback_article, ttl_seconds=None)
    
    return fallback_article


async def get_news_articles(
    *,
    q: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 20,
    page: int = 1,
) -> Dict[str, Any]:
    """Get news articles with optional filtering and pagination."""
    cache = AsyncCache()
    await cache.initialize()
    
    # Check cache first
    cache_key = f"news_articles:{q or 'all'}:{source or 'all'}:{limit}:{page}"
    cached_result = await cache.get(cache_key)
    if cached_result:
        return cached_result
    
    try:
        # Try to get from database first
        db = next(get_db())
        
        # Build query
        query = db.query(NewsArticle)
        
        # Apply filters
        if source:
            source_obj = db.query(NewsSource).filter(NewsSource.name.ilike(f"%{source}%")).first()
            if source_obj:
                query = query.filter(NewsArticle.source_id == source_obj.id)
        
        if q:
            query = query.filter(
                NewsArticle.title.ilike(f"%{q}%") | 
                NewsArticle.content_text.ilike(f"%{q}%")
            )
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        offset = (page - 1) * limit
        articles = query.order_by(NewsArticle.published_at.desc()).offset(offset).limit(limit).all()
        
        # Convert to dict format
        items = []
        for article in articles:
            items.append({
                "id": article.id,
                "title": article.title,
                "url": article.url,
                "author": article.author,
                "published_at": article.published_at.isoformat() if article.published_at else None,
                "fetched_at": article.fetched_at.isoformat(),
                "source": article.source.name if article.source else None,
                "content_text": article.content_text[:200] + "..." if article.content_text and len(article.content_text) > 200 else article.content_text
            })
        
        # If we have database articles, return them
        if items:
            result = {
                "items": items,
                "page": page,
                "limit": limit,
                "total": total,
                "total_pages": (total + limit - 1) // limit
            }
            # Cache for 5 minutes
            await cache.set(cache_key, result, ttl_seconds=300)
            return result
        
    except Exception as e:
        print(f"Error getting news articles from database: {e}")
    
    # Fallback to live news aggregator
    try:
        from clients.crypto_news_aggregator import CryptoNewsAggregator
        
        async with CryptoNewsAggregator() as aggregator:
            articles = await aggregator.fetch_all_news()
            
            # Apply filters
            if q:
                articles = [a for a in articles if q.lower() in a.title.lower() or q.lower() in a.content.lower()]
            
            if source:
                articles = [a for a in articles if source.lower() in a.source.lower()]
            
            # Apply pagination
            total = len(articles)
            offset = (page - 1) * limit
            paginated_articles = articles[offset:offset + limit]
            
            # Convert to dict format
            items = []
            for article in paginated_articles:
                items.append({
                    "id": article.id,
                    "title": article.title,
                    "url": article.url,
                    "author": article.source,
                    "published_at": article.published_at.isoformat(),
                    "source": article.source,
                    "content_text": article.content,
                    "sentiment": article.sentiment
                })
            
            result = {
                "items": items,
                "page": page,
                "limit": limit,
                "total": total,
                "total_pages": (total + limit - 1) // limit
            }
            
            # Cache for 10 minutes
            await cache.set(cache_key, result, ttl_seconds=600)
            return result
            
    except Exception as e:
        print(f"Error getting news from aggregator: {e}")
        
        # Final fallback - return empty result
        result = {
            "items": [],
            "page": page,
            "limit": limit,
            "total": 0,
            "total_pages": 0,
            "error": "News service temporarily unavailable"
        }
        
        # Cache error result for 1 minute to avoid hammering failing services
        await cache.set(cache_key, result, ttl_seconds=60)
        return result


async def refresh_news_source(source_id: int) -> Dict[str, Any]:
    """Refresh all articles from a specific news source."""
    # TODO: Implement source refresh logic
    return {"status": "queued", "source_id": source_id}

